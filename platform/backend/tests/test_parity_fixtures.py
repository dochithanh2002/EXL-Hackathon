"""The parity fixtures that make a support-matrix pair proven: for each pair naming a fixture, R's baseline runs on
the fixture's synthetic dataset, the known-good candidate runs against it, and every parity check passes."""
import json

import pytest

from controlplane import support
from controlplane.runner import Limits
from controlplane.schemas import ModelSpec
from controlplane.translation import run_candidate
from controlplane.workflow import Step
from tests.test_baseline import jobs, project_at_baseline  # noqa: F401

FIXTURES = sorted({p.fixture for p in support.load().pairs if p.fixture})


def test_every_fixture_directory_is_named_by_a_pair():
    on_disk = {d.name for d in support.FIXTURE_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")}
    assert on_disk == set(FIXTURES)


def test_only_the_fixture_families_are_proven():
    m = support.load()
    proven = {(p.r, f) for p in m.pairs if p.proven for f in p.families or [None]}
    assert proven == {("stats::glm", "binomial"), ("stats::glm", "poisson")}
    assert not m.lookup("stats::glm", "Gamma").proven


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_candidate_reaches_parity_with_r(name, jobs, tmp_path):
    d = support.FIXTURE_DIR / name
    spec = ModelSpec.model_validate(json.loads((d / "spec.json").read_text()))
    [dataset] = d.glob("*.csv")
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {dataset.name: dataset, "model.R": b""})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert s.step == Step.TRANSLATION, s.detail
    baseline = jobs.settings.data_dir / jobs.store.evidence(pid, s.baseline_id)["body"]["dir"]
    v = run_candidate(tmp_path / "cand", (d / "candidate.py").read_text(), spec, baseline, s.files,
                      jobs.settings.project_dir(pid) / "inputs", jobs.envs, Limits(), lambda p: None)
    assert v.passed, [(c.name, c.status, c.max_abs_diff, c.reason) for c in v.checks if not c.passed]
    by_name = {c.name: c for c in v.checks}
    assert by_name["coefficient estimates"].max_abs_diff < 1e-9
    assert by_name["coefficient standard errors"].max_abs_diff < 1e-9
    assert by_name["predictions"].max_abs_diff < 1e-9
    if spec.validation_plan.fairness:                       # protected attributes travel from R's set frames
        f = v.evaluation["checks"]["fairness"]
        assert f["status"] == "executed", f["needs"]
        assert set(f["results"]["columns"]) == set(spec.validation_plan.fairness.columns)
        assert f["criterion"]["outcome"] in ("met", "not met")
