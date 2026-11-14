"""Step 4 below the agent: a candidate runs in the stable Python environment against R's sets and is validated
against the frozen R baseline."""
import json
import threading

import pytest

from controlplane.runner import Limits, StepCancelled
from controlplane.schemas import ModelSpec
from controlplane.translation import Translator, run_candidate
from controlplane.workflow import Step
from tests.test_baseline import CLAIMS, FIXTURES, M4_DIR, jobs, poisson_spec, project_at_baseline  # noqa: F401

POISSON_PY = '''import numpy as np
import pandas as pd
import statsmodels.api as sm

RESPONSE = "claims"
AREAS = ["A", "B", "C"]


def prepare(path, sets):
    df = pd.read_csv(path)
    df.index = pd.RangeIndex(1, len(df) + 1)
    return {name: df.loc[rows] for name, rows in sets.items()}


def design(frame):
    X = pd.DataFrame({"(Intercept)": 1.0}, index=frame.index)
    for level in AREAS[1:]:
        X["area" + level] = (frame["area"] == level).astype(float)
    X["age"] = frame["age"].astype(float)
    return X


class RPoisson(sm.families.Poisson):
    def starting_mu(self, y):                      # glm.fit's mustart, so IRLS follows R's iterations
        return y + 0.1


def fit(train):
    return sm.GLM(train[RESPONSE].astype(float), design(train), family=RPoisson(),
                  offset=np.log(train["exposure"])).fit()


def predict(result, frame):
    return pd.Series(np.asarray(result.predict(design(frame), offset=np.log(frame["exposure"]))), index=frame.index)
'''


@pytest.fixture
def claims(jobs):
    """A project whose Poisson baseline (offset, factor, random train/test split) has run."""
    pid = project_at_baseline(jobs.store, jobs.settings, poisson_spec(), {"claims.csv": CLAIMS.encode(), "m.R": b""})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert s.step == Step.TRANSLATION, s.detail
    jobs.pid, jobs.spec = pid, poisson_spec()
    jobs.baseline_dir = jobs.settings.data_dir / jobs.store.evidence(pid, s.baseline_id)["body"]["dir"]
    return jobs


def run(j, module, tmp_path, limits=Limits(), on_start=lambda p: None):
    return run_candidate(tmp_path / "cand", module, j.spec, j.baseline_dir, j.store.state(j.pid).files,
                         j.settings.project_dir(j.pid) / "inputs", j.envs, limits, on_start)


def test_a_faithful_candidate_passes_with_r_rows_offset_and_test_set(claims, tmp_path):
    v = run(claims, POISSON_PY, tmp_path)
    assert v.passed, [c for c in v.checks if not c.passed]
    by_name = {c.name: c for c in v.checks}
    assert by_name["coefficient max rel diff"].value < 1e-12 and by_name["std error max rel diff"].value < 1e-12
    assert by_name["prediction max abs diff"].value < 1e-10
    assert {"fit deviance rel diff", "fit log_lik rel diff", "fit aic rel diff"} <= set(by_name)
    assert set(v.kpis) == {"train", "test"}
    assert "auc is not defined for the train set's response; not compared" in v.notes      # count response
    preds = (tmp_path / "cand" / "output" / "predictions.csv").read_text().splitlines()
    assert len(preds) == 201 and preds[0] == "row,set,y,response"

    # the candidate's working directory holds the dataset and R's sets, never R's estimates
    assert sorted(p.name for p in (tmp_path / "cand" / "inputs").iterdir()) == ["claims.csv", "split.csv"]


def test_statsmodels_own_starting_values_cost_standard_error_parity(claims, tmp_path):
    """Rulebook R-EST-02: the default start stops IRLS at a different iterate than R."""
    v = run(claims, POISSON_PY.replace("family=RPoisson()", "family=sm.families.Poisson()"), tmp_path)
    by_name = {c.name: c.value for c in v.checks}
    assert by_name["coefficient max rel diff"] < 1e-10 and 1e-8 < by_name["std error max rel diff"] < 1e-6


def test_a_drifted_candidate_fails_on_the_checks_that_moved(claims, tmp_path):
    drifted = POISSON_PY.replace("offset=np.log(train[\"exposure\"])", "offset=np.log(train[\"exposure\"]) * 0.99")
    v = run(claims, drifted, tmp_path)
    failed = {c.name for c in v.checks if not c.passed}
    assert not v.passed and {"coefficient max rel diff", "prediction max abs diff"} <= failed
    assert "coefficient names missing or extra" not in failed and "rows only on one side" not in failed
    assert v.differences["worst_terms"][0]["estimate_rel"] > 1e-6


def test_wrong_terms_and_imputed_rows_are_named(claims, tmp_path):
    renamed = POISSON_PY.replace('"(Intercept)"', '"Intercept"')
    v = run(claims, renamed, tmp_path)
    assert v.differences["missing_terms"] == ["(Intercept)"] and v.differences["extra_terms"] == ["Intercept"]

    nan_age = POISSON_PY.replace('X["age"] = frame["age"].astype(float)',
                                 'X["age"] = frame["age"].astype(float).where(frame.index != 7)')
    v = run(claims, nan_age, tmp_path / "2")
    failed = {c.name for c in v.checks if not c.passed}
    assert "prediction NA pattern differing" in failed
    assert v.differences["prediction_na_differs_rows"][0]["row"] == 7


@pytest.mark.parametrize("change, stage, message", [
    ('return {name: df.loc[rows] for name, rows in sets.items()}',
     "checking prepare()'s frames", "must be indexed by exactly the 140 rows R put in the train set"),
    ('df.index = pd.RangeIndex(1, len(df) + 1)', "prepare()", "not in index"),
    ('return sm.GLM(', "fit()", "boom"),
])
def test_a_candidate_that_breaks_the_contract_fails_validation_with_the_stage(claims, tmp_path, change, stage,
                                                                               message):
    replacement = {
        'return {name: df.loc[rows] for name, rows in sets.items()}':
            'return {name: df.loc[rows[:-1]] for name, rows in sets.items()}',
        'df.index = pd.RangeIndex(1, len(df) + 1)': 'df.index = pd.RangeIndex(0, len(df))',
        'return sm.GLM(': 'raise ValueError("boom")\n    return sm.GLM(',
    }[change]
    v = run(claims, POISSON_PY.replace(change, replacement), tmp_path)
    assert (v.passed, v.stage, v.error["stage"]) == (False, "runtime", stage)
    assert message in v.error["message"] and v.checks == []
    assert "Traceback" in (tmp_path / "cand" / "step.log").read_text()


def test_a_candidate_over_its_wall_clock_limit_fails_validation(claims, tmp_path):
    slow = POISSON_PY.replace("def fit(train):\n", "def fit(train):\n    while True:\n        pass\n")
    v = run(claims, slow, tmp_path, Limits(wall_s=3))
    assert v.stage == "runtime" and "wall-clock limit" in v.error["message"]


def test_rebinding_the_dataset_path_cannot_read_another_file(claims, tmp_path):
    module = POISSON_PY.replace("df = pd.read_csv(path)", 'path = "inputs/split.csv"\n    df = pd.read_csv(path)')
    v = run(claims, module, tmp_path)
    assert v.stage == "runtime" and v.error["stage"] == "prepare()"
    assert v.error["message"] == "only the dataset may be read"


def test_fit_cannot_return_a_candidate_defined_results_object(claims, tmp_path):
    module = POISSON_PY.replace("def fit(train):", "class FakeResults:\n    pass\n\ndef fit(train):\n    return FakeResults()")
    v = run(claims, module, tmp_path)
    assert v.stage == "runtime" and v.error["stage"] == "reading the fitted results"
    assert v.error["message"] == "fit() must return a statsmodels results object"


def test_cancel_kills_the_candidate(claims, tmp_path):
    slow = POISSON_PY.replace("def fit(train):\n", "def fit(train):\n    while True:\n        pass\n")
    with pytest.raises(StepCancelled):
        run(claims, slow, tmp_path, on_start=lambda p: threading.Timer(1.0, p.cancel).start())


def test_the_translator_sees_names_and_metadata_but_not_r_estimates(claims):
    t = Translator(claims.spec, claims.baseline_dir, "claims.csv", claims.envs, lambda: {})
    prompt = t.kickoff()
    assert '["(Intercept)", "areaB", "areaC", "age"]' in prompt and "statsmodels==0.15.0" in prompt
    assert "prepare(path: str, sets" in prompt and "fit <- glm(claims ~ area" in prompt
    estimates = [line.split(",")[1] for line in (claims.baseline_dir / "coefficients.csv").read_text().splitlines()[1:]]
    assert not any(e.strip('"') in prompt for e in estimates)
    assert t.check({"module": "import os\n", "notes": []})[1][0].startswith(
        "guardrail: line 1: module 'os' is not available")
    assert t.check({"module": POISSON_PY})[1] == ["notes: Field required"]
    assert t.check({"module": POISSON_PY, "notes": []})[1] == []


@pytest.mark.skipif(not (M4_DIR / "data" / "cs-training.csv").exists(), reason=f"M4 not found at {M4_DIR}")
def test_m4_known_good_candidate_reaches_exact_parity(jobs, tmp_path):
    spec = ModelSpec.model_validate_json((FIXTURES / "m4" / "spec.json").read_text())
    original = M4_DIR / "code" / "givemesomecredit_logistic_notebook.R"
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {
        "cs-training.csv": M4_DIR / "data" / "cs-training.csv", original.name: original})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    baseline_dir = jobs.settings.data_dir / jobs.store.evidence(pid, s.baseline_id)["body"]["dir"]
    v = run_candidate(tmp_path / "cand", (FIXTURES / "m4" / "candidate.py").read_text(), spec, baseline_dir, s.files,
                      jobs.settings.project_dir(pid) / "inputs", jobs.envs)
    assert v.passed, json.dumps([c.model_dump() for c in v.checks if not c.passed])
    worst = max(c.value for c in v.checks if c.name in ("coefficient max rel diff", "std error max rel diff"))
    assert worst < 1e-10
    assert set(v.kpis["train"]) == {"auc", "gini", "log_loss"}
