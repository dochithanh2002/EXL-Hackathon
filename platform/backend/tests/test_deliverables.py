"""Bundle regressions: environment identity and the exact same checks as the approval gate."""
import hashlib
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from controlplane.deliverables import PARITY, VERIFY, build_bundle
from controlplane.environments import Environments
from controlplane.validation import compare
from tests.test_schemas import spec as make_spec
from tests.test_validation import COEFS, PREDS, PY_FIT, pair


@pytest.mark.parametrize("changed", ["uv.lock", "renv.lock"])
def test_bundle_refuses_lockfiles_changed_since_approval(tmp_path, changed):
    envs = Environments(tmp_path / "envs")
    for name, path in envs.lockfiles.items():
        path.parent.mkdir(parents=True)
        path.write_text(f"original {name}")
    envs.python_lockfile.with_name("pyproject.toml").write_text("[project]\n")
    record = SimpleNamespace(approved=None, environment_version=envs.version())
    envs.lockfiles[changed].write_text("updated environment")
    dest = tmp_path / "bundle"
    with pytest.raises(ValueError, match="lockfiles no longer match the approved validation"):
        build_bundle(dest, record, "report", envs)
    assert not dest.exists() and not dest.with_suffix(".zip").exists()


@pytest.mark.parametrize("case", ["identical", "reordered_terms", "fit_statistics", "response", "kpi"])
def test_standalone_verifier_repeats_approval_checks_without_backend_dependencies(tmp_path, case):
    coefs, preds, fit = list(COEFS), [list(p) for p in PREDS], dict(PY_FIT)
    if case == "reordered_terms":
        coefs.reverse()
    elif case == "fit_statistics":
        fit["llf"] += 100
    elif case == "response":
        preds[0][2] = 0.0
    elif case == "kpi":
        preds[0][3] += 1e-9             # prediction still passes; the stricter requested KPI tolerance does not
    baseline, output = pair(tmp_path, coefs=coefs, preds=preds, py_fit=fit)
    spec = make_spec(kpis=["auc", "gini", "log_loss", "actual_expected", "aic"], tolerances={"kpi_abs": 1e-12})
    expected = compare(baseline, output, spec)
    assert expected.passed == (case in ("identical", "reordered_terms"))

    root = tmp_path / "bundle"
    root.mkdir()
    shutil.copy(VERIFY, root / "verify.py")
    shutil.copy(PARITY, root / "parity.py")
    shutil.copytree(baseline, root / "reference")
    (root / "spec.json").write_text(spec.model_dump_json())
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("synthetic fixture\n")
    (root / "verify.json").write_text(json.dumps({"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}))
    # Substitute only the harness execution; exercise the shipped verifier in its own process with site packages
    # disabled. Its inputs are synthetic capture files, and the backend must not be needed by the bundle.
    (root / "harness.py").write_text(
        "import json, shutil, sys\n"
        f"shutil.copytree({str(output)!r}, json.load(open(sys.argv[1]))['out'])\n")
    result = subprocess.run([sys.executable, "-S", "verify.py", str(dataset)], cwd=root,
                            env={}, capture_output=True, text=True)
    assert result.returncode == (0 if expected.passed else 1), result.stdout + result.stderr
    lines = [line.split("  ")[:2] for line in result.stdout.splitlines() if line.startswith(("pass  ", "FAIL  "))]
    assert lines == [["pass" if c.passed else "FAIL", c.name] for c in expected.checks]
    assert [line.removeprefix("note  ") for line in result.stdout.splitlines() if line.startswith("note  ")] == (
        expected.notes)
    if case == "kpi":
        assert all(c.passed for c in expected.checks if not c.name.startswith("kpi "))
        assert any(not c.passed for c in expected.checks if c.name.startswith("kpi "))
