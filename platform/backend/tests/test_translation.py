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
    assert by_name["coefficient estimates"].max_abs_diff < 1e-10 and by_name["coefficient estimates"].value < 1e-3
    assert by_name["coefficient standard errors"].max_abs_diff < 1e-10
    assert by_name["predictions"].max_abs_diff < 1e-10 and by_name["predictions"].compared == 200
    assert {"fit deviance", "fit log_lik", "fit aic", "Python fit converged", "valid predictions on test"} <= set(by_name)
    assert set(v.metrics) == {"train", "test"} and set(v.metrics["test"]) == {"rmse", "deviance", "actual_expected"}
    preds = (tmp_path / "cand" / "output" / "predictions.csv").read_text().splitlines()
    assert len(preds) == 201 and preds[0] == "row,set,key,y,prediction"
    # deterministic summaries for the Diagnostician and the report
    assert v.design["fitted_rows"] == {"r": 140, "python": 140}
    assert v.design["reference_levels"]["area"] == {"r": "A", "python": "A"}
    assert v.design["offset"]["differences"]["sum"] == pytest.approx(0, abs=1e-9)
    assert v.assessment["python"]["decision_performance"]["status"] == "not assessed"     # count response
    assert v.assessment["python"]["stability"]["results"]["psi_vs_development"]["test"] is not None

    # the candidate's working directory holds the dataset and R's sets, never R's estimates
    assert sorted(p.name for p in (tmp_path / "cand" / "inputs").iterdir()) == ["claims.csv", "split.csv"]


def test_statsmodels_own_starting_values_cost_standard_error_parity(claims, tmp_path):
    """Rulebook R-EST-02: the default start stops IRLS at a different iterate than R."""
    v = run(claims, POISSON_PY.replace("family=RPoisson()", "family=sm.families.Poisson()"), tmp_path)
    by_name = {c.name: c for c in v.checks}
    se = by_name["coefficient standard errors"]
    assert by_name["coefficient estimates"].max_abs_diff < 1e-10 and 1e-10 < se.max_abs_diff < 1e-7


def test_a_drifted_candidate_fails_on_the_checks_that_moved(claims, tmp_path):
    drifted = POISSON_PY.replace("offset=np.log(train[\"exposure\"])", "offset=np.log(train[\"exposure\"]) * 0.99")
    v = run(claims, drifted, tmp_path)
    failed = {c.name for c in v.checks if not c.passed}
    assert not v.passed and {"coefficient estimates", "predictions"} <= failed
    assert "coefficient names missing or extra" not in failed and "alignment keys only on one side" not in failed
    assert v.differences["worst_terms"][0]["estimate_score"] > 1


def test_wrong_terms_and_imputed_rows_are_named(claims, tmp_path):
    renamed = POISSON_PY.replace('"(Intercept)"', '"Intercept"')
    v = run(claims, renamed, tmp_path)
    assert v.differences["missing_terms"] == ["(Intercept)"] and v.differences["extra_terms"] == ["Intercept"]

    nan_age = POISSON_PY.replace('X["age"] = frame["age"].astype(float)',
                                 'X["age"] = frame["age"].astype(float).where(frame.index != 7)')
    v = run(claims, nan_age, tmp_path / "2")
    failed = {c.name for c in v.checks if not c.passed}
    assert "prediction missingness differing" in failed
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
    assert v.error["message"] == "only the project's data files may be read"


def test_a_results_object_without_coefficients_or_convergence_cannot_pass(claims, tmp_path):
    """Any results object may be returned, but what it does not report is unavailable, never passed."""
    module = POISSON_PY.replace("def fit(train):", "class Bare:\n    def predict(self, X, offset=None):\n"
                                "        return np.exp(offset)\n\ndef fit(train):\n    return Bare()")
    v = run(claims, module, tmp_path)
    c = {x.name: x for x in v.checks}
    assert v.stage == "parity" and not v.passed
    assert c["coefficient names missing or extra"].value == 4 and c["coefficient estimates"].status == "unavailable"
    assert c["Python fit converged"].status == "unavailable" and c["fit deviance"].status == "unavailable"


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
    assert "Offset: log(exposure)." in prompt and "Prediction: scalar, count in claims per policy" in prompt
    assert t.check({"module": "from sklearn.linear_model import PoissonRegressor\n" + POISSON_PY, "notes": []})[1][0] \
        == "guardrail: line 1: scikit-learn is excluded: implement the estimator with statsmodels, scipy or numpy"


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
    worst = max(c.max_abs_diff for c in v.checks if c.name in ("coefficient estimates", "coefficient standard errors"))
    assert worst < 1e-10
    assert set(v.metrics["train"]) == {"auc", "gini", "log_loss"}
    assert v.assessment["python"]["stability"]["status"] == "not assessed"                 # no holdout, no periods
    checks = v.evaluation["checks"]
    assert {k: c["status"] for k, c in checks.items()} == {
        "baseline_comparison": "executed", "calibration": "partially executed", "subgroup_performance": "executed",
        "stability": "partially executed", "sensitivity": "executed", "assumptions": "executed"}, json.dumps(
        {k: c["needs"] for k, c in checks.items()})
    perf = checks["baseline_comparison"]["results"]["sets"]["train"]
    assert perf["auc"]["python"] > 0.6 > perf["auc"]["constant"] and perf["log_loss"]["python_vs_constant"] > 0
    cal = checks["calibration"]["results"]["sets"]["train"]["value"]["python"]
    assert cal["observed_expected"] == pytest.approx(1, abs=1e-6)        # a logit with an intercept, in sample
    cv = checks["stability"]["results"]["cross_validation"]
    assert len(cv["folds"]) == 5 and cv["summary"]["auc"]["sd"] < 0.05
    up = checks["sensitivity"]["results"]["columns"]["RevolvingUtilizationOfUnsecuredLines"]["sets"]["train"]
    assert up["value"]["up"]["mean_change"] != 0


MNLOGIT_PY = '''import numpy as np
import pandas as pd
import statsmodels.api as sm

RESPONSE = "band"
KEYS = ["A", "B", "C"]


def prepare(path, sets):
    df = pd.read_csv(path)
    df.index = pd.RangeIndex(1, len(df) + 1)
    df[RESPONSE] = df[RESPONSE].astype(str)
    return {name: df.loc[rows] for name, rows in sets.items()}


def design(frame):
    return pd.DataFrame({"(Intercept)": 1.0, "x": frame["x"].astype(float),
                         "regionS": (frame["region"] == "S").astype(float)}, index=frame.index)


def fit(train):
    y = pd.Categorical(train[RESPONSE], categories=KEYS).codes
    return sm.MNLogit(y, design(train)).fit(method="newton", tol=1e-12, maxiter=100, disp=False)


def coefficients(result):
    rows = {}
    for j, level in enumerate(KEYS[1:]):
        for term in result.params.index:
            rows[f"{level}:{term}"] = (result.params.iloc[:, j][term], result.bse.iloc[:, j][term])
    return pd.DataFrame(rows, index=["estimate", "std_error"]).T


def predict(result, frame):
    return pd.DataFrame(np.asarray(result.predict(design(frame))), index=frame.index, columns=KEYS)
'''


def test_class_probabilities_from_any_results_object_are_compared_by_row_and_class(jobs, tmp_path):
    from tests.test_baseline import GROUPS
    from tests.test_schemas import spec as make_spec
    r = ('df <- read.csv("inputs/groups.csv")\ndf$band <- factor(df$band)\ntrain <- df[df$id <= 90, ]\n'
         'test <- df[df$id > 90, ]\nfit <- nnet::multinom(band ~ x + region, data = train, trace = FALSE, '
         'reltol = 1e-14, maxit = 1000)\n')
    loose = {"atol": 1e-4, "rtol": 1e-4}
    spec = make_spec(r_function="nnet::multinom", python_implementation="statsmodels.api.MNLogit", family=None,
                     link=None, formula="band ~ x + region", response="band", response_type="categorical",
                     support_tier="approximate", metrics=["log_loss", "brier", "aic"], r_packages=["nnet"], baseline_r=r,
                     tolerances={"coefficients": loose, "std_errors": {"atol": 1e-3, "rtol": 1e-2},
                                 "predictions": loose, "fit_statistics": loose, "metrics": loose},
                     prediction={"kind": "class_probability", "quantity": "probability", "units": "band probability",
                                 "keys": ["A", "B", "C"], "alignment_keys": ["row", "class"],
                                 "r_predict": 'predict(model, newdata = newdata, type = "probs")'})
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {"groups.csv": GROUPS.encode(), "m.R": b""})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert s.step == Step.TRANSLATION, s.detail
    baseline_dir = jobs.settings.data_dir / jobs.store.evidence(pid, s.baseline_id)["body"]["dir"]
    v = run_candidate(tmp_path / "cand", MNLOGIT_PY, spec, baseline_dir, s.files,
                      jobs.settings.project_dir(pid) / "inputs", jobs.envs)
    by_name = {c.name: c for c in v.checks}
    assert v.passed, json.dumps([c.model_dump() for c in v.checks if not c.passed])
    assert by_name["predictions"].compared == 120 * 3 and by_name["alignment keys only on one side"].value == 0
    assert set(v.metrics["test"]) == {"log_loss", "brier"}
    assert by_name["aic on train"].passed
    assert v.assessment["python"]["decision_performance"]["status"] == "not assessed"
    assert set(v.assessment["python"]["calibration"]["results"]["sets"]["test"]) == {"A", "B", "C"}


QUANTILE_PY = '''import numpy as np
import pandas as pd
from scipy.optimize import linprog

KEYS = ["0.5", "0.9"]

def prepare(path, sets):
    df = pd.read_csv(path, float_precision="round_trip")
    df.index = pd.RangeIndex(1, len(df) + 1)
    return {name: df.loc[rows] for name, rows in sets.items()}

def design(frame):
    return np.column_stack([np.ones(len(frame)), frame["x"]])

class Fit:
    pass

def fit(train):
    result = Fit()
    X, y = design(train), train["y"].to_numpy()
    n, k = X.shape
    result.fits = []
    for key in KEYS:
        tau = float(key)
        cost = np.concatenate([np.zeros(k), np.full(n, tau), np.full(n, 1-tau)])
        constraints = np.column_stack([X, np.eye(n), -np.eye(n)])
        result.fits.append(linprog(cost, A_eq=constraints, b_eq=y,
                                  bounds=[(None, None)] * k + [(0, None)] * (2*n), method="highs"))
    result.converged = all(f.success for f in result.fits)
    return result

def coefficients(result):
    rows = {}
    for i, term in enumerate(["(Intercept)", "x"]):
        for key, f in zip(KEYS, result.fits):
            rows[f"{term}:tau= {key}"] = (f.x[i], np.nan)
    return pd.DataFrame(rows, index=["estimate", "std_error"]).T

def predict(result, frame):
    return pd.DataFrame({key: design(frame) @ f.x[:2] for key, f in zip(KEYS, result.fits)}, index=frame.index)
'''


@pytest.mark.parametrize("solver_warning", [False, True])
def test_quantile_migration_passes_with_solver_specific_convergence(jobs, tmp_path, solver_warning):
    import math
    from tests.test_schemas import spec as make_spec
    # Exactly representable inputs keep this regression about solvers, not CSV parser rounding.
    data = "x,y\n" + "\n".join(
        f"{i/8},{round((2+.3*i/8+math.sin(i/8*1.37))*1024)/1024}" for i in range(1, 81))
    r = ('df <- read.csv("inputs/quantiles.csv")\ntrain <- df\n'
         'fit <- quantreg::rq(y ~ x, tau = c(0.5, 0.9), method = "fn", data = train)\n')
    if solver_warning:
        r += 'warning("solver did not converge")\n'
    spec = make_spec(r_function="quantreg::rq", python_implementation="scipy.optimize.linprog", family=None,
                     link=None, response="y", response_type="continuous", formula="y ~ x", event=None,
                     metrics=["pinball_loss"], r_packages=["quantreg"], baseline_r=r,
                     baseline_objects={"model": "fit", "train": "train", "test": None},
                     convergence={"mode": "expression",
                                  "r_check": 'identical(model$method, "fn") && length(fit_warnings) == 0 && all(is.finite(coef(model)))',
                                  "rationale": "rq's Frisch-Newton solver signals failure through warnings or errors; require a finite solution and no warnings"},
                     prediction={"kind": "quantile", "quantity": "continuous", "units": "response units",
                                 "keys": ["0.5", "0.9"], "alignment_keys": ["row", "quantile"],
                                 "r_predict": "predict(model, newdata = newdata)"})
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {"quantiles.csv": data.encode(), "m.R": b""})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert s.step == Step.TRANSLATION, s.detail
    baseline = jobs.settings.data_dir / jobs.store.evidence(pid, s.baseline_id)["body"]["dir"]
    assert json.loads((baseline / "meta.json").read_text())["converged"] is (not solver_warning)
    v = run_candidate(tmp_path / "quantile", QUANTILE_PY, spec, baseline, s.files,
                      jobs.settings.project_dir(pid) / "inputs", jobs.envs)
    if solver_warning:
        assert not v.passed
        assert next(c for c in v.checks if c.name == "R reference fit converged").status == "failed"
        return
    assert v.passed, [(c.name, c.status) for c in v.checks if not c.passed]
    assert {c.name for c in v.checks if c.passed} >= {
        "R reference fit converged", "Python fit converged", "predictions", "pinball_loss on train"}


def test_diagnostics_that_fail_leave_the_validation_passed_and_record_missing_evidence(claims, tmp_path):
    """A candidate that cannot refit on fewer rows passes parity; its cross-validation is missing evidence by stage
    and type, and the sensitivity and VIF diagnostics still run."""
    fragile = POISSON_PY.replace("def fit(train):\n", "def fit(train):\n    if len(train) < 140:\n"
                                 "        raise ValueError('needs every train row: secret-7')\n")
    v = run(claims, fragile, tmp_path)
    assert v.passed, [c for c in v.checks if not c.passed]
    checks = v.evaluation["checks"]
    assert "cross-validation refits (failed in fit() on cross-validation fold 1 (ValueError))" in \
        checks["stability"]["needs"]
    assert checks["stability"]["results"]["training_curve"]["iterations"] > 1
    age = checks["sensitivity"]["results"]["columns"]["age"]
    assert age["status"] == "executed" and age["sets"]["test"]["value"]["up"]["mean_change"] > 0
    assert checks["sensitivity"]["status"] == "partially executed"                  # area is not numeric
    assert checks["assumptions"]["results"]["assumptions"]["multicollinearity"]["status"] == "executed"
    assert "secret-7" not in json.dumps(v.evaluation)


def test_cross_validation_folds_are_fixed_by_row_number(claims, tmp_path):
    first = run(claims, POISSON_PY, tmp_path / "a").evaluation["checks"]["stability"]["results"]
    again = run(claims, POISSON_PY, tmp_path / "b").evaluation["checks"]["stability"]["results"]
    assert first["cross_validation"] == again["cross_validation"]
    assert sum(f["held_out_rows"] for f in first["cross_validation"]["folds"]) == 140
