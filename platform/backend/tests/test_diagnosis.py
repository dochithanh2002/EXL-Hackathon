"""What the Diagnostician is shown: every check and name, never R's numbers, row-level values or data values."""
import json

import pytest
from pydantic import ValidationError

from controlplane.diagnosis import REDACTED, redact, validation_view, vocabulary
from controlplane.schemas import Diagnosis
from tests.test_baseline import poisson_spec

PROFILE = {"columns": [{"name": "area", "r_name": "area", "levels": [{"value": "A", "n": 3}, {"value": 7, "n": 1}]},
                       {"name": "policy id", "r_name": "policy.id"}]}
SOURCE = """import pandas as pd
SCALE = 0.25


def prepare(path, sets):
    df = pd.read_csv(path)
    df["area_code"] = 1
"""
WORDS = vocabulary(PROFILE, ["(Intercept)", "areaB"], poisson_spec())

PARITY = {
    "passed": False, "stage": "parity", "error": None, "notes": ["note"],
    "checks": [{"name": "coefficient estimates", "quantity": "coefficients", "kind": "tolerance", "value": 12300.0,
                "tolerance": 1.0, "status": "failed", "passed": False, "required": True, "reason": "", "atol": 1e-8,
                "rtol": 1e-6, "max_abs_diff": 0.0038624, "compared": 4, "missing_both": 0, "missing_one_side": 1,
                "nonfinite_mismatch": 0, "total": None}],
    "metrics": {"test": {"rmse": {"r": 0.81, "python": 0.83}}},
    "design": {"fitted_rows": {"r": 140, "python": 139}, "reference_levels": {"area": {"r": "A", "python": "B"}}},
    "assessment": {"python": {"stability": {"status": "not assessed", "needs": ["a holdout"]}}},
    "evaluation": {"missing_evidence": ["Sensitivity: not in the validation plan (no lever)"], "checks": {
        "assumptions": {"status": "executed", "rationale": "count", "needs": [], "notes": [], "results": {
            "assumptions": {"overdispersion": {"status": "executed", "results": {
                "r": {"pearson_dispersion": 1.4142135}, "python": {"pearson_dispersion": 1.4142136}}}}}},
        "stability": {"status": "executed", "rationale": "refits", "needs": [], "notes": [], "results": {
            "cross_validation": {"coefficients": [{"term": "areaB", "estimate": -0.3180216, "fold_sd": 0.02,
                                                   "relative_sd": 0.06, "sign_changes": 0}]}}},
        "calibration": {"status": "executed", "rationale": "pricing", "needs": [], "notes": [], "results": {
            "sets": {"test": {"value": {"python": {"slope": 0.97, "bins": [{"mean_observed": 1.6180339}]}}}}}}}},
    "differences": {
        "worst_terms": [{"term": "areaB", "estimate_r": -0.3141592, "estimate_python": -0.3180216,
                         "estimate_rel": 0.012294518, "estimate_score": 12290.1, "std_error_r": 0.0271828,
                         "std_error_python": 0.0271828, "std_error_rel": 0.0, "std_error_score": 0.0},
                        {"term": "age", "estimate_r": None, "estimate_python": 0.0161803, "estimate_rel": None,
                         "estimate_score": None, "std_error_r": None, "std_error_python": 0.0011235,
                         "std_error_rel": None, "std_error_score": None}],
        "worst_predictions": [{"set": "test", "row": 17, "key": "", "r": 1.6180339, "python": 1.6290000,
                               "abs_diff": 0.0109661, "score": 6773.4}],
        "fit_statistics": {"deviance": {"r": 211.4142, "python": 209.7320}},
        "missing_terms": [], "rows_only_in_r": [[["test", 3]]],
    },
}


def test_the_view_keeps_checks_names_rows_metrics_and_summaries_but_no_estimates_or_predictions():
    view = validation_view(PARITY, WORDS, SOURCE)
    text = json.dumps(view)
    for value in ("0.3141592", "0.3180216", "0.0271828", "0.0161803", "0.0011235", "1.6180339", "1.629",
                  "211.4142", "209.732"):
        assert value not in text
    [check] = view["checks"]
    assert check["status"] == "failed" and check["missing_one_side"] == 1 and "total" not in check
    assert view["metrics"] == PARITY["metrics"]
    assert view["design"] == PARITY["design"] and view["assessment"] == PARITY["assessment"]
    ev = view["evaluation"]                         # Python's side only: no R values, estimates or binned values
    assert ev["missing_evidence"] == PARITY["evaluation"]["missing_evidence"]
    assert ev["checks"]["assumptions"]["results"]["assumptions"]["overdispersion"]["results"] == {
        "python": {"pearson_dispersion": 1.4142136}}
    assert ev["checks"]["stability"]["results"]["cross_validation"]["coefficients"] == [
        {"term": "areaB", "fold_sd": 0.02, "relative_sd": 0.06, "sign_changes": 0}]
    assert ev["checks"]["calibration"]["results"]["sets"]["test"]["value"]["python"] == {"slope": 0.97}
    assert view["differences"]["worst_terms"][0] == {
        "term": "areaB", "estimate_rel_diff": 0.0123, "estimate_tolerance_units": 12300.0,
        "estimate": "larger in magnitude", "std_error_rel_diff": 0.0, "std_error_tolerance_units": 0.0,
        "std_error": "equal"}
    assert view["differences"]["worst_terms"][1]["estimate"] == "NA in R only"
    assert view["differences"]["worst_predictions"] == [
        {"set": "test", "row": 17, "key": "", "abs_diff": 0.011, "tolerance_units": 6770.0}]
    assert view["differences"]["fit_statistics"] == {"deviance": "smaller in magnitude"}
    assert view["differences"]["rows_only_in_r"] == [[["test", 3]]]


def test_runtime_errors_keep_names_and_frames_but_lose_data_values():
    v = {"passed": False, "stage": "runtime", "checks": [], "metrics": {}, "notes": [], "differences": {},
         "error": {"stage": "prepare()", "type": "ValueError", "candidate_frames": [{"line": 5, "function": "prepare"}],
                   "message": "could not convert string to float: 'secret-12' in column 'policy id' near 3.75"}}
    e = validation_view(v, WORDS, SOURCE)["error"]
    assert e["message"] == f"could not convert string to float: {REDACTED} in column 'policy id' near {REDACTED}"
    assert e["candidate_frames"] == [{"line": 5, "function": "prepare"}] and e["stage"] == "prepare()"


def test_redaction_keeps_what_the_agents_already_see():
    assert redact("KeyError: 'areaB'", WORDS, SOURCE) == "KeyError: 'areaB'"                 # R term
    assert redact("KeyError: 'area_code'", WORDS, SOURCE) == "KeyError: 'area_code'"         # candidate literal
    assert redact("level '7' and 'A' vs 'Z'", WORDS, SOURCE) == f"level '7' and 'A' vs {REDACTED}"
    assert redact("can't multiply by non-int of type 'float'", WORDS, SOURCE) == \
        "can't multiply by non-int of type 'float'"
    assert redact("scale 0.25 not 0.5; 140 rows, shape (140, 4), statsmodels 0.15.0", WORDS, SOURCE) == \
        f"scale 0.25 not {REDACTED}; 140 rows, shape (140, 4), statsmodels 0.15.0"
    assert redact('value "x1" or 1e-08', WORDS, SOURCE) == f"value {REDACTED} or {REDACTED}"


SPEC_CHANGE = {"category": "specification", "spec_change_needed": True}


@pytest.mark.parametrize("change, ok", [
    ({}, False), ({"repair_instructions": ["x"]}, True), (SPEC_CHANGE, False),
    (SPEC_CHANGE | {"proposed_spec_change": "describe the weights as precision weights"}, True),
    ({"category": "model_fitness", "spec_change_needed": True, "proposed_spec_change": "add a holdout"}, True),
    ({"category": "model_fitness", "repair_instructions": ["recalibrate"]}, False)])
def test_a_diagnosis_must_be_actionable(change, ok):
    raw = {"root_cause": "r", "evidence": ["e"], "repair_instructions": [], "spec_change_needed": False,
           "category": "migration_defect"} | change
    if ok:
        Diagnosis.model_validate(raw)
    else:
        with pytest.raises(ValidationError):
            Diagnosis.model_validate(raw)
