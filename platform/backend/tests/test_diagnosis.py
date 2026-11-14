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
    "checks": [{"name": "coefficient max rel diff", "value": 0.0123, "tolerance": 1e-6, "passed": False}],
    "kpis": {"test": {"rmse": {"r": 0.81, "python": 0.83}}},
    "differences": {
        "worst_terms": [{"term": "areaB", "estimate_r": -0.3141592, "estimate_python": -0.3180216,
                         "estimate_rel": 0.012294518, "std_error_r": 0.0271828, "std_error_python": 0.0271828,
                         "std_error_rel": 0.0},
                        {"term": "age", "estimate_r": None, "estimate_python": 0.0161803, "estimate_rel": None,
                         "std_error_r": None, "std_error_python": 0.0011235, "std_error_rel": None}],
        "worst_predictions": [{"set": "test", "row": 17, "r": 1.6180339, "python": 1.6290000, "abs_diff": 0.0109661}],
        "fit_statistics": {"deviance": {"r": 211.4142, "python": 209.7320}},
        "missing_terms": [], "rows_only_in_r": [[["test", 3]]],
    },
}


def test_the_view_keeps_checks_names_rows_and_kpis_but_no_estimates_or_predictions():
    view = validation_view(PARITY, WORDS, SOURCE)
    text = json.dumps(view)
    for value in ("0.3141592", "0.3180216", "0.0271828", "0.0161803", "0.0011235", "1.6180339", "1.629",
                  "211.4142", "209.732"):
        assert value not in text
    assert view["checks"] == PARITY["checks"] and view["kpis"] == PARITY["kpis"]
    assert view["differences"]["worst_terms"][0] == {
        "term": "areaB", "estimate_rel_diff": 0.0123, "estimate": "larger in magnitude",
        "std_error_rel_diff": 0.0, "std_error": "equal"}
    assert view["differences"]["worst_terms"][1]["estimate"] == "NA in R only"
    assert view["differences"]["worst_predictions"] == [{"set": "test", "row": 17, "abs_diff": 0.011}]
    assert view["differences"]["fit_statistics"] == {"deviance": "smaller in magnitude"}
    assert view["differences"]["rows_only_in_r"] == [[["test", 3]]]


def test_runtime_errors_keep_names_and_frames_but_lose_data_values():
    v = {"passed": False, "stage": "runtime", "checks": [], "kpis": {}, "notes": [], "differences": {},
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


@pytest.mark.parametrize("change, ok", [
    ({}, False), ({"repair_instructions": ["x"]}, True), ({"spec_change_needed": True}, False),
    ({"spec_change_needed": True, "proposed_spec_change": "loosen se_rel"}, True)])
def test_a_diagnosis_must_be_actionable(change, ok):
    raw = {"root_cause": "r", "evidence": ["e"], "repair_instructions": [], "spec_change_needed": False} | change
    if ok:
        Diagnosis.model_validate(raw)
    else:
        with pytest.raises(ValidationError):
            Diagnosis.model_validate(raw)
