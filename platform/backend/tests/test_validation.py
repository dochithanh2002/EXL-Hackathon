import json
import math

import pytest

from controlplane.assessment import assess_both, design_comparison, psi
from controlplane.runner.parity import metric, score
from controlplane.validation import compare
from tests.test_schemas import spec as make_spec

COEFS = [("(Intercept)", 0.5, 0.1), ("x", -1.25, 0.02)]
PREDS = [(1, "train", 1.0, 0.8), (2, "train", 0.0, 0.3), (3, "train", 1.0, 0.6), (4, "test", 0.0, 0.4),
         (5, "test", 1.0, float("nan"))]
FIT = {"nobs": 3, "df_residual": 1, "deviance": 2.5, "null_deviance": 3.8, "dispersion": 1, "pearson_chi2": 2.9,
       "log_lik": -1.25, "aic": 6.5, "bic": 2.5 + 2 * math.log(3)}
PY_FIT = {"nobs": 3.0, "df_resid": 1.0, "deviance": 2.5, "null_deviance": 3.8, "scale": 1.0, "pearson_chi2": 2.9,
          "llf": -1.25, "converged": True}
DESIGN = {"n_fit": 3, "n_train": 3, "columns": [{"name": "(Intercept)", "mean": 1, "sd": 0, "min": 1, "max": 1},
                                                {"name": "x", "mean": 0.2, "sd": 1.1, "min": -1, "max": 2}],
          "reference_levels": {}, "weights": None, "offset": None}


def num(x: float) -> str:
    return "" if math.isnan(x) else repr(x)


def write(d, coefs, preds, fit, meta=None, r=True, design=DESIGN):
    """preds: (row, set, y, prediction) or (row, set, key, y, prediction)."""
    d.mkdir()
    head = "term,estimate,std_error" + (",statistic,p_value" if r else "")
    (d / "coefficients.csv").write_text("\n".join([head] + [f'"{t}",{num(e)},{num(s)}' + (",0,0" if r else "")
                                                            for t, e, s in coefs]))
    rows = [p if len(p) == 5 else (p[0], p[1], "", p[2], p[3]) for p in preds]
    head = "row,set,key,y,weight,prediction" if r else "row,set,key,y,prediction"
    (d / "predictions.csv").write_text("\n".join(
        [head] + [f'{row},"{s}","{k}",{num(y)},' + ("1," if r else "") + num(p) for row, s, k, y, p in rows]))
    (d / "fit.json").write_text(json.dumps(fit))
    (d / "design.json").write_text(json.dumps(design))
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))
        (d / "split.csv").write_text("row,set,in_fit\n" + "\n".join(
            f'{row},"{s}",TRUE' for row, s in dict.fromkeys((p[0], p[1]) for p in rows)))
    return d


def pair(tmp_path, coefs=COEFS, preds=PREDS, py_fit=PY_FIT, r_fit=FIT, family="binomial", link="logit",
         cls=("glm", "lm"), converged=True, r_preds=PREDS):
    meta = {"family": family, "link": link, "class": list(cls), "converged": converged,
            "closed_form": list(cls) == ["lm"], "n": {"train": 3}, "n_fit": 3}
    r = write(tmp_path / "r", COEFS, r_preds, r_fit, meta)
    py = write(tmp_path / "py", coefs, preds, py_fit, r=False)
    return r, py


def checks(v) -> dict:
    return {c.name: c for c in v.checks}


def test_identical_output_passes_and_missing_values_match_missing_values(tmp_path):
    v = compare(*pair(tmp_path), make_spec(metrics=["auc", "log_loss", "actual_expected", "aic"],
                                         optional_metrics={"test": ["auc"]}))
    assert v.passed, [c for c in v.checks if not c.passed and c.required]
    c = checks(v)
    assert c["fit aic"].value == 0 and c["fit bic"].value == 0     # parameters counted like R
    assert c["predictions"].missing_both == 1 and c["predictions"].status == "passed"
    assert v.metrics["train"]["auc"] == {"r": 1.0, "python": 1.0}
    assert v.metrics["test"]["auc"] == {"r": None, "python": None}
    # an undefined optional metric is unavailable, never passed, and does not fail the validation
    assert (c["auc on test"].status, c["auc on test"].passed, c["auc on test"].required) == ("unavailable", False, False)
    assert "aic on train" in c and c["R reference fit converged"].passed and c["Python fit converged"].passed


def test_tolerance_is_absolute_plus_relative():
    assert score(100.0, 100.0 + 1e-4, 1e-8, 1e-6) == pytest.approx(1e-4 / (1e-8 + 1e-4))
    assert score(0.0, 5e-9, 1e-8, 1e-6) == pytest.approx(0.5)                # atol covers values near zero
    assert score(0.0, 5e-9, 0, 1e-6) == math.inf                             # relative-only cannot pass at zero
    assert score(math.nan, math.nan, 1e-8, 1e-6) == 0 and score(math.inf, math.inf, 1e-8, 1e-6) == 0
    assert score(1.0, math.nan, 1, 1) == math.inf and score(1.0, math.inf, 1, 1) == math.inf


def test_requested_unavailable_metric_blocks_acceptance(tmp_path):
    v = compare(*pair(tmp_path), make_spec(metrics=["auc"]))
    assert not v.passed
    assert checks(v)["auc on test"].required
    assert checks(v)["auc on test"].status == "unavailable"


def test_explicit_r_convergence_exemption_is_not_reported_as_passed(tmp_path):
    s = make_spec(metrics=[], convergence={"mode": "not_applicable", "rationale": "Closed-form reference solver"})
    v = compare(*pair(tmp_path, converged=None), s)
    c = checks(v)["R reference fit converged"]
    assert v.passed and c.status == "unavailable" and not c.required and not c.passed
    assert c.reason == s.convergence.rationale


def test_generic_aic_uses_estimator_statistic_without_extra_scale(tmp_path):
    rfit = {"nobs": 3, "log_lik": -10., "aic": 24.}
    pyfit = {"nobs": 3, "llf": -10., "aic": 24., "converged": True}
    v = compare(*pair(tmp_path, cls=("multinom",), r_fit=rfit, py_fit=pyfit), make_spec(metrics=["aic"]))
    assert v.passed and v.metrics["train"]["aic"] == {"r": 24., "python": 24.}
    pyfit.pop("aic")
    sub = tmp_path / "missing"
    sub.mkdir()
    v = compare(*pair(sub, cls=("multinom",), r_fit=rfit, py_fit=pyfit), make_spec(metrics=["aic"]))
    assert not v.passed and checks(v)["aic on train"].status == "unavailable"


def test_grouped_brier_equals_individual_event_score():
    grouped = [(0, "", .75, .6, 4.)]
    individuals = [(0, "", 1., .6, 3.), (1, "", 0., .6, 1.)]
    assert metric("brier", grouped) == pytest.approx(.21)
    assert metric("brier", grouped) == pytest.approx(metric("brier", individuals))


def test_missing_class_predictions_return_partial_assessment(tmp_path):
    preds = [(1, "train", "a", 1., .7), (1, "train", "b", 0., math.nan),
             (2, "train", "a", 0., .4), (2, "train", "b", 1., math.nan)]
    s = make_spec(response_type="categorical", metrics=[],
                  prediction={"kind": "class_probability", "quantity": "probability", "units": "probability",
                              "keys": ["a", "b"], "alignment_keys": ["row", "class"]},
                  baseline_objects={"model": "fit", "train": "train", "test": None})
    v = compare(*pair(tmp_path, preds=preds, r_preds=preds), s)
    cal = v.assessment["python"]["calibration"]
    assert cal["status"] == "partially assessed" and cal["meets"] is None
    assert any("output b" in n for n in cal["needs"])
    assert cal["results"]["sets"]["train"]["b"]["status"] == "not assessed"


@pytest.mark.parametrize("holdout, threshold, recall_limit, status, meets", [
    (False, .5, .5, "partially assessed", None),
    (True, .99, .5, "assessed", False),
    (True, .5, .5, "assessed", True),
    (True, .5, None, "partially assessed", None),
])
def test_decision_assessment_requires_independent_data_and_threshold_criterion(
        tmp_path, holdout, threshold, recall_limit, status, meets):
    preds = [(1, "train", 0., .1), (2, "train", 1., .9)]
    if holdout:
        preds += [(3, "test", 0., .2), (4, "test", 1., .8)]
    s = make_spec(baseline_objects={"model": "fit", "train": "train", "test": "test" if holdout else None})
    body = s.model_dump(mode="json")
    body["assessment"]["decision"] = {"threshold": threshold, "action": "decline"}
    body["assessment"]["criteria"] |= {"min_auc": .7, "min_events_captured": recall_limit}
    a = assess_both(*pair(tmp_path, preds=preds, r_preds=preds), body)["python"]["decision_performance"]
    assert a["status"] == status and a["meets"] is meets


def test_each_kind_of_difference_fails_its_own_check(tmp_path):
    preds = [list(p) for p in PREDS]
    preds[0][3] = 0.8 + 1e-5                      # prediction drift beyond 1e-8 + 1e-6 * 0.8
    preds[1][2] = 1.0                             # observed value differs
    preds[3][3] = float("nan")                    # NA where R predicts
    coefs = [("(Intercept)", 0.5, 0.1), ("x", -1.25, 0.03)]
    v = compare(*pair(tmp_path, coefs=coefs, preds=[tuple(p) for p in preds[:-1]]), make_spec(metrics=[]))
    failed = {c.name for c in v.checks if not c.passed}
    assert failed == {"coefficient standard errors", "alignment keys only on one side", "observed values differing",
                      "prediction missingness differing", "predictions", "valid predictions on test"}
    assert v.differences["rows_only_in_r"] == [("test", 5, "")]
    assert v.differences["worst_terms"][0]["term"] == "x"
    assert v.differences["prediction_na_differs_rows"] == [{"set": "test", "row": 4, "key": "", "r_is_na": False}]
    assert checks(v)["predictions"].max_abs_diff == pytest.approx(1e-5)


def test_per_quantity_tolerances_apply(tmp_path):
    coefs = [("(Intercept)", 0.5 + 1e-4, 0.1), ("x", -1.25, 0.02)]
    loose = make_spec(metrics=[], tolerances={"coefficients": {"atol": 1e-3, "rtol": 0}})
    assert checks(compare(*pair(tmp_path, coefs=coefs), loose))["coefficient estimates"].passed
    (tmp_path / "tight").mkdir()
    v = compare(*pair(tmp_path / "tight", coefs=coefs), make_spec(metrics=[]))
    assert not checks(v)["coefficient estimates"].passed and not v.passed


def test_one_sided_missing_coefficient_fails_and_serialises(tmp_path):
    coefs = [("(Intercept)", 0.5, 0.1), ("x", float("nan"), float("nan"))]
    v = compare(*pair(tmp_path, coefs=coefs), make_spec(metrics=[]))
    c = checks(v)
    assert not c["coefficient estimates"].passed and c["coefficient estimates"].value == math.inf
    assert c["coefficient estimates"].missing_one_side == 1
    body = json.loads(v.model_dump_json())                   # strict JSON: non-finite values become null
    assert [x["value"] for x in body["checks"] if x["name"] == "coefficient estimates"] == [None]


def test_mismatched_names_leave_coefficient_checks_unavailable_not_passed(tmp_path):
    v = compare(*pair(tmp_path, coefs=[("(Intercept)", 0.5, 0.1), ("z", -1.25, 0.02)]), make_spec(metrics=[]))
    c = checks(v)
    assert c["coefficient estimates"].status == "unavailable" and not v.passed


def test_convergence_problems_and_missing_statistics_are_surfaced(tmp_path):
    v = compare(*pair(tmp_path, converged=False, py_fit=PY_FIT | {"converged": None}), make_spec(metrics=[]))
    c = checks(v)
    assert c["R reference fit converged"].status == "failed"
    assert c["Python fit converged"].status == "unavailable" and not v.passed

    (tmp_path / "stat").mkdir()
    v = compare(*pair(tmp_path / "stat", py_fit={k: v for k, v in PY_FIT.items() if k != "pearson_chi2"}),
                make_spec(metrics=[]))
    assert checks(v)["fit pearson_chi2"].status == "unavailable" and not v.passed

    (tmp_path / "gamma").mkdir()
    v = compare(*pair(tmp_path / "gamma", family="Gamma", link="inverse"), make_spec(family="Gamma", metrics=[]))
    assert "fit log_lik" not in checks(v)
    assert any("Gamma likelihood" in n for n in v.notes)


def test_a_set_without_valid_predictions_fails(tmp_path):
    preds = [(1, "train", 1.0, 0.8), (2, "train", 0.0, 0.3), (3, "train", 1.0, 0.6),
             (4, "test", 0.0, float("nan")), (5, "test", 1.0, float("nan"))]
    v = compare(*pair(tmp_path, preds=preds, r_preds=preds), make_spec(metrics=[]))
    c = checks(v)
    assert c["valid predictions on test"].status == "failed" and c["valid predictions on train"].passed
    assert not v.passed


def test_lm_statistics_are_read_from_ols_results(tmp_path):
    r_fit = {"nobs": 3, "df_residual": 1, "rss": 0.5, "sigma": math.sqrt(0.5), "r_squared": 0.9,
             "adj_r_squared": 0.8, "log_lik": -1.0, "aic": 8.0, "bic": 2 + 3 * math.log(3)}
    py_fit = {"nobs": 3.0, "df_resid": 1.0, "ssr": 0.5, "scale": 0.5, "rsquared": 0.9, "rsquared_adj": 0.8,
              "llf": -1.0, "converged": None}
    v = compare(*pair(tmp_path, r_fit=r_fit, py_fit=py_fit, family="gaussian", link="identity", cls=("lm",)),
                make_spec(family="gaussian", metrics=[]))
    assert v.passed, [c for c in v.checks if not c.passed]
    assert {"fit sigma", "fit aic", "fit r_squared"} <= set(checks(v))
    assert "Python fit converged" not in checks(v)               # closed form: no flag needed


def test_class_probabilities_align_on_row_and_class(tmp_path):
    preds = [(1, "train", "a", 1.0, 0.7), (1, "train", "b", 0.0, 0.3), (2, "train", "a", 0.0, 0.4),
             (2, "train", "b", 1.0, 0.6)]
    s = make_spec(family=None, link=None, r_function="nnet::multinom", python_implementation="statsmodels.api.MNLogit",
                  response_type="categorical", metrics=["log_loss", "brier"],
                  prediction={"kind": "class_probability", "quantity": "probability", "units": "class probability",
                              "keys": ["a", "b"], "alignment_keys": ["row", "class"]},
                  baseline_objects={"model": "fit", "train": "train", "test": None})
    v = compare(*pair(tmp_path, preds=preds, r_preds=preds, cls=("multinom", "nnet"), family=None, link=None,
                      r_fit={"nobs": 2, "log_lik": -1.0}, py_fit={"nobs": 2.0, "llf": -1.0, "converged": True}), s)
    assert v.passed, [c for c in v.checks if not c.passed]
    assert v.metrics["train"]["log_loss"]["r"] == pytest.approx(-(math.log(0.7) + math.log(0.6)) / 2)
    assert v.metrics["train"]["brier"]["r"] == pytest.approx((0.09 * 2 + 0.16 * 2) / 2)


UNITS = [(i, "", y, p, 1.0) for i, (y, p) in enumerate([(1.0, 0.9), (0.0, 0.1), (1.0, 0.4), (0.0, 0.4)])]


@pytest.mark.parametrize("name, family, value", [
    ("auc", "binomial", 0.875),
    ("gini", "binomial", 0.75),
    ("ks", "binomial", 0.5),
    ("rmse", "binomial", math.sqrt((0.01 + 0.01 + 0.36 + 0.16) / 4)),
    ("mae", "binomial", (0.1 + 0.1 + 0.6 + 0.4) / 4),
    ("actual_expected", "binomial", 2 / 1.8),
    ("log_loss", "binomial", -(math.log(0.9) + math.log(0.9) + math.log(0.4) + math.log(0.6)) / 4),
    ("brier", "binomial", (0.01 + 0.01 + 0.36 + 0.16) / 4),
    ("deviance", "gaussian", 0.01 + 0.01 + 0.36 + 0.16),
    ("deviance", "binomial", -2 * (math.log(0.9) + math.log(0.9) + math.log(0.4) + math.log(0.6))),
    ("deviance", "quasipoisson", 2 * (math.log(1 / 0.9) + math.log(1 / 0.4) - 0.2)),
])
def test_metrics(name, family, value):
    assert metric(name, UNITS, family) == pytest.approx(value)


def test_weights_and_grouped_binomial_rows_count_as_observations():
    doubled = [(i, k, y, p, 2.0 if i == 0 else 1.0) for i, k, y, p, _ in UNITS]
    expanded = UNITS + [UNITS[0]]
    for name in ("auc", "log_loss", "rmse", "actual_expected"):
        assert metric(name, doubled) == pytest.approx(metric(name, expanded)), name
    assert metric("rmse", doubled, weighted=False) == pytest.approx(metric("rmse", UNITS))
    # a grouped row of 4 trials with 3 successes ranks like 3 events and 1 non-event at the same score
    grouped = [(0, "", 0.75, 0.6, 4.0), (1, "", 0.0, 0.2, 1.0)]
    ungrouped = [(0, "", 1.0, 0.6, 3.0), (0, "", 0.0, 0.6, 1.0), (1, "", 0.0, 0.2, 1.0)]
    assert metric("auc", grouped) == pytest.approx(metric("auc", ungrouped))
    assert metric("log_loss", grouped) == pytest.approx(metric("log_loss", ungrouped))
    assert metric("auc", [(0, "", 2.0, 0.9, 1.0), (1, "", 0.0, 0.1, 1.0)]) is None


def test_pinball_loss():
    units = [(0, "0.5", 10.0, 8.0, 1.0), (0, "0.9", 10.0, 12.0, 1.0)]
    assert metric("pinball_loss", units, kind="quantile") == pytest.approx((0.5 * 2 + 0.1 * 2) / 2)


def test_psi_is_zero_for_the_same_distribution_and_grows_with_shift():
    base = [i / 100 for i in range(100)]
    edges = sorted({base[i * 10] for i in range(1, 10)})
    assert psi(base, base, edges) == pytest.approx(0)
    assert psi(base, [x / 2 for x in base], edges) > 0.5


def test_assessment_reports_not_assessed_and_what_it_needs(tmp_path):
    r, py = pair(tmp_path)
    no_test = make_spec(baseline_objects={"model": "fit", "train": "train", "test": None})
    a = assess_both(r, py, no_test.model_dump(mode="json"))
    assert set(a) == {"r", "python"}
    py_a = a["python"]
    assert py_a["stability"]["status"] == "not assessed" and "holdout" in py_a["stability"]["needs"][0]
    assert py_a["decision_performance"]["status"] == "partially assessed"
    assert any("threshold" in n for n in py_a["decision_performance"]["needs"])
    assert py_a["calibration"]["status"] == "partially assessed"
    assert any("development set only" in n for n in py_a["calibration"]["needs"])
    s = py_a["sample_suitability"]["results"]
    assert s["sets"]["train"]["events"] == 2 and s["per_parameter"] == 0.5        # one non-event, two parameters


def test_assessment_with_criteria_and_threshold_is_assessed(tmp_path):
    r, py = pair(tmp_path)
    s = make_spec(assessment={
        "business_use": "approve or decline", "intended_population": "applicants", "sample_population": "file",
        "evaluation_sets": [{"set": "train", "role": "development", "description": "fit"},
                            {"set": "test", "role": "holdout", "description": "held out"}],
        "decision": {"threshold": 0.5, "action": "decline"},
        "criteria": {"min_events_per_parameter": 10, "min_auc": 0.6, "max_psi": 0.25,
                     "max_calibration_deviation": 0.1}, "missing_information": []})
    a = assess_both(r, py, s.model_dump(mode="json"))["python"]
    d = a["decision_performance"]
    assert d["status"] == "partially assessed" and d["meets"] is None
    assert any("computable AUC" in n for n in d["needs"])
    assert d["results"]["at_threshold"]["train"]["flagged_share"] == pytest.approx(2 / 3)
    assert a["sample_suitability"]["status"] == "assessed" and a["sample_suitability"]["meets"] is False
    assert a["stability"]["status"] == "assessed"


def test_design_comparison_shows_fitted_rows_columns_and_reference_levels(tmp_path):
    r, py = pair(tmp_path)
    (r / "meta.json").write_text(json.dumps(json.loads((r / "meta.json").read_text())
                                            | {"xlevels": {"area": ["A", "B", "C"]}}))
    (r / "design.json").write_text(json.dumps(DESIGN | {"reference_levels": {"area": "A"},
                                                        "weights": {"n": 3, "sum": 3, "mean": 1, "min": 1, "max": 1,
                                                                    "n_zero": 0}}))
    cols = DESIGN["columns"] + [{"name": "areaA", "mean": 0.3, "sd": 0.4, "min": 0, "max": 1},
                                {"name": "areaC", "mean": 0.3, "sd": 0.4, "min": 0, "max": 1}]
    (py / "design.json").write_text(json.dumps({"n_fit": 3, "columns": cols, "weights": None, "offset": None}))
    d = design_comparison(r, py)
    assert d["fitted_rows"] == {"r": 3, "python": 3}
    assert d["reference_levels"]["area"] == {"r": "A", "python": "B"}       # Python dropped B, R dropped A
    assert d["columns_only_in_python"] == ["areaA", "areaC"]
    assert d["weights"]["python"] is None and d["weights"]["r"]["sum"] == 3
