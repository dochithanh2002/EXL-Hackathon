"""The validation plan's checks, executed by code on recorded predictions: what each computes, what it records as
missing, and the plots and tables drawn from it."""
import csv
import json
import math
import xml.etree.ElementTree as ET

import pytest

from controlplane import runner, statistics
from controlplane.evaluation import Inputs, business_fitness, evaluate, evaluation_view, tables
from controlplane.plots import render
from controlplane.report import evaluation_facts
from controlplane.statistics import linear_recalibration, logistic_recalibration
from controlplane.validation import Validation, compare, with_evaluation
from tests.test_schemas import plan
from tests.test_schemas import spec as make_spec
from tests.test_validation import pair

N = 400


def run(r, py, diagnostics, spec: dict) -> dict:
    return evaluate(Inputs(r, py, diagnostics, spec))


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def write_side(d, rows, r=True):
    """rows: (row, set, y, prediction)."""
    d.mkdir()
    with open(d / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "set", "key", "y", "weight", "prediction"] if r else ["row", "set", "key", "y",
                                                                                 "prediction"])
        w.writerows([(row, s, "", y, 1, p) if r else (row, s, "", y, p) for row, s, y, p in rows])
    (d / "coefficients.csv").write_text("term,estimate,std_error\n(Intercept),-0.5,0.1\nx,1.0,0.1\n")
    (d / "fit.json").write_text(json.dumps({"pearson_chi2": 396.0, "df_residual": 398, "df_resid": 398,
                                            "deviance": 420.0}))
    return d


@pytest.fixture
def sides(tmp_path):
    """A calibrated logit on N rows, three quarters train: R's capture with segments, and Python's output."""
    rows = []
    for i in range(1, N + 1):
        x = (i % 40) / 10 - 2
        p = logistic(-0.5 + x)
        y = 1 if ((i * 7919) % 1000) / 1000 < p else 0
        rows.append((i, "train" if i <= 300 else "test", y, p))
    r = write_side(tmp_path / "r", rows)
    py = write_side(tmp_path / "py", rows, r=False)
    (r / "meta.json").write_text(json.dumps({"segments": {"region": "categorical", "x": "numeric",
                                                          "absent": "missing"}, "iterations": 4,
                                             "separation_warning": False}))
    with open(r / "segments.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "row", "column", "value"])
        for row, s, *_ in rows:
            w.writerow([s, row, "region", "rare" if row == 7 else "north" if row % 2 else "south"])
            w.writerow([s, row, "x", "" if row == 9 else (row % 40) / 10 - 2])
    return r, py


def body(**plan_overrides) -> dict:
    return make_spec(metrics=["auc", "brier"], validation_plan=plan(
        subgroup_performance={"run": True, "rationale": "regions", "columns": ["region", "x", "absent"],
                              "min_rows": 30},
        assumptions={"run": True, "rationale": "logit", "assumptions": ["separation", "multicollinearity"]},
        sensitivity={"run": False, "rationale": "no policy lever"}, **plan_overrides)).model_dump(mode="json")


def test_recalibration_recovers_a_calibrated_model():
    a, b = logistic_recalibration([(logistic(x / 50), logistic(x / 50), 1.0) for x in range(-200, 201)])
    assert a == pytest.approx(0, abs=1e-8) and b == pytest.approx(1, abs=1e-8)       # y = p exactly
    overconfident = [(logistic(x / 50), logistic(2 * x / 50), 1.0) for x in range(-200, 201)]
    assert logistic_recalibration(overconfident)[1] == pytest.approx(0.5, abs=1e-8)
    separated = [(float(x > 0), logistic(x / 50), 1.0) for x in range(-200, 201)]
    assert logistic_recalibration(separated) is None                                   # no finite slope
    assert linear_recalibration([(2 * p + 1, p, 1.0) for p in range(5)]) == pytest.approx((1, 2))


def test_every_planned_check_runs_and_what_could_not_run_is_missing_evidence(sides):
    r, py = sides
    ev = run(r, py, None, body())
    c = ev["checks"]
    assert ev["primary_metric"] == "log_loss" and ev["metrics"] == ["log_loss", "auc", "brier"]

    perf = c["baseline_comparison"]["results"]["sets"]["test"]
    assert perf["auc"]["constant"] == 0.5 and perf["auc"]["python"] > 0.6
    assert perf["log_loss"]["python_vs_constant"] > 0 and perf["auc"]["r"] == perf["auc"]["python"]

    cal = c["calibration"]["results"]["sets"]["test"]["value"]["python"]
    assert 0.5 < cal["slope"] < 1.5 and len(cal["bins"]) == 10 and c["calibration"]["status"] == "executed"

    sub = c["subgroup_performance"]
    assert sub["status"] == "partially executed" and sub["needs"] == ["subgroup column absent in R's set frames"]
    region = sub["results"]["columns"]["region"]
    assert [g["group"] for g in region["groups"]] == ["south", "north", "rare"]     # by development rows
    assert region["small_groups"] == ["rare"] and region["groups"][0]["sets"]["test"]["rows"] == 50
    x = sub["results"]["columns"]["x"]
    assert [g["group"] for g in x["groups"]][-1] == "(missing)" and x["groups"][0]["lower"] is None

    assert c["stability"]["status"] == "partially executed" and set(c["stability"]["results"]) == {"psi"}
    assert "cross-validation refits (the Python diagnostics did not run)" in c["stability"]["needs"]
    assert c["sensitivity"]["status"] == "not planned"
    a = c["assumptions"]["results"]["assumptions"]
    assert a["separation"]["results"]["r"] == {"extreme_prediction_share": 0, "fitted_probabilities_warning": False}
    assert a["multicollinearity"]["status"] == "not executed" and c["assumptions"]["status"] == "partially executed"
    assert {"Sensitivity: not in the validation plan (no policy lever)",
            "Subgroup performance: subgroup column absent in R's set frames",
            "Stability: cross-validation refits (the Python diagnostics did not run)"} <= set(ev["missing_evidence"])


def test_a_failed_diagnostics_run_is_recorded_by_stage_and_type_only(sides, tmp_path):
    r, py = sides
    d = tmp_path / "diagnostics"
    d.mkdir()
    (d / "error.json").write_text(json.dumps({"stage": "prepare()", "type": "KeyError", "message": "'secret-42'"}))
    ev = run(r, py, d, body())
    needs = ev["checks"]["stability"]["needs"]
    assert "cross-validation refits (failed in prepare() (KeyError))" in needs
    assert "secret" not in json.dumps(ev)


def test_agent_subgroup_evidence_uses_the_number_of_rows_scored_by_each_side(sides):
    r, py = sides
    spec = body()
    spec["validation_plan"]["subgroup_performance"]["min_rows"] = 1
    spec["validation_plan"]["subgroup_performance"]["columns"] = ["region"]
    path = py / "predictions.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    # R still scores every row. Leave Python just one usable prediction for the north holdout group.
    for row in rows:
        if row["set"] == "test" and int(row["row"]) % 2 and row["row"] != "301":
            row["prediction"] = ""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ev = run(r, py, None, spec)
    raw = ev["checks"]["subgroup_performance"]["results"]["columns"]["region"]
    assert next(g for g in raw["groups"] if g["group"] == "rare")["sets"]["train"]["python"]

    diagnostic = evaluation_view(ev)["checks"]["subgroup_performance"]["results"]["columns"]["region"]
    groups = {g["group"]: g["sets"] for g in diagnostic["groups"]}
    assert "rare" not in groups and "test" not in groups["north"]
    assert groups["south"]["test"]["python"]["log_loss"] is not None

    report = evaluation_facts(ev)["subgroup_performance"]["columns"]["region"]
    assert "rare" not in report["groups"] and report["cells_withheld"] == 2
    assert set(report["groups"]["north"]["test"]) == {"rows", "r"}
    assert report["groups"]["south"]["test"]["python"]["log_loss"] is not None


def test_cross_validation_and_vif_results(sides, tmp_path):
    r, py = sides
    d = tmp_path / "diagnostics"
    d.mkdir()
    with open(d / "cv_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "row", "set", "key", "y", "prediction"])
        for row in range(1, 301):
            p = logistic(-0.5 + (row % 40) / 10 - 2)
            w.writerow([row % 3 + 1, row, "train", "", int(((row * 7919) % 1000) / 1000 < p), p])
    (d / "cv_coefficients.csv").write_text("fold,term,estimate,std_error\n" + "\n".join(
        f"{k},(Intercept),{-0.5 + k / 100},0.1\n{k},x,{1 if k < 3 else -1},0.1" for k in (1, 2, 3)))
    fits = [{"fold": k, "fit_rows": 200, "held_out_rows": 100, "converged": k != 2} for k in (1, 2, 3)]
    (d / "diagnostics.json").write_text(json.dumps({
        "cross_validation": {"status": "executed", "folds": 3, "fits": fits},
        "vif": {"status": "executed", "condition_number": 4.0, "columns": [{"name": "x", "vif": 1.5},
                                                                           {"name": "z", "vif": 7.25}]}}))
    (py / "fit_history.json").write_text(json.dumps({"deviance": [500.0, 430.0, 420.5, 420.0]}))
    ev = run(r, py, d, body())
    st = ev["checks"]["stability"]
    assert st["status"] == "partially executed" and "converged refits on folds [2]" in st["needs"]
    cv = st["results"]["cross_validation"]
    assert [f["fold"] for f in cv["folds"]] == [1, 2, 3] and cv["summary"]["auc"]["sd"] > 0
    assert cv["coefficients"][0]["term"] == "x" and cv["coefficients"][0]["sign_changes"] == 1
    assert st["results"]["training_curve"] == {"metric": "deviance", "values": [500, 430, 420.5, 420],
                                               "iterations": 4, "r_iterations": 4}
    vif = ev["checks"]["assumptions"]["results"]["assumptions"]["multicollinearity"]["results"]
    assert vif["python"]["max_vif"] == 7.25 and vif["columns"][0]["name"] == "z"

    # the Diagnostician sees Python's side only: no R values, coefficient estimates or bins
    view = json.dumps(evaluation_view(ev))
    assert '"r"' not in view and '"estimate"' not in view and '"bins"' not in view and '"fold_sd"' in view

    # every plot is well-formed SVG with its title; every table has its header
    v = Validation(passed=True, stage="parity", evaluation=ev).model_dump(mode="json")
    written = render(v, tmp_path / "plots")
    names = {p["name"] for p in written}
    assert {"performance.svg", "calibration-test.svg", "subgroups-region-test.svg", "subgroups-x-test.svg",
            "training-curve.svg", "cross-validation.svg", "vif.svg"} <= names
    for p in written:
        root = ET.parse(tmp_path / "plots" / p["name"]).getroot()
        assert root.get("role") == "img" and root.findall(".//{http://www.w3.org/2000/svg}title")
    assert render(v, tmp_path / "again") == written
    assert all((tmp_path / "again" / p["name"]).read_text() == (tmp_path / "plots" / p["name"]).read_text()
               for p in written)
    t = tables(v)
    assert t["cv_folds.csv"][0][:4] == ["fold", "fit_rows", "held_out_rows", "converged"]
    # two sides x (train: 3 regions + 5 quintiles + missing; test: 2 regions + 5 quintiles)
    assert len(t["subgroups.csv"][1]) == 2 * ((3 + 6) + (2 + 5))


def test_a_broken_plot_never_fails_the_validation(sides, tmp_path, caplog):
    r, py = sides
    ev = run(r, py, None, body())
    ev["checks"]["calibration"]["results"]["sets"]["test"]["value"]["python"]["bins"] = "not a list"
    written = render(Validation(passed=True, stage="parity", evaluation=ev).model_dump(mode="json"), tmp_path)
    assert "performance.svg" in {p["name"] for p in written} and "could not draw a plot" in caplog.text


def criteria(spec: dict, **limits) -> dict:
    spec["assessment"]["criteria"] |= limits
    return spec


def test_status_and_criterion_outcome_are_independent(sides):
    r, py = sides
    c = run(r, py, None, criteria(body(), min_events_per_parameter=1000, max_calibration_deviation=10))["checks"]
    sample = c["sample_suitability"]
    assert (sample["status"], sample["criterion"]["outcome"]) == ("executed", "not met")
    assert sample["criterion"]["rule"] == "per_parameter >= 1000" and sample["needs"] == []
    assert sample["criterion"]["value"]["per_parameter"] == sample["results"]["per_parameter"]
    cal = c["calibration"]
    assert (cal["status"], cal["criterion"]["outcome"], cal["criterion"]["judged_on"]) == ("executed", "met", ["test"])
    stab = c["stability"]                                   # PSI computed, no max_psi stated
    assert stab["criterion"]["outcome"] == "no criterion" and stab["criterion"]["rule"] is None
    assert stab["results"]["psi"]["python"]["vs_development"]["test"] is not None
    assert c["baseline_comparison"]["criterion"] is None and c["sensitivity"]["status"] == "not planned"


def test_decision_performance_is_not_applicable_to_a_count_model(tmp_path):
    r, py = pair(tmp_path)
    c = run(r, py, None, make_spec(family="poisson").model_dump(mode="json"))["checks"]["decision_performance"]
    assert c["status"] == "not applicable" and c["criterion"] is None and c["needs"] == []
    assert "count response" in c["rationale"]


@pytest.mark.parametrize("holdout, threshold, recall, status, outcome", [
    (False, .5, .5, "partially executed", "met"),            # judged on development data only
    (True, .99, .5, "executed", "not met"),
    (True, .5, .5, "executed", "met"),
    (True, .5, None, "executed", "no criterion"),
    (True, None, .5, "partially executed", "not judged"),     # no threshold: recall cannot be computed
])
def test_decision_performance_judges_the_python_candidate_on_independent_data(
        tmp_path, holdout, threshold, recall, status, outcome):
    preds = [(1, "train", 0., .1), (2, "train", 1., .9)] + ([(3, "test", 0., .2), (4, "test", 1., .8)]
                                                             if holdout else [])
    s = make_spec(baseline_objects={"model": "fit", "train": "train", "test": "test" if holdout else None})
    spec = criteria(s.model_dump(mode="json"), min_auc=.7, min_events_captured=recall)
    if threshold is not None:
        spec["assessment"]["decision"] = {"threshold": threshold, "action": "decline"}
    c = run(*pair(tmp_path, preds=preds, r_preds=preds), None, spec)["checks"]["decision_performance"]
    assert (c["status"], c["criterion"]["outcome"]) == (status, outcome), c
    assert c["criterion"]["judged_on"] == (["test"] if holdout else ["train"])


def test_a_check_left_out_of_the_plan_runs_when_the_spec_sets_its_criterion(sides):
    r, py = sides
    spec = criteria(body(calibration={"run": False, "rationale": "not asked for"}), max_calibration_deviation=.5)
    cal = run(r, py, None, spec)["checks"]["calibration"]
    assert cal["status"] == "executed" and cal["criterion"]["outcome"] == "met"
    assert cal["notes"][0] == ("not in the validation plan (not asked for); executed because the specification "
                               "sets max_calibration_deviation")
    left_out = run(r, py, None, body(calibration={"run": False, "rationale": "not asked for"}))["checks"]
    assert left_out["calibration"]["status"] == "not planned" and left_out["calibration"]["criterion"] is None


def test_predictions_are_read_once_per_side(tmp_path, monkeypatch):
    calls = []
    real = runner.parity._predictions

    def counted(path):
        calls.append(path)
        return real(path)
    monkeypatch.setattr(runner.parity, "_predictions", counted)
    monkeypatch.setattr(statistics, "_predictions", counted)
    r, py = pair(tmp_path)
    spec = make_spec(metrics=["log_loss"])
    inputs = Inputs(r, py, None, spec.model_dump(mode="json"))
    v = with_evaluation(compare(r, py, spec, inputs), inputs)
    assert v.evaluation["checks"]["sample_suitability"]["status"] == "executed"
    assert sorted(p.parent.name for p in calls) == ["py", "r"]


def check(status="executed", outcome=None, rule="x >= 1", needs=()):
    criterion = None if outcome is None else {"rule": rule, "limit": {"x": 1} if rule else {}, "value": {"x": 2},
                                              "outcome": outcome, "judged_on": ["test"]}
    return {"status": status, "needs": list(needs), "criterion": criterion}


@pytest.mark.parametrize("checks, verdict, basis", [
    ({"sample_suitability": check(outcome="met"), "calibration": check(outcome="met"),
      "sensitivity": check("not planned"), "decision_performance": check("not applicable")},
     "criteria met", ["Sample suitability: x >= 1 met (x 2)", "Calibration: x >= 1 met (x 2)"]),
    ({"sample_suitability": check(outcome="met"), "calibration": check("partially executed", "not met",
                                                                        needs=["a holdout"])},
     "criteria not met", ["Calibration: x >= 1 not met (x 2)"]),
    ({"sample_suitability": check(outcome="no criterion", rule=None)},
     "not established", ["Sample suitability: no criterion for min_events_per_parameter"]),
    ({"stability": check(outcome="not judged")}, "not established", ["Stability: x >= 1 not judged"]),
    ({"sample_suitability": check(outcome="met"), "assumptions": check("partially executed", needs=["vif"])},
     "not established", ["Model assumptions: partially executed (needs vif)"]),
    ({}, "not established", ["the model validation was not recorded"]),
])
def test_business_fitness_reads_criterion_outcomes_and_execution_gaps(checks, verdict, basis):
    assert business_fitness(checks) == {"verdict": verdict, "basis": basis}


# ---------------------------------------------------------------- fairness
FAIR = {"run": True, "rationale": "region is a proxy for a protected group", "columns": ["region", "absent"],
        "min_rows": 30, "favourable_outcome": "below_threshold"}


def fair_body(fairness=FAIR, criteria=None, **kw) -> dict:
    a = {"decision": {"threshold": 0.5, "action": "decline"}, "criteria": criteria or {}}
    return make_spec(**{"metrics": ["auc", "brier"]} | kw, validation_plan=plan(fairness=fairness),
                     assessment={"business_use": "credit decisions", "intended_population": "card holders",
                                 "sample_population": "the uploaded file", "missing_information": [],
                                 "evaluation_sets": [{"set": "train", "role": "development", "description": "fit"},
                                                     {"set": "test", "role": "holdout", "description": "held out"}],
                                 **a}).model_dump(mode="json")


def test_fairness_compares_favourable_outcomes_and_calibration_across_protected_groups(sides, tmp_path):
    r, py = sides
    ev = run(r, py, None, fair_body(criteria={"min_impact_ratio": 0.8, "max_group_calibration_deviation": 0.5}))
    f = ev["checks"]["fairness"]
    assert f["status"] == "partially executed" and f["needs"] == ["protected attribute absent in R's set frames"]
    region = f["results"]["columns"]["region"]
    assert region["small_groups"] == ["rare"]

    # the impact ratio, recomputed from the fixture's test rows: favourable = prediction below 0.5
    shares = {}
    for g in ("north", "south"):
        rows = [i for i in range(301, N + 1) if (i % 2 == 1) == (g == "north") and i != 7]
        shares[g] = sum(logistic(-0.5 + (i % 40) / 10 - 2) < 0.5 for i in rows) / len(rows)
    expected = min(shares.values()) / max(shares.values())
    assert region["impact_ratio"]["test"] == pytest.approx(expected, abs=1e-6)
    assert f["results"]["impact_ratio"] == pytest.approx(expected, abs=1e-6)
    by_group = {g["group"]: g["sets"] for g in region["groups"]}
    assert by_group["north"]["test"]["python"]["favourable_share"] == pytest.approx(shares["north"], abs=1e-6)
    assert set(by_group["rare"]) == {"train"} and by_group["rare"]["train"]["small"]     # one row: not judged

    cr = f["criterion"]
    assert cr["rule"] == "impact_ratio >= 0.8 and max_group_deviation <= 0.5" and cr["judged_on"] == ["test"]
    assert cr["outcome"] == ("met" if expected >= 0.8 and f["results"]["max_group_deviation"] <= 0.5 else "not met")
    assert any(row[0] == "region" for row in tables({"checks": [], "evaluation": ev})["fairness.csv"][1])
    facts = evaluation_facts(ev)["fairness"]
    assert facts["impact_ratio"] == f["results"]["impact_ratio"] and "region" in facts["columns"]
    view = evaluation_view(ev)["checks"]["fairness"]["results"]
    assert view["columns"]["region"]["impact_ratio"] == region["impact_ratio"]
    plots = render({"checks": [], "evaluation": ev}, tmp_path / "plots")
    assert {"fairness-region-test.svg"} <= {p["name"] for p in plots}


def test_a_spec_without_a_fairness_plan_records_it_as_not_planned(sides):
    r, py = sides
    ev = run(r, py, None, body())
    f = ev["checks"]["fairness"]
    assert f["status"] == "not planned" and "names no protected attributes" in f["rationale"]
    assert not any(b.startswith("Fairness") for b in ev["business_fitness"]["basis"])


def test_without_a_favourable_side_the_impact_ratio_is_not_judged(sides):
    r, py = sides
    plan_ = FAIR | {"favourable_outcome": None, "columns": ["region"]}
    f = run(r, py, None, fair_body(plan_, criteria={"min_impact_ratio": 0.8}))["checks"]["fairness"]
    assert f["status"] == "partially executed" and f["criterion"]["outcome"] == "not judged"
    assert f["needs"] == ["the decision threshold and which side of it is favourable, for the impact ratio"]


def test_fairness_criterion_on_a_count_model_is_calibration_only(sides):
    r, py = sides
    spec = fair_body(FAIR | {"favourable_outcome": None, "columns": ["region"]},
                     criteria={"max_group_calibration_deviation": 0.5}, family="poisson", link="log", metrics=[],
                     formula="y ~ x", response="y")
    spec["assessment"]["decision"] = None
    cr = run(r, py, None, spec)["checks"]["fairness"]["criterion"]
    assert cr["applies"] == ["max_group_deviation"] and cr["outcome"] in ("met", "not met")
    assert "impact_ratio" not in cr["value"]


def test_fairness_plan_is_validated_and_its_columns_are_exported_by_r():
    with pytest.raises(ValueError, match="fairness.columns: name the protected attributes"):
        fair_body(FAIR | {"columns": []})
    with pytest.raises(ValueError, match="favourable_outcome: needs assessment.decision"):
        make_spec(validation_plan=plan(fairness=FAIR))
    s = make_spec(validation_plan=plan(
        subgroup_performance={"run": True, "rationale": "regions", "columns": ["region", "sex"]},
        fairness=FAIR | {"favourable_outcome": None, "columns": ["sex", "age"]}))
    assert s.capture_config()["segments"] == ["region", "sex", "age"]
    assert make_spec().capture_config()["segments"] == []
