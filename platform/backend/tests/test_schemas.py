import re

import pytest
from pydantic import ValidationError

from controlplane import support
from controlplane.schemas import ROLES, Diagnosis, ModelSpec, ReportNarrative, canonical_hash

BINARY = dict(response_type="binary", event={"meaning": "default", "complement": "non-default", "coding": "1"},
              prediction={"kind": "scalar", "quantity": "probability", "units": "probability of default",
                          "alignment_keys": ["row"]})
COUNT = dict(response_type="count", event=None,
             prediction={"kind": "scalar", "quantity": "count", "units": "claims per policy",
                         "alignment_keys": ["row"]})


def assessment(test: bool = True, **overrides) -> dict:
    sets = [{"set": "train", "role": "development", "description": "fitting rows"}]
    if test:
        sets.append({"set": "test", "role": "holdout", "description": "held-out rows"})
    return {"business_use": "credit decisions", "intended_population": "card holders",
            "sample_population": "the uploaded file", "evaluation_sets": sets, "missing_information": []} | overrides


def plan(**overrides) -> dict:
    """A validation plan that runs the checks needing no columns; `overrides` replaces any check."""
    def c(run, **kw):
        return {"run": run, "rationale": "relevant to the use" if run else "not relevant here", **kw}
    return {"baseline_comparison": c(True), "calibration": c(True), "subgroup_performance": c(False),
            "stability": c(True, folds=3), "sensitivity": c(False), "assumptions": c(False)} | overrides


def spec(**overrides) -> ModelSpec:
    """A valid spec; families other than binomial get count semantics, a null test set drops the test role."""
    base = dict(
        target_model="Probability of default for credit card holders",
        r_function="stats::glm", python_implementation="statsmodels.api.GLM", support_tier="exact",
        family="binomial", link="logit", formula="default ~ age + limit_bal", response="default",
        preprocessing=[], split={"method": "sample", "r_code": "idx <- sample(nrow(df), 0.7 * nrow(df))", "seed": 1},
        metrics=["auc", "gini"], unsupported_items=[], substitutions=[], r_packages=["stats"],
        baseline_r="df <- read.csv(data)\nfit <- glm(default ~ age + limit_bal, binomial, df[idx, ])",
        baseline_objects={"model": "fit", "train": "train", "test": "test"},
    )
    base |= BINARY if overrides.get("family", "binomial") in ("binomial", "quasibinomial") else COUNT | {"metrics": []}
    base |= overrides
    base.setdefault("validation_plan", plan())
    if "assessment" not in overrides:
        base["assessment"] = assessment(test=base["baseline_objects"]["test"] is not None)
    return ModelSpec.model_validate(base)


@pytest.mark.parametrize("role, version", [("analyst", "3"), ("diagnostician", "2"), ("reporter", "3"),
                                           ("translator", "2")])
def test_every_role_has_a_versioned_json_schema(role, version):
    js = ROLES[role].model_json_schema()
    assert js["properties"]["schema_version"]["const"] == version
    assert js["additionalProperties"] is False


def test_model_spec_rejects_unqualified_r_function_and_extra_fields():
    with pytest.raises(ValidationError):
        spec(r_function="glm")
    with pytest.raises(ValidationError):
        spec(surprise=True)


@pytest.mark.parametrize("change, error", [
    ({"python_implementation": "sklearn.linear_model.LogisticRegression"}, "scikit-learn implementations are excluded"),
    ({"support_tier": "unsupported"}, "limitation"),
    ({"event": None}, "event: required for a binary response"),
    ({"response_type": "grouped_binomial"}, "grouped_binomial: required"),
    ({"prediction": {"kind": "scalar", "quantity": "count", "units": "x", "alignment_keys": ["row"]}},
     "a binary model predicts a probability"),
    ({"prediction": {"kind": "class_probability", "quantity": "probability", "units": "x", "keys": ["a", "b"],
                     "alignment_keys": ["row"]}}, "alignment_keys: must be ['row', 'class']"),
    ({"prediction": {"kind": "time_indexed", "quantity": "probability", "units": "x",
                     "alignment_keys": ["row", "month"]}}, "assessment.period_column: required"),
    ({"metrics": ["pinball_loss"]}, "pinball_loss needs a quantile output"),
    ({"tolerances": {"per_metric": {"rmse": {"atol": 1, "rtol": 0}}}}, "['rmse'] are not in metrics"),
    ({"baseline_objects": {"model": "fit", "train": "train", "test": None}, "assessment": assessment(test=True)},
     "must list test exactly when baseline_objects.test is set"),
])
def test_semantics_are_explicit_and_consistent(change, error):
    with pytest.raises(ValidationError, match=error.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(")):
        spec(**change)


def test_tolerances_are_absolute_plus_relative_per_quantity():
    s = spec(tolerances={"predictions": {"atol": 0.01, "rtol": 0}, "per_metric": {"auc": {"atol": 1e-4, "rtol": 0}}})
    assert s.tolerances.predictions.atol == 0.01 and s.tolerances.coefficients.rtol == 1e-6
    with pytest.raises(ValidationError, match="cannot both be 0"):
        spec(tolerances={"coefficients": {"atol": 0, "rtol": 0}})


def test_convergence_and_metric_exceptions_require_explicit_contracts():
    with pytest.raises(ValidationError):
        spec(convergence={"mode": "not_applicable"})
    with pytest.raises(ValidationError):
        spec(convergence={"mode": "expression", "rationale": "solver status"})
    with pytest.raises(ValidationError):
        spec(optional_metrics={"test": ["mae"]})
    a = spec(convergence={"mode": "expression", "r_check": "model$success", "rationale": "solver success flag"})
    assert a.baseline_key() != spec().baseline_key()
    assert a.capture_config()["convergence"]["r_check"] == "model$success"


def test_grouped_binomial_and_quantile_specs_validate():
    spec(response_type="grouped_binomial", response="rate",
         grouped_binomial={"form": "proportion_and_trials", "trials": "loans"})
    with pytest.raises(ValidationError, match="successes: required"):
        spec(response_type="grouped_binomial", grouped_binomial={"form": "successes_and_trials", "trials": "n"})
    spec(r_function="quantreg::rq", python_implementation="statsmodels.api.QuantReg", support_tier="exact",
         family=None, link=None, response="loss", response_type="monetary_amount", event=None, metrics=["pinball_loss"],
         prediction={"kind": "quantile", "quantity": "monetary_amount", "units": "EUR", "keys": ["0.5", "0.9"],
                     "alignment_keys": ["row", "quantile"]})


def test_baseline_key_ignores_tolerances_and_metrics_but_covers_semantics():
    a, b = spec(), spec(metrics=["auc"], tolerances={"coefficients": {"atol": 0, "rtol": 1e-4}})
    assert canonical_hash(a.baseline_key()) == canonical_hash(b.baseline_key())
    assert canonical_hash(a) != canonical_hash(b)
    assert canonical_hash(spec(r_packages=["MASS"]).baseline_key()) != canonical_hash(a.baseline_key())
    recoded = spec(event={"meaning": "default", "complement": "non-default", "coding": "Yes"})
    assert canonical_hash(recoded.baseline_key()) != canonical_hash(a.baseline_key())


@pytest.mark.parametrize("change", [
    {"split": {"method": "sample", "r_code": "idx <- sample(nrow(df), 0.7 * nrow(df))", "seed": 2}},
    {"split": {"method": "sample", "r_code": "idx <- sample(nrow(df), 0.7 * nrow(df))", "seed": None}},
    {"response": "defaulted"},
    {"weights": {"expression": "w", "type": "frequency", "normalization": "none"}},
])
def test_baseline_key_covers_every_input_of_the_baseline_run(change):
    """The harness also consumes the seed, the response and its semantics, so changing any must not reuse a baseline."""
    assert canonical_hash(spec(**change).baseline_key()) != canonical_hash(spec().baseline_key())


def claims(text="Shown by {{v1.passed}}.", ids=("v1",)):
    return [{"text": text, "evidence_ids": list(ids)}]


def narrative(**overrides) -> dict:
    migration = {k: claims() for k in ("parity_results", "skipped_checks", "acceptance_decision")}
    validation = {k: claims() for k in ("performance", "calibration", "subgroup_performance", "stability",
                                        "sensitivity", "assumptions", "sample_suitability", "decision_performance")}
    return {"scope": claims(), "model_and_data_semantics": claims(), "migration": migration,
            "validation": validation, "missing_evidence": claims(), "limitations": claims(),
            "business_fitness": claims()} | overrides


def test_report_follows_the_template_and_claims_may_not_contain_literal_numbers():
    ReportNarrative.model_validate(narrative())
    with pytest.raises(ValidationError, match="placeholders"):
        ReportNarrative.model_validate(narrative(business_fitness=claims("The test AUC is 0.81.")))
    with pytest.raises(ValidationError):
        ReportNarrative.model_validate(narrative(limitations=[]))
    with pytest.raises(ValidationError):
        ReportNarrative.model_validate(narrative(scope=claims("Fine.", ())))
    missing = narrative()
    del missing["validation"]["calibration"]
    with pytest.raises(ValidationError):
        ReportNarrative.model_validate(missing)
    paths = [p for p, _ in ReportNarrative.model_validate(narrative()).sections()]
    assert paths.index("migration.acceptance_decision") < paths.index("validation.performance")
    assert paths[-1] == "business_fitness"


@pytest.mark.parametrize("change, error", [
    ({"subgroup_performance": {"run": True, "rationale": "segments"}}, "subgroup_performance.columns: name"),
    ({"sensitivity": {"run": True, "rationale": "pricing"}}, "sensitivity.columns: name"),
    ({"assumptions": {"run": True, "rationale": "glm"}}, "assumptions.assumptions: list them"),
    ({"assumptions": {"run": True, "rationale": "glm", "assumptions": ["zero_inflation"]}},
     "['zero_inflation'] do not apply to a binary response"),
    ({"stability": {"run": True, "rationale": "refits", "folds": 1}}, "greater than or equal to 2"),
    ({"calibration": {"run": False, "rationale": ""}}, "at least 1 character"),
])
def test_the_validation_plan_must_be_actionable_and_fit_the_response(change, error):
    with pytest.raises(ValidationError, match=re.escape(error)):
        spec(validation_plan=plan(**change))


def test_every_check_is_planned_or_excluded_with_a_rationale():
    body = spec().model_dump(mode="json")
    del body["validation_plan"]["sensitivity"]
    with pytest.raises(ValidationError, match="sensitivity"):
        ModelSpec.model_validate(body)
    assert spec(validation_plan=plan(assumptions={"run": True, "rationale": "logit", "assumptions": [
        "separation", "multicollinearity"]})).validation_plan.assumptions.run


def test_subgroup_columns_are_captured_with_the_baseline():
    grouped = spec(validation_plan=plan(subgroup_performance={"run": True, "rationale": "segments",
                                                              "columns": ["age"]}))
    assert grouped.capture_config()["segments"] == ["age"] and spec().capture_config()["segments"] == []
    assert canonical_hash(grouped.baseline_key()) != canonical_hash(spec().baseline_key())
    unchanged = spec(validation_plan=plan(stability={"run": True, "rationale": "refits", "folds": 7}))
    assert canonical_hash(unchanged.baseline_key()) == canonical_hash(spec().baseline_key())
