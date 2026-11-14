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
    if "assessment" not in overrides:
        base["assessment"] = assessment(test=base["baseline_objects"]["test"] is not None)
    return ModelSpec.model_validate(base)


@pytest.mark.parametrize("role", sorted(ROLES))
def test_every_role_has_a_versioned_json_schema(role):
    js = ROLES[role].model_json_schema()
    assert js["properties"]["schema_version"]["const"] == "2"
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
    fa = {q: claims() for q in ("sample_suitability", "stability", "calibration", "decision_performance")}
    return {"scope": claims(), "model_and_data_semantics": claims(), "parity_results": claims(),
            "financial_assessment": fa, "skipped_checks": claims(), "limitations": claims(),
            "acceptance_decision": claims()} | overrides


def test_report_follows_the_template_and_claims_may_not_contain_literal_numbers():
    ReportNarrative.model_validate(narrative())
    with pytest.raises(ValidationError, match="placeholders"):
        ReportNarrative.model_validate(narrative(parity_results=claims("The test AUC is 0.81.")))
    with pytest.raises(ValidationError):
        ReportNarrative.model_validate(narrative(limitations=[]))
    with pytest.raises(ValidationError):
        ReportNarrative.model_validate(narrative(scope=claims("Fine.", ())))
    missing = narrative()
    del missing["financial_assessment"]["calibration"]
    with pytest.raises(ValidationError):
        ReportNarrative.model_validate(missing)


def test_diagnosis_category_decides_who_acts():
    base = dict(root_cause="r", evidence=["e"], repair_instructions=["fix"], spec_change_needed=False,
                category="migration_defect")
    Diagnosis.model_validate(base)
    with pytest.raises(ValidationError, match="spec_change_needed must be false exactly"):
        Diagnosis.model_validate(base | {"category": "model_fitness"})
    with pytest.raises(ValidationError, match="limitation is required"):
        Diagnosis.model_validate(base | {"category": "not_reproducible", "spec_change_needed": True,
                                         "proposed_spec_change": "stop"})


def test_support_matrix_is_a_reference_not_an_allowlist():
    m = support.load()
    assert m.check(spec()) == []
    assert m.support(spec()).model_dump() == {"tier": "exact", "proven": False, "known": True}
    assert "known 'exact' pair" in m.check(spec(support_tier="approximate"))[0]
    # a model outside the matrix is attempted at the Analyst's tier, unproven
    other = spec(r_function="betareg::betareg", python_implementation="scipy.optimize.minimize",
                 support_tier="approximate")
    assert m.check(other) == [] and m.support(other).model_dump() == {
        "tier": "approximate", "proven": False, "known": False}
    # a known R function with a different implementation keeps the Analyst's tier and is unproven
    custom = spec(python_implementation="scipy.optimize.minimize", support_tier="approximate")
    assert m.support(custom).model_dump() == {"tier": "approximate", "proven": False, "known": False}
    limited = spec(r_function="randomForest::randomForest", support_tier="unsupported", python_implementation="none",
                   limitation="the forest's bootstrap and split search cannot be reproduced without scikit-learn")
    assert m.support(limited).tier == "unsupported"


def test_pairs_are_unproven_until_a_fixture_exists(tmp_path):
    m = support.load()
    assert not any(p.proven for p in m.pairs)

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("version: 2\npairs:\n  - {r: stats::lm, python: [statsmodels.api.OLS], "
                      "tier: exact, fixture: lm_ols}\n")
    with pytest.raises(ValueError, match="fixtures not found"):
        support.load(matrix, tmp_path / "fixtures")
    (tmp_path / "fixtures" / "lm_ols").mkdir(parents=True)
    proven = support.load(matrix, tmp_path / "fixtures")
    assert proven.support(spec(r_function="stats::lm", family=None, link=None,
                               python_implementation="statsmodels.api.OLS", response_type="continuous",
                               prediction={"kind": "scalar", "quantity": "continuous", "units": "EUR",
                                           "alignment_keys": ["row"]}, metrics=[])).proven
