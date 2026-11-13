import pytest
from pydantic import ValidationError

from controlplane import support
from controlplane.schemas import ROLES, ModelSpec, ReportNarrative, canonical_hash


def spec(**overrides) -> ModelSpec:
    base = dict(
        target_model="Probability of default for credit card holders",
        r_function="stats::glm", statsmodels_counterpart="statsmodels.api.GLM", support_tier="exact",
        family="binomial", link="logit", formula="default ~ age + limit_bal", response="default",
        preprocessing=[], split={"method": "sample", "r_code": "idx <- sample(nrow(df), 0.7 * nrow(df))", "seed": 1},
        kpis=["auc", "gini"], unsupported_items=[], substitutions=[], r_packages=["stats"],
        baseline_r="df <- read.csv(data)\nfit <- glm(default ~ age + limit_bal, binomial, df[idx, ])",
        baseline_objects={"model": "fit", "train": "train", "test": "test"},
    )
    return ModelSpec.model_validate(base | overrides)


@pytest.mark.parametrize("role", sorted(ROLES))
def test_every_role_has_a_versioned_json_schema(role):
    js = ROLES[role].model_json_schema()
    assert js["properties"]["schema_version"]["const"] == "1"
    assert js["additionalProperties"] is False


def test_model_spec_rejects_unqualified_r_function_and_extra_fields():
    with pytest.raises(ValidationError):
        spec(r_function="glm")
    with pytest.raises(ValidationError):
        spec(surprise=True)


def test_baseline_key_ignores_tolerances_and_kpis():
    a, b = spec(), spec(kpis=["auc"], tolerances={"coef_rel": 1e-4})
    assert canonical_hash(a.baseline_key()) == canonical_hash(b.baseline_key())
    assert canonical_hash(a) != canonical_hash(b)
    assert canonical_hash(spec(r_packages=["MASS"]).baseline_key()) != canonical_hash(a.baseline_key())


def test_report_claims_may_not_contain_literal_numbers():
    ok = {"sections": [{"heading": "Parity", "claims": [
        {"text": "Coefficients match to {{v1.coef_rel}}.", "evidence_ids": ["v1"]}]}]}
    ReportNarrative.model_validate(ok)
    bad = {"sections": [{"heading": "Parity", "claims": [
        {"text": "The test AUC is 0.81.", "evidence_ids": ["v1"]}]}]}
    with pytest.raises(ValidationError, match="placeholders"):
        ReportNarrative.model_validate(bad)
    no_evidence = {"sections": [{"heading": "Parity", "claims": [{"text": "Fine.", "evidence_ids": []}]}]}
    with pytest.raises(ValidationError):
        ReportNarrative.model_validate(no_evidence)


def test_support_matrix_tiers():
    m = support.load()
    assert m.tier("stats::glm", "poisson") == "exact"
    assert m.tier("stats::glm", "tweedie") == "unsupported"
    assert m.tier("mgcv::gam") == "approximate"
    assert m.tier("randomForest::randomForest") == "unsupported"
    assert m.check(spec()) == []
    errors = m.check(spec(r_function="randomForest::randomForest"))
    assert "unsupported" in errors[0]
    assert "statsmodels_counterpart" in m.check(spec(statsmodels_counterpart="sklearn.LogisticRegression"))[0]


def test_pairs_are_unproven_until_a_fixture_exists(tmp_path):
    m = support.load()
    assert not any(p.proven for p in m.pairs)
    assert m.support(spec()).model_dump() == {"tier": "exact", "proven": False}
    assert m.support(spec(r_function="randomForest::randomForest")).model_dump() == {
        "tier": "unsupported", "proven": False}

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("version: 1\npairs:\n  - {r: stats::lm, statsmodels: [statsmodels.api.OLS], "
                      "tier: exact, fixture: lm_ols}\n")
    with pytest.raises(ValueError, match="fixtures not found"):
        support.load(matrix, tmp_path / "fixtures")
    (tmp_path / "fixtures" / "lm_ols").mkdir(parents=True)
    proven = support.load(matrix, tmp_path / "fixtures")
    assert proven.support(spec(r_function="stats::lm", family=None,
                               statsmodels_counterpart="statsmodels.api.OLS")).proven
