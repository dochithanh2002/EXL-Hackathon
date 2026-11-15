"""Output contracts of the four agent roles. Each is submitted through one `submit_*` tool whose parameters are
the JSON Schema of the model below; a submission that fails validation goes back to the agent with the error."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Tier = Literal["exact", "approximate", "unsupported"]
QualifiedName = Field(pattern=r"^[A-Za-z][A-Za-z0-9.]*::[A-Za-z.][A-Za-z0-9._]*$",
                      description="Package-qualified R function, e.g. stats::glm")
SKLEARN = re.compile(r"sklearn|scikit", re.IGNORECASE)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- Analyst
class PreprocessingStep(Contract):
    order: int = Field(ge=1)
    description: str
    r_code: str = Field(description="The exact R line(s) from the uploaded code")
    columns: list[str] = Field(description="Column names affected")


class SplitDefinition(Contract):
    method: str = Field(description="How R forms the train and test sets, e.g. caret::createDataPartition")
    r_code: str
    seed: int | None = None


class BaselineObjects(Contract):
    model: str = Field(description="Name of the fitted model object in baseline.R")
    train: str = Field(description="Name of the training data frame in baseline.R")
    test: str | None = Field(description="Name of the test data frame in baseline.R, holding held-out rows of the "
                                         "dataset; null if R has no such set (a frame read from another file is not one)")


ResponseType = Literal["binary", "grouped_binomial", "count", "continuous", "monetary_amount", "categorical",
                       "ordinal"]
BINARY = ("binary", "grouped_binomial")


class Event(Contract):
    meaning: str = Field(description="What the modelled event is, e.g. 'default: 90+ days past due within 24 months'")
    complement: str = Field(description="What its absence means, e.g. 'non-default'")
    coding: str = Field(description="The response value that codes the event, as R reads it: '1' for a 0/1 or "
                                    "logical column, else the level, e.g. 'Yes'. It must be the outcome whose "
                                    "probability the R model predicts")


class GroupedBinomial(Contract):
    form: Literal["successes_and_trials", "proportion_and_trials"] = Field(
        description="cbind(successes, trials - successes) ~ ..., or a proportion response with weights = trials")
    successes: str | None = Field(default=None, description="R expression for the successes, evaluated in the set "
                                                            "frame; required for successes_and_trials")
    trials: str = Field(description="R expression for the number of trials, evaluated in the set frame")


class Weights(Contract):
    expression: str = Field(description="The weights argument exactly as R passes it, evaluated in the set frame")
    type: Literal["frequency", "sampling", "precision"] = Field(
        description="frequency: each row stands for that many identical observations; sampling: inverse selection "
                    "probabilities; precision: variance weights (inverse of each observation's variance)")
    normalization: Literal["none", "mean_one", "sum_one", "sum_n"] = Field(
        description="How baseline.R rescales the weights before fitting, if at all")


OutputKind = Literal["scalar", "class_probability", "quantile", "time_indexed"]
EXPECTED_KEYS = {"scalar": ["row"], "class_probability": ["row", "class"], "quantile": ["row", "quantile"]}


class Prediction(Contract):
    kind: OutputKind = Field(description="scalar: one value per row; class_probability: one probability per row "
                                         "and class; quantile: one value per row and quantile level; time_indexed: "
                                         "one value per row, each row a time point named by assessment.period_column")
    quantity: Literal["probability", "count", "frequency", "monetary_amount", "continuous"]
    units: str = Field(min_length=1, description="What one predicted value means, e.g. 'probability of default "
                                                 "within 24 months', 'claims per policy-year', 'EUR per claim'")
    exposure: str | None = Field(default=None, description="For a frequency: the exposure the rate is per, "
                                                           "e.g. Exposure in policy-years")
    keys: list[str] = Field(default=[], description="Class labels or quantile levels, in the column order of R's "
                                                    "prediction; empty for scalar and time_indexed outputs")
    alignment_keys: list[str] = Field(description="What identifies one predicted value besides the set: ['row'] "
                                                  "(scalar), ['row', 'class'], ['row', 'quantile'], or ['row', "
                                                  "<period column>] (time_indexed)")
    r_predict: str = Field(default='predict(model, newdata = newdata, type = "response", na.action = na.pass)',
                           description="R expression giving the predictions, with `model` and `newdata` bound: a "
                                       "numeric vector for scalar and time_indexed outputs, otherwise a matrix with "
                                       "one column per key")


class EvaluationSet(Contract):
    set: Literal["train", "test"]
    role: Literal["development", "holdout", "out_of_time"]
    description: str = Field(description="Which rows, periods or segments the set holds")


class Decision(Contract):
    threshold: float = Field(description="Cut-off on the prediction scale, as the business applies it")
    action: str = Field(description="What a prediction at or above the threshold triggers, e.g. 'decline'")


class Criteria(Contract):
    """Acceptance thresholds for the financial assessment. Only thresholds the code, the documents or the DS state."""
    min_events_per_parameter: float | None = Field(default=None, gt=0)
    max_psi: float | None = Field(default=None, gt=0)
    max_calibration_deviation: float | None = Field(default=None, gt=0, description="Largest |observed / expected "
                                                                                    "- 1| on an evaluation set")
    min_auc: float | None = Field(default=None, gt=0, le=1)
    min_events_captured: float | None = Field(default=None, ge=0, le=1,
        description="Minimum event recall at the stated decision threshold on independent data")
    min_impact_ratio: float | None = Field(default=None, gt=0, le=1,
        description="Fairness: lowest favourable-outcome rate of a protected group divided by the highest, at the "
                    "decision threshold (the four-fifths rule is 0.8)")
    max_group_calibration_deviation: float | None = Field(default=None, gt=0,
        description="Fairness: largest |observed / expected - 1| of a protected group on an evaluation set")


class Assessment(Contract):
    business_use: str = Field(description="The decision or process the model supports")
    intended_population: str = Field(description="Who the model is meant to score")
    sample_population: str = Field(description="What the data covers (product, channel, period, exclusions) as far "
                                               "as the code and documents say")
    evaluation_sets: list[EvaluationSet] = Field(min_length=1)
    period_column: str | None = Field(default=None, description="Column of the set frames that dates each row, for "
                                                                "stability by period")
    decision: Decision | None = None
    criteria: Criteria = Field(default_factory=Criteria)
    missing_information: list[str] = Field(description="What the DS must still provide (data, periods, thresholds, "
                                                       "business assumptions); never invent it")


Metric = Literal["auc", "gini", "ks", "actual_expected", "rmse", "mae", "deviance", "aic", "log_loss", "brier",
                 "pinball_loss"]
BINARY_METRICS = {"auc", "gini", "ks"}


class Tolerance(Contract):
    """|python - r| <= atol + rtol * |r|. atol is in the quantity's own units."""
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)

    @model_validator(mode="after")
    def _positive(self) -> Tolerance:
        if self.atol == 0 and self.rtol == 0:
            raise ValueError("atol and rtol cannot both be 0")
        return self


def _tol(atol: float, rtol: float):
    return Field(default_factory=lambda: Tolerance(atol=atol, rtol=rtol))


class Tolerances(Contract):
    coefficients: Tolerance = _tol(1e-8, 1e-6)
    std_errors: Tolerance = _tol(1e-8, 1e-6)
    predictions: Tolerance = Field(default_factory=lambda: Tolerance(atol=1e-8, rtol=1e-6),
                                   description="atol in the prediction's units, e.g. currency for amounts")
    fit_statistics: Tolerance = _tol(1e-8, 1e-6)
    metrics: Tolerance = _tol(1e-8, 1e-6)
    per_metric: dict[Metric, Tolerance] = Field(default={}, description="Overrides of `metrics` by metric name")


class Substitution(Contract):
    original: str
    proposed: str
    rationale: str


class Convergence(Contract):
    mode: Literal["auto", "expression", "not_applicable"] = "auto"
    r_check: str | None = Field(default=None, description="R expression returning one logical solver-success value; "
        "model and fit_warnings are bound. Inspect actual solver evidence, never return a constant true")
    rationale: str | None = Field(default=None, description="Required for a custom check or an explicit exemption")

    @model_validator(mode="after")
    def _contract(self) -> Convergence:
        if self.mode != "auto" and not (self.rationale or "").strip():
            raise ValueError("custom convergence handling requires a rationale")
        if (self.mode == "expression") != bool((self.r_check or "").strip()):
            raise ValueError("r_check is required exactly for expression mode")
        return self


class PlannedCheck(Contract):
    run: bool = Field(description="Whether the check applies to this prediction task and business use")
    rationale: str = Field(min_length=1, description="Why it matters here, or why it is left out")


class SubgroupCheck(PlannedCheck):
    columns: list[str] = Field(default=[], description="Columns of R's set frames that define the subgroups the "
                                                       "business cares about (segments, products, regions, bands); "
                                                       "numeric columns are cut into quintiles of the development set")
    min_rows: int = Field(default=100, ge=1, description="Subgroups with fewer rows are reported as too small")


class StabilityCheck(PlannedCheck):
    folds: int = Field(default=5, ge=2, le=10, description="Cross-validation refits of the Python model on the train "
                                                           "rows, with folds assigned deterministically by row")


class SensitivityCheck(PlannedCheck):
    columns: list[str] = Field(default=[], description="Numeric predictor columns of the frames the Python model "
                                                       "reads, each shifted down and up while all else is held fixed")
    shift_sd: float = Field(default=0.1, gt=0, le=2, description="Shift size in training standard deviations")


Assumption = Literal["overdispersion", "zero_inflation", "multicollinearity", "residual_distribution",
                     "heteroscedasticity", "separation"]
ASSUMPTIONS_FOR: dict[str, set[str]] = {
    "binary": {"separation", "multicollinearity"},
    "grouped_binomial": {"overdispersion", "separation", "multicollinearity"},
    "count": {"overdispersion", "zero_inflation", "multicollinearity"},
    "continuous": {"residual_distribution", "heteroscedasticity", "multicollinearity"},
    "monetary_amount": {"overdispersion", "residual_distribution", "heteroscedasticity", "multicollinearity"},
    "categorical": {"multicollinearity"},
    "ordinal": {"multicollinearity"},
}


class AssumptionCheck(PlannedCheck):
    assumptions: list[Assumption] = Field(default=[], description="Assumptions of this model to screen; only those "
                                                                  "relevant to its response type and family")


class FairnessCheck(PlannedCheck):
    columns: list[str] = Field(default=[], description="Protected attributes, or close proxies, among the columns of "
                                                       "R's set frames (sex, age, marital status, ...); numeric "
                                                       "columns are cut into quintiles of the development set")
    min_rows: int = Field(default=100, ge=1, description="Groups with fewer rows are reported but not judged")
    favourable_outcome: Literal["below_threshold", "at_or_above_threshold"] | None = Field(
        default=None, description="Which side of the decision threshold is favourable to the person scored, e.g. "
                                  "below_threshold when a prediction at or above it declines")


class ValidationPlan(Contract):
    """Model validation beyond migration parity. The Analyst decides which checks apply and why; the platform runs
    them on R's and Python's predictions and records what each could not establish."""
    baseline_comparison: PlannedCheck = Field(description="Performance against a constant comparator fitted on the "
                                                          "development set")
    calibration: PlannedCheck = Field(description="Observed against predicted: overall, by decile, slope and "
                                                  "intercept")
    subgroup_performance: SubgroupCheck
    stability: StabilityCheck = Field(description="Score stability across sets and periods, cross-validated "
                                                  "metrics and coefficients")
    sensitivity: SensitivityCheck
    assumptions: AssumptionCheck
    fairness: FairnessCheck | None = Field(default=None, description="Outcomes and calibration by protected group. "
                                                                     "Required in new specifications; older ones "
                                                                     "without it record fairness as not planned")

    def checks(self) -> list[tuple[str, PlannedCheck]]:
        return [(name, c) for name in ValidationPlan.model_fields if (c := getattr(self, name)) is not None]


class ModelSpec(Contract):
    schema_version: Literal["3"] = "3"
    target_model: str = Field(description="What the model predicts and for whom, in one sentence")
    r_function: str = QualifiedName
    python_implementation: str = Field(description="The Python estimator that reproduces it, e.g. "
                                                   "statsmodels.api.GLM; scikit-learn is excluded")
    support_tier: Tier
    limitation: str | None = Field(default=None, description="Why no faithful Python implementation is possible; "
                                                             "required when support_tier is unsupported")
    family: str | None = Field(default=None, description="R family name, e.g. poisson")
    link: str | None = None
    formula: str = Field(description="Model formula exactly as R fits it")
    response: str = Field(description="Response column of the set frames")
    response_type: ResponseType
    event: Event | None = Field(default=None, description="Required for binary and grouped_binomial responses")
    grouped_binomial: GroupedBinomial | None = Field(default=None, description="Required for grouped_binomial")
    offset: str | None = Field(default=None, description="Offset expression, e.g. log(Exposure)")
    weights: Weights | None = None
    prediction: Prediction
    convergence: Convergence = Field(default_factory=Convergence)
    preprocessing: list[PreprocessingStep]
    split: SplitDefinition
    assessment: Assessment
    validation_plan: ValidationPlan
    metrics: list[Metric] = Field(description="Evaluation metrics compared between R and Python on every set")
    optional_metrics: dict[Literal["train", "test"], list[Metric]] = Field(default={},
        description="Explicit per-set exceptions: unavailable metrics do not block acceptance. All requested "
                    "metrics are required unless listed here before spec confirmation")
    tolerances: Tolerances = Field(default_factory=Tolerances)
    unsupported_items: list[str] = Field(description="Constructs in the uploaded code that cannot be translated")
    substitutions: list[Substitution]
    r_packages: list[str] = Field(description="R packages baseline.R loads")
    baseline_r: str = Field(description="Minimal slice of the uploaded code that produces the target model. It reads "
                                        "the dataset as inputs/<file name> with read.csv or read.table and keeps R's "
                                        "row names, which trace every row back to the dataset file")
    baseline_objects: BaselineObjects

    @model_validator(mode="after")
    def _semantics(self) -> ModelSpec:
        errors = []
        if SKLEARN.search(self.python_implementation):
            errors.append("python_implementation: scikit-learn implementations are excluded")
        if self.support_tier == "unsupported" and not (self.limitation or "").strip():
            errors.append("limitation: explain why no faithful Python implementation is possible")
        if self.response_type in BINARY and self.event is None:
            errors.append(f"event: required for a {self.response_type} response")
        if (self.response_type == "grouped_binomial") != (self.grouped_binomial is not None):
            errors.append("grouped_binomial: required exactly when response_type is grouped_binomial")
        g = self.grouped_binomial
        if g and g.form == "successes_and_trials" and not (g.successes or "").strip():
            errors.append("grouped_binomial.successes: required for the successes_and_trials form")
        p, a = self.prediction, self.assessment
        if p.kind in ("class_probability", "quantile") and not p.keys:
            errors.append(f"prediction.keys: list the {'classes' if p.kind == 'class_probability' else 'quantiles'}")
        if p.kind in ("scalar", "time_indexed") and p.keys:
            errors.append(f"prediction.keys: must be empty for a {p.kind} output")
        if p.kind == "quantile":
            try:
                if not all(0 < float(k) < 1 for k in p.keys):
                    raise ValueError
            except ValueError:
                errors.append("prediction.keys: quantile levels must be numbers strictly between 0 and 1")
        if p.kind == "class_probability" and self.response_type not in ("categorical", "ordinal", "binary"):
            errors.append("prediction.kind: class probabilities need a categorical, ordinal or binary response")
        if p.kind in ("scalar", "time_indexed") and self.response_type in BINARY and p.quantity != "probability":
            errors.append(f"prediction.quantity: a {self.response_type} model predicts a probability")
        if p.kind == "time_indexed" and not a.period_column:
            errors.append("assessment.period_column: required for a time_indexed output")
        expected = EXPECTED_KEYS.get(p.kind, ["row", a.period_column or "<period column>"])
        if p.alignment_keys != expected:
            errors.append(f"prediction.alignment_keys: must be {expected} for a {p.kind} output")
        sets = [e.set for e in a.evaluation_sets]
        if len(sets) != len(set(sets)) or "train" not in sets:
            errors.append("assessment.evaluation_sets: list train once, and test once if R has a test set")
        if ("test" in sets) != (self.baseline_objects.test is not None):
            errors.append("assessment.evaluation_sets: must list test exactly when baseline_objects.test is set")
        if bad := sorted(BINARY_METRICS & set(self.metrics)) if self.response_type not in BINARY else []:
            errors.append(f"metrics: {bad} need a binary or grouped_binomial response")
        if "pinball_loss" in self.metrics and p.kind != "quantile":
            errors.append("metrics: pinball_loss needs a quantile output")
        if extra := sorted(set(self.tolerances.per_metric) - set(self.metrics)):
            errors.append(f"tolerances.per_metric: {extra} are not in metrics")
        for name, metrics in self.optional_metrics.items():
            if name not in sets or not set(metrics) <= set(self.metrics):
                errors.append("optional_metrics must name an evaluation set and requested metrics")
        vp = self.validation_plan
        for name in ("subgroup_performance", "sensitivity"):
            if getattr(vp, name).run and not getattr(vp, name).columns:
                errors.append(f"validation_plan.{name}.columns: name the columns, or set run to false")
        if vp.fairness and vp.fairness.run and not vp.fairness.columns:
            errors.append("validation_plan.fairness.columns: name the protected attributes, or set run to false")
        if vp.fairness and vp.fairness.favourable_outcome and a.decision is None:
            errors.append("validation_plan.fairness.favourable_outcome: needs assessment.decision")
        if vp.assumptions.run:
            allowed = ASSUMPTIONS_FOR[self.response_type]
            if not vp.assumptions.assumptions:
                errors.append("validation_plan.assumptions.assumptions: list them, or set run to false")
            if bad := sorted(set(vp.assumptions.assumptions) - allowed):
                errors.append(f"validation_plan.assumptions.assumptions: {bad} do not apply to a "
                              f"{self.response_type} response; choose from {sorted(allowed)}")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def baseline_key(self) -> dict:
        """Every spec field the R baseline run consumes. Equal keys mean a confirmed baseline can be reused."""
        return {"baseline_r": self.baseline_r, "baseline_objects": self.baseline_objects.model_dump(),
                "r_packages": sorted(self.r_packages), "seed": self.split.seed, **self.capture_config()}

    def capture_config(self) -> dict:
        """The semantics the R capture harness and the Python harness both apply to the set frames."""
        return {"response": self.response, "response_type": self.response_type,
                "event_coding": self.event.coding if self.event else None,
                "grouped": self.grouped_binomial.model_dump() if self.grouped_binomial else None,
                "weights": self.weights.expression if self.weights else None,
                "prediction": {"kind": self.prediction.kind, "keys": self.prediction.keys,
                               "r_predict": self.prediction.r_predict},
                "convergence": self.convergence.model_dump(),
                "period_column": self.assessment.period_column,
                "segments": self.segment_columns()}

    def segment_columns(self) -> list[str]:
        """The set-frame columns capture.R exports per row: the subgroup columns, then the protected attributes."""
        vp = self.validation_plan
        columns = vp.subgroup_performance.columns if vp.subgroup_performance.run else []
        return list(dict.fromkeys([*columns, *(vp.fairness.columns if vp.fairness else [])]))


# ---------------------------------------------------------------- Translator
class Candidate(Contract):
    schema_version: Literal["2"] = "2"
    module: str = Field(description="Complete Python module implementing the candidate interface")
    notes: list[str]
    limitations: list[str] = Field(default=[], description="Anything the module cannot reproduce faithfully, and "
                                                           "why; never hidden by a different model")
    changes: str = Field(default="", description="For repairs: what changed and why")


# ---------------------------------------------------------------- Diagnostician
class Diagnosis(Contract):
    schema_version: Literal["2"] = "2"
    category: Literal["migration_defect", "specification", "not_reproducible", "model_fitness"] = Field(
        description="migration_defect: the code departs from baseline.R (goes to the Translator); specification: the "
                    "spec misdescribes baseline.R or its tolerances; not_reproducible: no faithful Python "
                    "implementation can match R; model_fitness: the migration is faithful but the DS's concern is "
                    "the model itself. All but migration_defect go to the DS")
    root_cause: str
    evidence: list[str] = Field(min_length=1, description="Observed differences or errors supporting the root cause")
    repair_instructions: list[str] = Field(description="What the Translator must change in the candidate")
    assessment_findings: list[str] = Field(default=[], description="What the model validation checks show "
                                                                   "(each check's results and criterion outcome), "
                                                                   "or what a check could not establish")
    spec_change_needed: bool = Field(description="True if the fix needs the DS (a new ModelSpec) rather than new code")
    proposed_spec_change: str | None = Field(default=None, description="Required when spec_change_needed is true")
    limitation: str | None = Field(default=None, description="Required for not_reproducible: what blocks a "
                                                             "faithful implementation")

    @model_validator(mode="after")
    def _actionable(self) -> Diagnosis:
        if self.spec_change_needed != (self.category != "migration_defect"):
            raise ValueError("spec_change_needed must be false exactly for a migration_defect")
        if self.spec_change_needed and not (self.proposed_spec_change or "").strip():
            raise ValueError("proposed_spec_change is required when spec_change_needed is true")
        if not self.spec_change_needed and not self.repair_instructions:
            raise ValueError("repair_instructions must not be empty unless spec_change_needed is true")
        if self.category == "not_reproducible" and not (self.limitation or "").strip():
            raise ValueError("limitation is required for not_reproducible")
        return self


# ---------------------------------------------------------------- Reporter
PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_:-]+)\.([A-Za-z0-9_.:\[\]-]+)\}\}")
CODE_SPAN = re.compile(r"`[^`\n]+`")


def quote(text: str, start: int, end: int, context: int = 12) -> str:
    """text[start:end] with up to `context` characters of whole words around it, quoted, e.g. …"in 2019 the"…"""
    a, b = max(0, start - context), min(len(text), end + context)
    while a < start and a > 0 and not text[a - 1].isspace():
        a += 1
    while b > end and b < len(text) and not text[b].isspace():
        b -= 1
    fragment = text[a:b].strip()
    return ("…" if a > 0 else "") + f'"{fragment}"' + ("…" if b < len(text) else "")


def _no_literal_numbers(v: str) -> str:
    """Digits may appear only in {{evidence_id.field}} placeholders and in `code spans`, whose names the Reporter's
    checks match against the project's known names. The error quotes the text around the first literal number."""
    blank = lambda m: " " * len(m.group(0))  # noqa: E731
    masked = CODE_SPAN.sub(blank, PLACEHOLDER.sub(blank, v))
    at = next((i for i, c in enumerate(masked) if c.isnumeric()), None)
    if at is not None:
        raise ValueError(f"literal number in {quote(v, at, at + 1)}: numbers must be {{{{evidence_id.field}}}} placeholders; "
                         "write names that contain digits in `backticks`; change only this fragment")
    return v


class Claim(Contract):
    text: str = Field(description="Prose with no literal numbers: numbers are {{evidence_id.field}} placeholders, "
                                  "and names that contain digits are written in `backticks`")
    evidence_ids: list[str] = Field(min_length=1, description="Every evidence id the claim relies on, including "
                                                              "those of its placeholders")

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        return _no_literal_numbers(v)


class ReportNarrative(Contract):
    """The Reporter's interpretation; code renders every factual section of the report from the evidence."""
    schema_version: Literal["4"] = "4"
    key_findings: list[Claim] = Field(min_length=1, max_length=6, description="What the evidence shows, most "
                                      "important first: the migration's acceptance and the business fitness verdict "
                                      "as separate findings, and what the checks found")
    limitations: list[Claim] = Field(default=[], max_length=6, description="What the evidence cannot show, and why")
    recommendations: list[Claim] = Field(default=[], max_length=6, description="Actions that follow from cited "
                                         "evidence: criteria to set, checks that failed or could not run, "
                                         "unavailable parity checks")

    def sections(self) -> list[tuple[str, list[Claim]]]:
        """(field, claims) in report order."""
        return [("key_findings", self.key_findings), ("limitations", self.limitations),
                ("recommendations", self.recommendations)]


ROLES: dict[str, type[Contract]] = {
    "analyst": ModelSpec,
    "translator": Candidate,
    "diagnostician": Diagnosis,
    "reporter": ReportNarrative,
}
