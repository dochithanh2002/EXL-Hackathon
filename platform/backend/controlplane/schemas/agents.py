"""Output contracts of the four agent roles. Each is submitted through one `submit_*` tool whose parameters are
the JSON Schema of the model below; a submission that fails validation goes back to the agent with the error."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Tier = Literal["exact", "approximate", "unsupported"]
QualifiedName = Field(pattern=r"^[A-Za-z][A-Za-z0-9.]*::[A-Za-z.][A-Za-z0-9._]*$",
                      description="Package-qualified R function, e.g. stats::glm")


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
    test: str | None = Field(description="Name of the test data frame in baseline.R; null if R has no test set")


Kpi = Literal["auc", "gini", "actual_expected", "rmse", "mae", "deviance", "aic", "log_loss"]


class Tolerances(Contract):
    coef_rel: float = Field(default=1e-6, gt=0)
    se_rel: float = Field(default=1e-6, gt=0)
    pred_abs: float = Field(default=1e-8, gt=0)
    fit_stat_rel: float = Field(default=1e-6, gt=0)
    kpi_abs: float = Field(default=1e-6, gt=0)


class Substitution(Contract):
    original: str
    proposed: str
    rationale: str


class ModelSpec(Contract):
    schema_version: Literal["1"] = "1"
    target_model: str = Field(description="What the model predicts and for whom, in one sentence")
    r_function: str = QualifiedName
    statsmodels_counterpart: str = Field(description="e.g. statsmodels.api.GLM")
    support_tier: Tier
    family: str | None = Field(default=None, description="R family name, e.g. poisson")
    link: str | None = None
    formula: str = Field(description="Model formula exactly as R fits it")
    response: str
    offset: str | None = Field(default=None, description="Offset expression, e.g. log(Exposure)")
    weights: str | None = None
    preprocessing: list[PreprocessingStep]
    split: SplitDefinition
    kpis: list[Kpi]
    tolerances: Tolerances = Field(default_factory=Tolerances)
    unsupported_items: list[str] = Field(description="Constructs in the uploaded code that cannot be translated")
    substitutions: list[Substitution]
    r_packages: list[str] = Field(description="R packages baseline.R loads")
    baseline_r: str = Field(description="Minimal slice of the uploaded code that produces the target model. It reads "
                                        "the dataset as inputs/<file name> with read.csv or read.table and keeps R's "
                                        "row names, which trace every row back to the dataset file")
    baseline_objects: BaselineObjects

    def baseline_key(self) -> dict:
        """Every spec field the R baseline run consumes. Equal keys mean a confirmed baseline can be reused."""
        return {"baseline_r": self.baseline_r, "baseline_objects": self.baseline_objects.model_dump(),
                "r_packages": sorted(self.r_packages), "seed": self.split.seed, "response": self.response}


# ---------------------------------------------------------------- Translator
class Candidate(Contract):
    schema_version: Literal["1"] = "1"
    module: str = Field(description="Complete Python module implementing the candidate interface")
    notes: list[str]
    changes: str = Field(default="", description="For repairs: what changed and why")


# ---------------------------------------------------------------- Diagnostician
class Diagnosis(Contract):
    schema_version: Literal["1"] = "1"
    root_cause: str
    evidence: list[str] = Field(min_length=1, description="Observed differences or errors supporting the root cause")
    repair_instructions: list[str]
    spec_change_needed: bool = Field(description="True if the fix needs a new ModelSpec rather than new code")
    proposed_spec_change: str | None = None


# ---------------------------------------------------------------- Reporter
PLACEHOLDER = re.compile(r"\{\{[A-Za-z0-9_.:-]+\}\}")


class Claim(Contract):
    text: str = Field(description="Prose with no digits; numbers are written as {{evidence_id.field}} placeholders")
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _no_literal_numbers(cls, v: str) -> str:
        if re.search(r"\d", PLACEHOLDER.sub("", v)):
            raise ValueError("numbers must be {{evidence_id.field}} placeholders, not literals")
        return v


class Section(Contract):
    heading: str
    claims: list[Claim] = Field(min_length=1)


class ReportNarrative(Contract):
    schema_version: Literal["1"] = "1"
    sections: list[Section] = Field(min_length=1)


ROLES: dict[str, type[Contract]] = {
    "analyst": ModelSpec,
    "translator": Candidate,
    "diagnostician": Diagnosis,
    "reporter": ReportNarrative,
}
