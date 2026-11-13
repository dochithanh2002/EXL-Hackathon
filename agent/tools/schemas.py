"""Typed contracts for every LLM step. LLM output that fails validation is rejected (guardrail G-10)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PreprocStep(BaseModel):
    order: int
    r_code: str = Field(description="The exact R line(s)")
    description: str
    columns: list[str] = Field(description="Column NAMES affected (positional refs resolved)")
    rule_ids: list[str]


class ModelDef(BaseModel):
    r_function: str
    family: str
    link: str
    response: str
    formula: str
    terms: list[str] = Field(description="Coefficient names exactly as R reports them")
    factor_levels: dict[str, list[str]]


class Analysis(BaseModel):
    summary: str
    preprocessing: list[PreprocStep]
    model: ModelDef
    randomness: str = Field(description="How the data is split / randomised in R")
    protected_attributes_used: list[str]
    risks: list[str]
    escalations: list[str] = Field(description="Questions a human must answer before migration can continue; empty if none")


class PlanStep(BaseModel):
    order: int
    r_construct: str
    python_approach: str
    rule_ids: list[str]


class Plan(BaseModel):
    steps: list[PlanStep]
    packages: list[str]
    unmapped: list[str] = Field(description="R constructs with no confident mapping (triggers escalation)")
    escalations: list[str]


class CodeCandidate(BaseModel):
    python_code: str
    rule_ids_applied: list[str]
    notes: list[str]
    change_summary: str = Field(default="", description="For repairs: what changed and why")


Verdict = Literal["PASS", "FAIL"]


# ---------------------------------------------------------------- Documentation agent
class DocVariable(BaseModel):
    name: str = Field(description="Variable name exactly as it appears in the model formula")
    role: Literal["input", "target"]
    kind: Literal["numeric", "categorical", "ordinal_as_numeric", "binary"]
    description: str = Field(description="Business meaning, from source comments / supporting documents")
    levels: list[str] = Field(default_factory=list, description="Categorical/binary: R factor levels in R order")
    level_meaning: dict[str, str] = Field(default_factory=dict, description="Level code -> meaning, if documented")
    transformation: str = Field(description="What the code does to this variable before fitting; 'none' if nothing")
    source_lines: list[int] = Field(description="Line numbers in the numbered legacy source that define or use it")


class DocStep(BaseModel):
    order: int
    description: str
    columns: list[str]
    source_lines: list[int]
    author_rationale: str = Field(description="The author's stated reason, from comments or documents; '' if none")
    is_assumption: bool = Field(description="True if the step rests on an unverified assumption by the author")


class DocLimitation(BaseModel):
    text: str
    severity: Literal["High", "Medium", "Low"]
    disclosure_ids: list[str] = Field(description="Required disclosure ids (D-xx) this limitation covers; [] if none")
    source_lines: list[int] = Field(default_factory=list)


class DocExtract(BaseModel):
    purpose: str
    intended_use: str
    out_of_scope_use: str
    methodology: str = Field(description="Estimation method in plain words; no statistics")
    target_definition: str
    variables: list[DocVariable] = Field(description="EVERY formula variable plus the target, nothing else")
    preprocessing: list[DocStep] = Field(description="In source execution order")
    variable_selection: str
    assumptions: list[str]
    limitations: list[DocLimitation]
    code_vs_docs_discrepancies: list[str]
    open_questions: list[str] = Field(description="Questions for the model owner; facts that cannot be established from the evidence")
