"""Typed control-plane results: the parity engine shared with standalone deliverables, plus the financial
assessment and design comparison, which are recorded with the validation but never decide it."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .assessment import assess_both, design_comparison
from .runner.parity import compare as compare_outputs
from .schemas import ModelSpec


class Check(BaseModel):
    name: str
    quantity: str = ""
    kind: Literal["tolerance", "count", "flag", "minimum"] = "tolerance"
    value: float | None                  # tolerance: |py - r| / (atol + rtol |r|), worst element; None: unavailable
    tolerance: float
    status: Literal["passed", "failed", "unavailable"] = "failed"
    passed: bool
    required: bool = True
    reason: str = ""
    atol: float | None = None
    rtol: float | None = None
    max_abs_diff: float | None = None
    compared: int | None = None
    missing_both: int | None = None
    missing_one_side: int | None = None
    nonfinite_mismatch: int | None = None
    total: int | None = None


class Validation(BaseModel):
    passed: bool
    stage: Literal["runtime", "parity"]
    error: dict | None = None
    checks: list[Check] = []
    metrics: dict[str, dict[str, dict[str, float | None]]] = {}     # set -> metric -> {"r", "python"}
    differences: dict = {}
    notes: list[str] = []
    assessment: dict = {}               # "r" and "python" -> question -> status, needs, criterion, meets, results
    design: dict = {}

    def metrics_for_mlflow(self) -> dict[str, float]:
        """Every check's value, for MLflow."""
        return {c.name.replace(" ", "_"): c.value for c in self.checks if c.value is not None
                and math.isfinite(c.value)}


def failed_run(error: dict) -> Validation:
    return Validation(passed=False, stage="runtime", error=error)


def compare(baseline: Path, candidate: Path, spec: ModelSpec) -> Validation:
    body = spec.model_dump(mode="json")
    v = compare_outputs(baseline, candidate, body)
    return Validation.model_validate(v | {"assessment": assess_both(baseline, candidate, body),
                                          "design": design_comparison(baseline, candidate)})
