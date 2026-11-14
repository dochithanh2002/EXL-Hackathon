"""Typed control-plane results for the parity engine shared with standalone deliverables."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .runner.parity import compare as compare_outputs
from .schemas import ModelSpec


class Check(BaseModel):
    name: str
    value: float | None                  # None: could not be computed, which fails the check
    tolerance: float
    passed: bool


class Validation(BaseModel):
    passed: bool
    stage: Literal["runtime", "parity"]
    error: dict | None = None
    checks: list[Check] = []
    kpis: dict[str, dict[str, dict[str, float | None]]] = {}     # set -> kpi -> {"r", "python"}
    differences: dict = {}
    notes: list[str] = []

    def metrics(self) -> dict[str, float]:
        """Every check's value, for MLflow."""
        return {c.name.replace(" ", "_"): c.value for c in self.checks if c.value is not None
                and math.isfinite(c.value)}


def failed_run(error: dict) -> Validation:
    return Validation(passed=False, stage="runtime", error=error)


def compare(baseline: Path, candidate: Path, spec: ModelSpec) -> Validation:
    return Validation.model_validate(compare_outputs(baseline, candidate, spec.model_dump(mode="json")))
