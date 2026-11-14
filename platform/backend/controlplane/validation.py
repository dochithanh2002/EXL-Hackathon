"""Typed control-plane results: the parity engine shared with standalone deliverables, plus the financial
assessment, the validation plan's model evaluation and the design comparison, which are recorded with the
validation but never decide it."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .assessment import assess_both, design_comparison
from .evaluation import diagnostics_config, evaluate
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
    evaluation: dict = {}               # evaluation.evaluate: the validation plan's checks

    def metrics_for_mlflow(self) -> dict[str, float]:
        """The comparison run's metrics in three namespaces: `r/<set>/<metric>` and `python/<set>/<metric>` are
        each model's actual performance; `parity/<check>/ratio` (worst |python - r| in tolerance units) and
        `parity/<check>/max_abs_diff` are the migration's differences."""
        out = {}
        for s, metrics in self.metrics.items():
            for name, sides in metrics.items():
                for side in ("r", "python"):
                    out[f"{side}/{s}/{name}"] = sides.get(side)
        for c in self.checks:
            key = "parity/" + c.name.replace(" ", "_")
            out[f"{key}/ratio" if c.kind == "tolerance" else key] = c.value
            out[f"{key}/max_abs_diff"] = c.max_abs_diff
        return {k: float(v) for k, v in out.items() if v is not None and math.isfinite(v)}


def failed_run(error: dict) -> Validation:
    return Validation(passed=False, stage="runtime", error=error)


def compare(baseline: Path, candidate: Path, spec: ModelSpec) -> Validation:
    body = spec.model_dump(mode="json")
    v = compare_outputs(baseline, candidate, body)
    return Validation.model_validate(v | {"assessment": assess_both(baseline, candidate, body),
                                          "design": design_comparison(baseline, candidate)})


def with_evaluation(v: Validation, baseline: Path, candidate: Path, diagnostics: Path | None,
                    spec: ModelSpec) -> Validation:
    """The validation with the validation plan's checks; `diagnostics` is the Python diagnostics output, if run."""
    ev = evaluate(baseline, candidate, diagnostics, spec.model_dump(mode="json"), v.assessment)
    if diagnostics is None and diagnostics_config(spec.model_dump(mode="json")):
        ev["notes"].append("the Python diagnostics (cross-validation, sensitivity, variance inflation) run only "
                           "for a candidate that passes the parity checks")
    return v.model_copy(update={"evaluation": ev})
