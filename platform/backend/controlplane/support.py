"""Known R -> Python pairs (`platform/envs/support_matrix.yaml`). The matrix is a reference, not an allowlist: a
model outside it is still attempted, at the tier the Analyst proposes, and the DS must accept the parity risk. A
known pair fixes the tier and, once it has a passing fixture, counts as proven. scikit-learn is never used."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from .schemas.agents import ModelSpec, Tier

MATRIX_PATH = Path(__file__).resolve().parents[2] / "envs" / "support_matrix.yaml"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "parity"


class Pair(BaseModel):
    """`tier` is the parity the pair is expected to reach. It counts as proven only once `fixture` names a parity
    fixture that passes; until then the DS has to accept the parity risk, even for an exact pair."""
    model_config = ConfigDict(extra="forbid")
    r: str
    python: list[str]
    tier: Literal["exact", "approximate"]
    families: list[str] | None = None
    fixture: str | None = None

    @property
    def proven(self) -> bool:
        return self.fixture is not None


class Support(BaseModel):
    tier: Tier
    proven: bool
    known: bool


class SupportMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int
    pairs: list[Pair]

    def lookup(self, r_function: str, family: str | None = None) -> Pair | None:
        for p in self.pairs:
            if p.r == r_function and (p.families is None or family in p.families):
                return p
        return None

    def support(self, spec: ModelSpec) -> Support:
        """A known pair with its listed implementation sets the tier; anything else keeps the Analyst's tier and is
        unproven."""
        p = self.lookup(spec.r_function, spec.family)
        if p and spec.support_tier != "unsupported" and spec.python_implementation in p.python:
            return Support(tier=p.tier, proven=p.proven, known=True)
        return Support(tier=spec.support_tier, proven=False, known=False)

    def check(self, spec: ModelSpec) -> list[str]:
        """Disagreements with a known pair; returned to the Analyst as validation errors."""
        p = self.lookup(spec.r_function, spec.family)
        if p and spec.python_implementation in p.python and spec.support_tier not in (p.tier, "unsupported"):
            return [f"support_tier is '{spec.support_tier}' but {spec.r_function} with {spec.python_implementation} "
                    f"is a known '{p.tier}' pair"]
        return []


@lru_cache
def load(path: Path = MATRIX_PATH, fixture_dir: Path = FIXTURE_DIR) -> SupportMatrix:
    m = SupportMatrix.model_validate(yaml.safe_load(path.read_text()))
    keys = [(p.r, f) for p in m.pairs for f in (p.families or [None])]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate R function / family entries")
    missing = [p.fixture for p in m.pairs if p.fixture and not (fixture_dir / p.fixture).is_dir()]
    if missing:
        raise ValueError(f"{path}: fixtures not found in {fixture_dir}: {missing}")
    return m
