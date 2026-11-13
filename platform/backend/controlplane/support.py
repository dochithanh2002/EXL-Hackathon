"""The R -> statsmodels support matrix (`platform/envs/support_matrix.yaml`)."""
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
    statsmodels: list[str]
    tier: Literal["exact", "approximate"]
    families: list[str] | None = None
    fixture: str | None = None

    @property
    def proven(self) -> bool:
        return self.fixture is not None


class Support(BaseModel):
    tier: Tier
    proven: bool


class SupportMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int
    pairs: list[Pair]

    def lookup(self, r_function: str, family: str | None = None) -> Pair | None:
        for p in self.pairs:
            if p.r == r_function and (p.families is None or family in p.families):
                return p
        return None

    def tier(self, r_function: str, family: str | None = None) -> Tier:
        p = self.lookup(r_function, family)
        return p.tier if p else "unsupported"

    def support(self, spec: ModelSpec) -> Support:
        p = self.lookup(spec.r_function, spec.family)
        return Support(tier=p.tier, proven=p.proven) if p else Support(tier="unsupported", proven=False)

    def check(self, spec: ModelSpec) -> list[str]:
        """Disagreements between the Analyst's spec and the matrix; returned to the Analyst as validation errors."""
        expected = self.tier(spec.r_function, spec.family)
        errors = []
        if spec.support_tier != expected:
            what = spec.r_function + (f" (family {spec.family})" if spec.family else "")
            errors.append(f"support_tier is '{spec.support_tier}' but the support matrix lists {what} as '{expected}'")
        pair = self.lookup(spec.r_function, spec.family)
        if pair and spec.statsmodels_counterpart not in pair.statsmodels:
            errors.append(f"statsmodels_counterpart must be one of {pair.statsmodels}")
        return errors


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
