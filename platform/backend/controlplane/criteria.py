"""Default acceptance criteria (`platform/envs/criteria_defaults.yaml`) by response type. The Analyst may propose
them when the code, the documents and the DS state none; they bind only once the DS confirms the spec."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import get_args

import yaml
from pydantic import BaseModel, ConfigDict

from .schemas.agents import Criteria, ResponseType

DEFAULTS_PATH = Path(__file__).resolve().parents[2] / "envs" / "criteria_defaults.yaml"


class CriteriaDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int
    defaults: dict[str, Criteria]

    def for_response(self, response_type: str) -> dict[str, float]:
        c = self.defaults.get(response_type)
        return c.model_dump(exclude_none=True) if c else {}


@lru_cache
def load(path: Path = DEFAULTS_PATH) -> CriteriaDefaults:
    d = CriteriaDefaults.model_validate(yaml.safe_load(path.read_text()))
    if unknown := sorted(set(d.defaults) - set(get_args(ResponseType))):
        raise ValueError(f"{path}: unknown response types {unknown}")
    return d
