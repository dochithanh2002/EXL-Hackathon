import hashlib
import json
from pathlib import Path

from pydantic import BaseModel


def canonical_hash(value: BaseModel | dict | list | str) -> str:
    """SHA-256 of the canonical JSON form (sorted keys, no whitespace)."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
