"""Model endpoints and the model assigned to each agent role. API keys are encrypted at rest and leave this module
only inside a run request to the agent service; the API reports `has_key`, never the key."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Annotated, Literal

from cryptography.fernet import Fernet
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from .schemas import ROLES
from .workflow import NotFound

Api = Literal["openai-responses", "openai-completions", "anthropic-messages", "google-generative-ai"]
Thinking = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS endpoints (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    base_url TEXT NOT NULL,
    api TEXT NOT NULL,
    models TEXT NOT NULL,
    key BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS role_models (
    role TEXT PRIMARY KEY,
    endpoint_id TEXT NOT NULL REFERENCES endpoints(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    thinking TEXT NOT NULL
);
"""


class EndpointIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    base_url: HttpUrl
    api: Api
    models: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    api_key: str | None = Field(default=None, min_length=1, description="Omit on update to keep the stored key")


class Endpoint(BaseModel):
    id: str
    name: str
    base_url: str
    api: Api
    models: list[str]
    has_key: bool = True


class RoleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint_id: str
    model: str
    thinking: Thinking = "medium"


class NotConfigured(Exception):
    pass


class InvalidSetting(Exception):
    pass


def load_cipher(data_dir: Path) -> Fernet:
    """The key comes from CONTROLPLANE_SECRET_KEY, or from `secret.key` in the data directory, created on first
    use and readable only by its owner."""
    if key := os.environ.get("CONTROLPLANE_SECRET_KEY"):
        return Fernet(key)
    path = data_dir / "secret.key"
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(Fernet.generate_key())
    return Fernet(path.read_bytes())


class Endpoints:
    def __init__(self, path: Path | str, cipher: Fernet):
        self.path, self.cipher = str(path), cipher
        with closing(self._db()) as db:
            db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def list(self) -> list[Endpoint]:
        with closing(self._db()) as db:
            rows = db.execute("SELECT * FROM endpoints ORDER BY name").fetchall()
        return [self._endpoint(r) for r in rows]

    def get(self, eid: str) -> Endpoint:
        with closing(self._db()) as db:
            row = db.execute("SELECT * FROM endpoints WHERE id = ?", (eid,)).fetchone()
        if row is None:
            raise NotFound(f"endpoint {eid}")
        return self._endpoint(row)

    def create(self, e: EndpointIn) -> Endpoint:
        if e.api_key is None:
            raise InvalidSetting("api_key is required")
        eid = uuid.uuid4().hex[:12]
        with closing(self._db()) as db, db:
            db.execute("INSERT INTO endpoints VALUES (?, ?, ?, ?, ?, ?)",
                       (eid, e.name, str(e.base_url), e.api, json.dumps(e.models), self._seal(e.api_key)))
        return self.get(eid)

    def update(self, eid: str, e: EndpointIn) -> Endpoint:
        self.get(eid)
        with closing(self._db()) as db, db:
            db.execute("UPDATE endpoints SET name = ?, base_url = ?, api = ?, models = ? WHERE id = ?",
                       (e.name, str(e.base_url), e.api, json.dumps(e.models), eid))
            if e.api_key is not None:
                db.execute("UPDATE endpoints SET key = ? WHERE id = ?", (self._seal(e.api_key), eid))
        return self.get(eid)

    def delete(self, eid: str) -> None:
        with closing(self._db()) as db, db:
            db.execute("DELETE FROM endpoints WHERE id = ?", (eid,))

    def roles(self) -> dict[str, RoleModel]:
        with closing(self._db()) as db:
            rows = db.execute("SELECT * FROM role_models").fetchall()
        return {r["role"]: RoleModel(endpoint_id=r["endpoint_id"], model=r["model"], thinking=r["thinking"])
                for r in rows}

    def assign(self, role: str, m: RoleModel) -> None:
        if role not in ROLES:
            raise NotFound(f"role {role}")
        if m.model not in self.get(m.endpoint_id).models:
            raise InvalidSetting(f"endpoint {m.endpoint_id} does not list model {m.model}")
        with closing(self._db()) as db, db:
            db.execute("INSERT INTO role_models VALUES (?, ?, ?, ?) ON CONFLICT (role) DO UPDATE SET "
                       "endpoint_id = excluded.endpoint_id, model = excluded.model, thinking = excluded.thinking",
                       (role, m.endpoint_id, m.model, m.thinking))

    def model_ref(self, endpoint_id: str, model: str, thinking: Thinking = "off") -> dict:
        """What the agent service needs to call the model, key included. Only for requests to the agent service."""
        with closing(self._db()) as db:
            row = db.execute("SELECT * FROM endpoints WHERE id = ?", (endpoint_id,)).fetchone()
        if row is None:
            raise NotFound(f"endpoint {endpoint_id}")
        return {"api": row["api"], "base_url": row["base_url"], "model": model, "thinking": thinking,
                "api_key": self.cipher.decrypt(row["key"]).decode()}

    def for_role(self, role: str) -> tuple[dict, str]:
        """(model_ref, "endpoint name / model") for the role; raises NotConfigured if none is assigned."""
        m = self.roles().get(role)
        if m is None:
            raise NotConfigured(f"no model is assigned to the {role} role; assign one in settings")
        return self.model_ref(m.endpoint_id, m.model, m.thinking), f"{self.get(m.endpoint_id).name} / {m.model}"

    def _seal(self, key: str) -> bytes:
        return self.cipher.encrypt(key.encode())

    @staticmethod
    def _endpoint(r: sqlite3.Row) -> Endpoint:
        return Endpoint(id=r["id"], name=r["name"], base_url=r["base_url"], api=r["api"],
                        models=json.loads(r["models"]))
