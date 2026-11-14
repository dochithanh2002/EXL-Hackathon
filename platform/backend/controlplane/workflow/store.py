"""SQLite persistence. A project's state changes only through `Store.apply`, which runs the state machine and writes
the new state and a transition row in one transaction. Jobs and recorded evidence are rows too."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, TypeAdapter

from ..schemas.agents import ModelSpec
from ..schemas.hashing import canonical_hash
from ..support import SupportMatrix
from .machine import (Bindings, Cancelled, DsMessaged, Event, GateError, ProjectState, SpecDrafted, Status, Step,
                      StepFailed, apply)

EVENT = TypeAdapter(Event)

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
    project_id TEXT NOT NULL REFERENCES projects(id),
    seq INTEGER NOT NULL,
    at TEXT NOT NULL,
    event TEXT NOT NULL,
    from_step INTEGER NOT NULL,
    from_status TEXT NOT NULL,
    to_step INTEGER NOT NULL,
    to_status TEXT NOT NULL,
    PRIMARY KEY (project_id, seq)
);
CREATE TABLE IF NOT EXISTS specs (
    project_id TEXT NOT NULL REFERENCES projects(id),
    version INTEGER NOT NULL,
    hash TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (project_id, version)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    step INTEGER NOT NULL,
    kind TEXT NOT NULL,
    spec_version INTEGER,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled', 'interrupted')),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running_job ON jobs(project_id) WHERE status = 'running';
CREATE TABLE IF NOT EXISTS chat (
    project_id TEXT NOT NULL REFERENCES projects(id),
    seq INTEGER NOT NULL,
    at TEXT NOT NULL,
    author TEXT NOT NULL CHECK (author IN ('ds', 'analyst', 'platform')),
    text TEXT NOT NULL,
    job_id TEXT,
    spec_version INTEGER,
    PRIMARY KEY (project_id, seq)
);
CREATE TABLE IF NOT EXISTS transcripts (
    project_id TEXT NOT NULL REFERENCES projects(id),
    role TEXT NOT NULL,
    messages TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (project_id, role)
);
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    kind TEXT NOT NULL,
    hash TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class NotFound(Exception):
    pass


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        db = sqlite3.connect(self.path)
        try:
            db.executescript(SCHEMA)
        finally:
            db.close()

    @contextmanager
    def _tx(self):
        db = sqlite3.connect(self.path, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    # ------------------------------------------------------------ projects
    def create_project(self, name: str, retry_limit: int = 3) -> str:
        pid = uuid.uuid4().hex[:12]
        state = ProjectState(retry_limit=retry_limit)
        with self._tx() as db:
            db.execute("INSERT INTO projects VALUES (?, ?, ?, ?)", (pid, name, now(), state.model_dump_json()))
        return pid

    def projects(self) -> list[dict]:
        with self._tx() as db:
            rows = db.execute("SELECT id, name, created_at, state FROM projects ORDER BY created_at DESC").fetchall()
        return [{"id": r["id"], "name": r["name"], "created_at": r["created_at"],
                 "state": ProjectState.model_validate_json(r["state"])} for r in rows]

    def state(self, pid: str, db: sqlite3.Connection | None = None) -> ProjectState:
        if db is None:
            with self._tx() as db:
                return self.state(pid, db)
        row = db.execute("SELECT state FROM projects WHERE id = ?", (pid,)).fetchone()
        if row is None:
            raise NotFound(pid)
        return ProjectState.model_validate_json(row["state"])

    def apply(self, pid: str, event: Event, db: sqlite3.Connection | None = None) -> ProjectState:
        if db is None:
            with self._tx() as db:
                return self.apply(pid, event, db)
        before = self.state(pid, db)
        after = apply(before, event)
        seq = db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM transitions WHERE project_id = ?",
                         (pid,)).fetchone()[0]
        db.execute("INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   (pid, seq, now(), event.model_dump_json(), before.step, before.status, after.step, after.status))
        db.execute("UPDATE projects SET state = ? WHERE id = ?", (after.model_dump_json(), pid))
        return after

    def transitions(self, pid: str) -> list[dict]:
        with self._tx() as db:
            rows = db.execute("SELECT * FROM transitions WHERE project_id = ? ORDER BY seq", (pid,)).fetchall()
        return [{**dict(r), "event": json.loads(r["event"])} for r in rows]

    def decisions(self, pid: str) -> list[dict]:
        """Every DS approval and rejection, each marked void unless it is bound to the current state."""
        current = self.state(pid).bindings()
        out = []
        for t in self.transitions(pid):
            e = t["event"]
            if e["type"] in ("candidate_approved", "candidate_rejected"):
                b = Bindings.model_validate(e["bindings"])
                out.append({"at": t["at"], "decision": e["type"].removeprefix("candidate_"),
                            "comment": e.get("comment"), "bindings": b, "void": b != current})
        return out

    # ------------------------------------------------------------ specs
    def add_spec(self, pid: str, spec: ModelSpec, matrix: SupportMatrix) -> int:
        """Store the next spec version and record it as drafted, with its tier and proven flag taken from the
        support matrix (never from the Analyst's own claim)."""
        with self._tx() as db:
            event = self.draft_spec(db, pid, spec, matrix)
            self.apply(pid, event, db)
        return event.version

    @staticmethod
    def draft_spec(db: sqlite3.Connection, pid: str, spec: ModelSpec, matrix: SupportMatrix) -> SpecDrafted:
        """Insert the next spec version; the caller applies the returned event in the same transaction."""
        support = matrix.support(spec)
        v = db.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM specs WHERE project_id = ?", (pid,)).fetchone()[0]
        db.execute("INSERT INTO specs VALUES (?, ?, ?, ?, ?)",
                   (pid, v, canonical_hash(spec), spec.model_dump_json(), now()))
        return SpecDrafted(version=v, tier=support.tier, proven=support.proven)

    def specs(self, pid: str) -> list[dict]:
        with self._tx() as db:
            rows = db.execute("SELECT version, hash, created_at FROM specs WHERE project_id = ? ORDER BY version",
                              (pid,)).fetchall()
        return [dict(r) for r in rows]

    def spec(self, pid: str, version: int) -> ModelSpec:
        with self._tx() as db:
            row = db.execute("SELECT body FROM specs WHERE project_id = ? AND version = ?",
                             (pid, version)).fetchone()
        if row is None:
            raise NotFound(f"{pid} spec v{version}")
        return ModelSpec.model_validate_json(row["body"])

    # ------------------------------------------------------------ chat
    def ds_message(self, pid: str, text: str) -> None:
        """The DS writes to the Analyst: record the message and hand the Analyst the turn."""
        with self._tx() as db:
            self.apply(pid, DsMessaged(), db)
            self.add_chat(db, pid, "ds", text)

    @staticmethod
    def add_chat(db: sqlite3.Connection, pid: str, author: str, text: str, job_id: str | None = None,
                 spec_version: int | None = None) -> None:
        seq = db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM chat WHERE project_id = ?", (pid,)).fetchone()[0]
        db.execute("INSERT INTO chat VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (pid, seq, now(), author, text, job_id, spec_version))

    def chat(self, pid: str) -> list[dict]:
        with self._tx() as db:
            self.state(pid, db)
            rows = db.execute("SELECT seq, at, author, text, job_id, spec_version FROM chat WHERE project_id = ? "
                              "ORDER BY seq", (pid,)).fetchall()
        return [dict(r) for r in rows]

    def transcript(self, pid: str, role: str) -> list:
        """The role's full agent transcript (Pi messages, opaque here), replayed into each new session."""
        with self._tx() as db:
            row = db.execute("SELECT messages FROM transcripts WHERE project_id = ? AND role = ?",
                             (pid, role)).fetchone()
        return json.loads(row["messages"]) if row else []

    @staticmethod
    def save_transcript(db: sqlite3.Connection, pid: str, role: str, messages: list) -> None:
        db.execute("INSERT INTO transcripts VALUES (?, ?, ?, ?) ON CONFLICT (project_id, role) "
                   "DO UPDATE SET messages = excluded.messages, updated_at = excluded.updated_at",
                   (pid, role, json.dumps(messages), now()))

    # ------------------------------------------------------------ evidence
    def record(self, pid: str, kind: str, body: BaseModel | dict) -> tuple[str, str]:
        """Store an immutable piece of evidence (baseline, candidate, validation, ...). Returns (id, hash)."""
        data = body.model_dump(mode="json") if isinstance(body, BaseModel) else body
        eid, h = f"{kind}-{uuid.uuid4().hex[:10]}", canonical_hash(data)
        with self._tx() as db:
            db.execute("INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                       (eid, pid, kind, h, json.dumps(data, sort_keys=True), now()))
        return eid, h

    def evidence(self, pid: str, eid: str) -> dict:
        with self._tx() as db:
            row = db.execute("SELECT * FROM evidence WHERE project_id = ? AND id = ?", (pid, eid)).fetchone()
        if row is None:
            raise NotFound(eid)
        return {**dict(row), "body": json.loads(row["body"])}

    # ------------------------------------------------------------ jobs
    def start_job(self, pid: str, kind: str) -> str:
        """Open a job for the project's current step. At most one job per project runs at a time."""
        jid = uuid.uuid4().hex[:12]
        with self._tx() as db:
            s = self.state(pid, db)
            if s.status != Status.WORKING:
                raise ValueError(f"project is {s.status.value}, not working")
            db.execute("INSERT INTO jobs (id, project_id, step, kind, spec_version, status, started_at) "
                       "VALUES (?, ?, ?, ?, ?, 'running', ?)", (jid, pid, s.step, kind, s.spec_version, now()))
        return jid

    def finish_job(self, jid: str, event: Event | Callable[[sqlite3.Connection], Event] | None,
                   status: str = "succeeded", detail: str = "") -> None:
        """Close a job and apply its result event in the same transaction. `event` may be a function that writes the
        result's rows and returns the event, so those rows commit only if the event applies. A job whose result no
        longer applies (the spec changed meanwhile) is closed as failed with the gate's message."""
        with self._tx() as db:
            job = self._job(db, jid)
            if job["status"] != "running":
                return
            if event is not None:
                try:
                    db.execute("SAVEPOINT result")
                    self.apply(job["project_id"], event(db) if callable(event) else event, db)
                    db.execute("RELEASE result")
                except GateError as e:
                    db.execute("ROLLBACK TO result")
                    status, detail = "failed", f"result discarded: {e}"
            db.execute("UPDATE jobs SET status = ?, ended_at = ?, detail = ? WHERE id = ?",
                       (status, now(), detail, jid))

    def fail_job(self, jid: str, detail: str) -> None:
        job = self._row_job(jid)
        self.finish_job(jid, StepFailed(step=Step(job["step"]), spec_version=job["spec_version"], detail=detail),
                        "failed", detail)

    def cancel_job(self, jid: str) -> None:
        job = self._row_job(jid)
        self.finish_job(jid, Cancelled(step=Step(job["step"])), "cancelled")

    def running_job(self, pid: str) -> dict | None:
        with self._tx() as db:
            row = db.execute("SELECT * FROM jobs WHERE project_id = ? AND status = 'running'", (pid,)).fetchone()
        return dict(row) if row else None

    def recover(self) -> list[str]:
        """On startup: every job still marked running was cut off by a restart. Mark it interrupted and move its
        project to awaiting the DS, who can retry the step."""
        with self._tx() as db:
            jobs = db.execute("SELECT * FROM jobs WHERE status = 'running'").fetchall()
            for job in jobs:
                db.execute("UPDATE jobs SET status = 'interrupted', ended_at = ? WHERE id = ?", (now(), job["id"]))
                s = self.state(job["project_id"], db)
                if s.status == Status.WORKING and s.step == job["step"] and s.spec_version == job["spec_version"]:
                    self.apply(job["project_id"], StepFailed(step=Step(job["step"]), spec_version=job["spec_version"],
                                                             detail="interrupted by a restart", interrupted=True), db)
        return [j["id"] for j in jobs]

    def _row_job(self, jid: str) -> dict:
        with self._tx() as db:
            return self._job(db, jid)

    @staticmethod
    def _job(db: sqlite3.Connection, jid: str) -> dict:
        row = db.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
        if row is None:
            raise NotFound(jid)
        return dict(row)
