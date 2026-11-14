"""Steps run as jobs: open the job, run the step under a tracked MLflow run, then apply its result event, or fail or
cancel the job. One job per project at a time (enforced by the store). `advance` starts whatever job a project's
state calls for, on a background thread; each finished job advances the project again."""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Callable

from .agentservice import AgentService, AgentServiceError
from .analyst import MAX_INVALID, Analyst, ToolError, tool_schema
from .baseline import BaselineFailed, run_baseline
from .endpoints import Endpoints
from .environments import Environments
from .runner import StepCancelled
from .schemas import ModelSpec, canonical_hash
from .settings import Settings
from .support import SupportMatrix
from .tracking import Tracker
from .workflow import Status, Step, Store
from .workflow.machine import AnalystReplied, BaselineSucceeded, StepFailed


class JobStream:
    """Live events of one running job, for the UI. Readers keep their own position."""

    def __init__(self):
        self.events: list[dict] = []
        self.closed = False
        self._cond = threading.Condition()

    def put(self, event: dict) -> None:
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def read(self, start: int, timeout: float) -> tuple[list[dict], bool]:
        with self._cond:
            self._cond.wait_for(lambda: len(self.events) > start or self.closed, timeout)
            return self.events[start:], self.closed and len(self.events) <= start


@dataclass
class AnalystTurn:
    jid: str
    analyst: Analyst
    spec: ModelSpec | None = None
    invalid: int = 0
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)


class Jobs:
    def __init__(self, store: Store, settings: Settings, envs: Environments, tracker: Tracker, endpoints: Endpoints,
                 agents: AgentService, matrix: SupportMatrix):
        self.store, self.settings, self.envs, self.tracker = store, settings, envs, tracker
        self.endpoints, self.agents, self.matrix = endpoints, agents, matrix
        self._cancellers: dict[str, Callable[[], None]] = {}
        self._cancel_requested: set[str] = set()
        self._streams: dict[str, JobStream] = {}
        self._turns: dict[str, AnalystTurn] = {}           # run token -> turn
        self._lock = threading.Lock()

    # ------------------------------------------------------------ dispatch
    def advance(self, pid: str) -> threading.Thread | None:
        s = self.store.state(pid)
        step = {Step.ALIGNMENT: self.analyst, Step.BASELINE: self.baseline}.get(s.step)
        if s.status != Status.WORKING or step is None:
            return None
        t = threading.Thread(target=self._run, args=(step, pid), daemon=True, name=f"job-{pid}")
        t.start()
        return t

    def _run(self, step: Callable[[str], str], pid: str) -> None:
        try:
            step(pid)
        except (sqlite3.IntegrityError, ValueError):
            return                                       # another job for the project started first
        self.advance(pid)

    def cancel(self, jid: str) -> None:
        """Stop the job's process group or agent run; the job thread then closes the job as cancelled."""
        with self._lock:
            self._cancel_requested.add(jid)
            cancel = self._cancellers.get(jid)
        if cancel:
            cancel()

    def stream(self, jid: str) -> JobStream | None:
        return self._streams.get(jid)

    def _started(self, jid: str, cancel: Callable[[], None]) -> None:
        with self._lock:
            self._cancellers[jid] = cancel
            requested = jid in self._cancel_requested
        if requested:
            cancel()

    def _open(self, pid: str, kind: str) -> tuple[str, JobStream]:
        jid = self.store.start_job(pid, kind)
        stream = self._streams[jid] = JobStream()
        stream.put({"type": "job", "job_id": jid, "kind": kind})
        return jid, stream

    def _close(self, jid: str) -> None:
        with self._lock:
            self._cancellers.pop(jid, None)
            self._cancel_requested.discard(jid)
        stream = self._streams.pop(jid, None)
        if stream:
            stream.close()

    # ------------------------------------------------------------ step 2: Analyst
    def analyst(self, pid: str) -> str:
        s = self.store.state(pid)
        if (s.step, s.status) != (Step.ALIGNMENT, Status.WORKING):
            raise ValueError(f"project is at step {s.step.value} ({s.status.value}), not waiting for the Analyst")
        jid, stream = self._open(pid, "analyst")
        token = secrets.token_urlsafe(32)
        try:
            with self.tracker.step(pid, "analyst", "analyst") as run:
                model_ref, model_label = self.endpoints.for_role("analyst")
                run.params(job_id=jid, model=model_label)
                turn = AnalystTurn(jid, Analyst(s.files, self.settings.project_dir(pid) / "inputs",
                                                self.settings.project_dir(pid) / "cache", self.matrix, self.envs,
                                                self.settings.limits))
                history = self.store.transcript(pid, "analyst")
                prompt = self._analyst_prompt(pid, turn.analyst, history)
                with self._lock:
                    self._turns[token] = turn

                def on_event(e: dict) -> None:
                    if e["type"] == "started":
                        self._started(jid, lambda: self.agents.abort(jid))
                    else:
                        stream.put(e)

                result = self.agents.run({
                    "run_id": jid, "role": "analyst", "model": model_ref, "history": history, "prompt": prompt,
                    "submit_schema": tool_schema(ModelSpec),
                    "callback": {"url": f"{self.settings.url}/internal/tools", "token": token},
                }, on_event)
                if result["type"] == "aborted" or jid in self._cancel_requested:
                    raise StepCancelled()
                if result["type"] == "error":
                    raise AgentServiceError(f"the Analyst's model request failed: {result['message']}")
                run.metrics(tool_calls=turn.tool_calls, invalid_submissions=turn.invalid)
                run.artifact_text(result["text"], "reply.md")
                if turn.spec is not None:
                    run.artifact_text(turn.spec.model_dump_json(indent=2), "spec.json")

            def commit(db):
                self.store.save_transcript(db, pid, "analyst", result["messages"])
                event = self.store.draft_spec(db, pid, turn.spec, self.matrix) if turn.spec else AnalystReplied()
                self.store.add_chat(db, pid, "analyst", result["text"], jid, getattr(event, "version", None))
                return event
            self.store.finish_job(jid, commit)
        except StepCancelled:
            self.store.cancel_job(jid)
        except Exception as e:
            self.store.fail_job(jid, str(e))
        finally:
            with self._lock:
                self._turns.pop(token, None)
            self._close(jid)
        return jid

    def _analyst_prompt(self, pid: str, analyst: Analyst, history: list) -> str:
        """The DS messages and platform notes the Analyst has not seen yet, after the project summary on the first
        turn."""
        chat = self.store.chat(pid)
        answered = max((c["seq"] for c in chat if c["author"] == "analyst"), default=0)
        parts = [] if history else [analyst.kickoff()]
        parts += [c["text"] for c in chat if c["author"] != "analyst" and c["seq"] > answered]
        return "\n\n".join(parts) or "Continue."

    def tool(self, token: str, name: str, args: dict) -> dict:
        """A tool call from an agent run, authenticated by the run's token."""
        with self._lock:
            turn = self._turns.get(token)
        if turn is None:
            raise PermissionError("unknown or expired run token")
        turn.tool_calls += 1
        try:
            if name == "read_file":
                return {"text": turn.analyst.read_file(str(args.get("name", "")))}
            if name == "dataset_profile":
                return {"text": json.dumps(turn.analyst.profile(), indent=1)}
            if name == "submit_model_spec":
                return {"text": self._submit(turn, args)}
            raise ToolError(f"unknown tool {name}")
        except ToolError as e:
            return {"text": str(e), "is_error": True}

    @staticmethod
    def _submit(turn: AnalystTurn, args: dict) -> str:
        if turn.invalid >= MAX_INVALID:
            raise ToolError(f"{MAX_INVALID} submissions failed validation; stop and tell the DS what blocks you")
        spec, errors = turn.analyst.check(args)
        if errors:
            turn.invalid += 1
            raise ToolError("The spec was not accepted:\n- " + "\n- ".join(errors)
                            + f"\n{MAX_INVALID - turn.invalid} submissions left in this turn.")
        turn.spec = spec
        return "Accepted. It becomes the next draft spec when you end your turn; now reply to the DS."

    # ------------------------------------------------------------ step 3: R baseline
    def baseline(self, pid: str) -> str:
        s = self.store.state(pid)
        if (s.step, s.status) != (Step.BASELINE, Status.WORKING):
            raise ValueError(f"project is at step {s.step.value} ({s.status.value}), not running the baseline")
        spec = self.store.spec(pid, s.spec_version)
        jid, _ = self._open(pid, "r_baseline")
        workdir = self.settings.project_dir(pid) / "runs" / jid
        try:
            with self.tracker.step(pid, "baseline", f"baseline v{s.spec_version}", workdir / "step.log") as run:
                run.params(job_id=jid, spec_version=s.spec_version, spec_hash=canonical_hash(spec),
                           baseline_key=canonical_hash(spec.baseline_key()), environment_version=self.envs.version())
                b = run_baseline(workdir, spec, s.files, self.settings.project_dir(pid) / "inputs", self.envs,
                                 self.settings.limits, lambda p: self._started(jid, p.cancel))
                run.params(baseline_hash=b.hash, r_version=b.meta["r_version"])
                run.metrics(runtime_s=b.runtime_s, **{k: v for k, v in b.fit.items()
                                                      if isinstance(v, (int, float)) and not isinstance(v, bool)})
                run.artifact(workdir / "baseline.R", "code")
                run.artifact(b.dir, "capture")
                run.artifact(workdir / "step.log")
                for lockfile in self.envs.lockfiles.values():
                    run.artifact(lockfile, "environment")
            eid, _ = self.store.record(pid, "baseline", {
                "spec_version": s.spec_version, "job_id": jid, "hash": b.hash, "files": b.files,
                "environment_version": b.environment_version, "dir": str(b.dir.relative_to(self.settings.data_dir)),
                "fit": b.fit, "meta": b.meta, "mlflow_run_id": run.run_id})
            self.store.finish_job(jid, BaselineSucceeded(spec_version=s.spec_version, baseline_id=eid))
        except StepCancelled:
            self.store.cancel_job(jid)
        except BaselineFailed as e:
            detail = str(e)
            note = (f"The R baseline for spec v{s.spec_version} failed. {e.reason}\n"
                    "Write to the Analyst to draft a corrected spec version, or retry the baseline.")

            def failed(db):
                self.store.add_chat(db, pid, "platform", note, jid, s.spec_version)
                return StepFailed(step=Step.BASELINE, spec_version=s.spec_version, detail=detail)
            self.store.finish_job(jid, failed, "failed", detail)
        except Exception as e:
            self.store.fail_job(jid, str(e))
        finally:
            self._close(jid)
        return jid
