"""Steps run as jobs: open the job, run the step under a tracked MLflow run, then apply its result event, or fail or
cancel the job. One job per project at a time (enforced by the store). `advance` starts whatever job a project's
state calls for, on a background thread; each finished job advances the project again."""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel

from .agentservice import AgentService, AgentServiceError
from .analyst import MAX_INVALID, Analyst, ToolError, tool_schema
from .baseline import BaselineFailed, run_baseline
from .endpoints import Endpoints
from .environments import Environments
from .runner import StepCancelled
from .deliverables import build_bundle
from .diagnosis import Diagnostician, failed_checks
from .report import Reporter, gather
from .schemas import Candidate, Diagnosis, ModelSpec, ReportNarrative, canonical_hash, sha256_file
from .settings import Settings
from .support import SupportMatrix
from .tracking import StepRun, Tracker
from .translation import Translator, rulebook, run_candidate
from .workflow import Status, Step, Store
from .workflow.machine import (AnalystReplied, BaselineSucceeded, Bindings, CandidateValidated, DeliverablesBuilt,
                               DiagnosisSubmitted, ReportBuilt, StepFailed)

log = logging.getLogger(__name__)


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
class Turn:
    """One agent run: its read-only tools, and the check and acknowledgement of its one submit tool."""
    submit: str
    check: Callable[[dict], tuple[BaseModel | None, list[str]]]
    accepted: str
    tools: dict[str, Callable[[dict], str]] = field(default_factory=dict)
    result: BaseModel | None = None
    invalid: int = 0
    tool_calls: int = 0


class Jobs:
    def __init__(self, store: Store, settings: Settings, envs: Environments, tracker: Tracker, endpoints: Endpoints,
                 agents: AgentService, matrix: SupportMatrix):
        self.store, self.settings, self.envs, self.tracker = store, settings, envs, tracker
        self.endpoints, self.agents, self.matrix = endpoints, agents, matrix
        self._cancellers: dict[str, Callable[[], None]] = {}
        self._cancel_requested: set[str] = set()
        self._streams: dict[str, JobStream] = {}
        self._turns: dict[str, Turn] = {}                  # run token -> turn
        self._lock = threading.Lock()

    # ------------------------------------------------------------ dispatch
    def advance(self, pid: str) -> threading.Thread | None:
        s = self.store.state(pid)
        step = {Step.ALIGNMENT: self.analyst, Step.BASELINE: self.baseline, Step.TRANSLATION: self.translation,
                Step.REPAIR: self.diagnosis, Step.REPORT: self.report,
                Step.DELIVERABLES: self.deliverables}.get(s.step)
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

    # ------------------------------------------------------------ agents
    def _agent(self, jid: str, role: str, model_ref: dict, history: list, prompt: str, schema: type[BaseModel],
               turn: Turn, stream: JobStream) -> dict:
        """Run one role in the agent service with the turn's tools. Returns the `done` event (text, messages)."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._turns[token] = turn

        def on_event(e: dict) -> None:
            if e["type"] == "started":
                self._started(jid, lambda: self.agents.abort(jid))
            else:
                stream.put(e)
        try:
            result = self.agents.run({
                "run_id": jid, "role": role, "model": model_ref, "history": history, "prompt": prompt,
                "submit_schema": tool_schema(schema),
                "callback": {"url": f"{self.settings.url}/internal/tools", "token": token},
            }, on_event)
        finally:
            with self._lock:
                self._turns.pop(token, None)
        if result["type"] == "aborted" or jid in self._cancel_requested:
            raise StepCancelled()
        if result["type"] == "error":
            raise AgentServiceError(f"the {role}'s model request failed: {result['message']}")
        return result

    def tool(self, token: str, name: str, args: dict) -> dict:
        """A tool call from an agent run, authenticated by the run's token."""
        with self._lock:
            turn = self._turns.get(token)
        if turn is None:
            raise PermissionError("unknown or expired run token")
        turn.tool_calls += 1
        try:
            if name == turn.submit:
                return {"text": self._submit(turn, args)}
            if name in turn.tools:
                return {"text": turn.tools[name](args)}
            raise ToolError(f"unknown tool {name}")
        except ToolError as e:
            return {"text": str(e), "is_error": True}

    @staticmethod
    def _submit(turn: Turn, args: dict) -> str:
        if turn.invalid >= MAX_INVALID:
            raise ToolError(f"{MAX_INVALID} submissions failed validation; stop and say what blocks you")
        result, errors = turn.check(args)
        if errors:
            turn.invalid += 1
            raise ToolError("The submission was not accepted:\n- " + "\n- ".join(errors)
                            + f"\n{MAX_INVALID - turn.invalid} submissions left in this turn.")
        turn.result = result
        return turn.accepted

    # ------------------------------------------------------------ step 2: Analyst
    def analyst(self, pid: str) -> str:
        s = self.store.state(pid)
        if (s.step, s.status) != (Step.ALIGNMENT, Status.WORKING):
            raise ValueError(f"project is at step {s.step.value} ({s.status.value}), not waiting for the Analyst")
        jid, stream = self._open(pid, "analyst")
        try:
            with self.tracker.step(pid, "analyst", "analyst") as run:
                model_ref, model_label = self.endpoints.for_role("analyst")
                run.params(job_id=jid, model=model_label)
                analyst = self._analyst_inputs(pid, s.files)
                turn = Turn("submit_model_spec", analyst.check,
                            "Accepted. It becomes the next draft spec when you end your turn; now reply to the DS.",
                            {"read_file": lambda a: analyst.read_file(str(a.get("name", ""))),
                             "dataset_profile": lambda a: json.dumps(analyst.profile(a.get("name")), indent=1)})
                history = self.store.transcript(pid, "analyst")
                result = self._agent(jid, "analyst", model_ref, history, self._analyst_prompt(pid, analyst, history),
                                     ModelSpec, turn, stream)
                run.metrics(tool_calls=turn.tool_calls, invalid_submissions=turn.invalid)
                run.artifact_text(result["text"], "reply.md")
                spec = turn.result
                if spec is not None:
                    run.artifact_text(spec.model_dump_json(indent=2), "spec.json")

            def commit(db):
                self.store.save_transcript(db, pid, "analyst", result["messages"])
                event = self.store.draft_spec(db, pid, spec, self.matrix) if spec else AnalystReplied()
                self.store.add_chat(db, pid, "analyst", result["text"], jid, getattr(event, "version", None))
                return event
            self.store.finish_job(jid, commit)
        except StepCancelled:
            self.store.cancel_job(jid)
        except Exception as e:
            self.store.fail_job(jid, str(e))
        finally:
            self._close(jid)
        return jid

    def _analyst_inputs(self, pid: str, files) -> Analyst:
        d = self.settings.project_dir(pid)
        return Analyst(files, d / "inputs", d / "cache", self.matrix, self.envs, self.settings.limits)

    def _analyst_prompt(self, pid: str, analyst: Analyst, history: list) -> str:
        """The DS messages and platform notes the Analyst has not seen yet, after the project summary on the first
        turn."""
        chat = self.store.chat(pid)
        answered = max((c["seq"] for c in chat if c["author"] == "analyst"), default=0)
        parts = [] if history else [analyst.kickoff()]
        parts += [c["text"] for c in chat if c["author"] != "analyst" and c["seq"] > answered]
        return "\n\n".join(parts) or "Continue."

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

    # ------------------------------------------------------------ step 4: translation and validation
    def translation(self, pid: str) -> str:
        """The Translator writes a candidate; the candidate runs against R's sets and is validated against the
        baseline. Each is its own MLflow run; together they are one job."""
        s = self.store.state(pid)
        if (s.step, s.status) != (Step.TRANSLATION, Status.WORKING):
            raise ValueError(f"project is at step {s.step.value} ({s.status.value}), not translating")
        spec = self.store.spec(pid, s.spec_version)
        baseline = self.store.evidence(pid, s.baseline_id)["body"]
        baseline_dir = self.settings.data_dir / baseline["dir"]
        jid, stream = self._open(pid, "translation")
        workdir = self.settings.project_dir(pid) / "runs" / jid
        common = dict(job_id=jid, spec_version=s.spec_version, baseline_id=s.baseline_id)
        attempt = dict(attempt=s.repairs, diagnosis_id=s.diagnosis_id)
        try:
            with self.tracker.step(pid, "translation", f"translation v{s.spec_version}") as run:
                model_ref, model_label = self.endpoints.for_role("translator")
                run.params(model=model_label, **common, **attempt)
                analyst = self._analyst_inputs(pid, s.files)
                dataset = next(f.name for f in s.files if f.role == "dataset")
                translator = Translator(spec, baseline_dir, dataset, self.envs, analyst.profile,
                                        [f.name for f in s.files if f.role == "data"],
                                        self._repair(pid, s.diagnosis_id, s.repairs, s.retry_limit))
                turn = Turn("submit_candidate", translator.check,
                            "Accepted. It runs against R's sets when you end your turn; now summarise it in a few "
                            "sentences.",
                            {"dataset_profile": lambda a: json.dumps(translator.profile(a.get("name")), indent=1),
                             "read_rulebook": lambda a: rulebook()})
                result = self._agent(jid, "translator", model_ref, [], translator.kickoff(), Candidate, turn, stream)
                run.metrics(tool_calls=turn.tool_calls, invalid_submissions=turn.invalid)
                run.artifact_text(result["text"], "reply.md")
                run.artifact_text(json.dumps(result["messages"], indent=1), "transcript.json")
                cand: Candidate | None = turn.result
                if cand is None:
                    raise AgentServiceError("the Translator ended its turn without an accepted candidate: "
                                            + (result["text"].strip()[:500] or "(no reply)"))
                h = hashlib.sha256(cand.module.encode()).hexdigest()
                run.params(candidate_hash=h)
                run.artifact_text(cand.module, "candidate.py")
                cid, _ = self.store.record(pid, "candidate", {
                    **common, **attempt, "hash": h, "module": cand.module, "notes": cand.notes,
                    "limitations": cand.limitations, "changes": cand.changes,
                    "reply": result["text"], "model": model_label, "mlflow_run_id": run.run_id})

            with self.tracker.step(pid, "validation", f"validation v{s.spec_version}", workdir / "step.log") as run:
                run.params(candidate_id=cid, candidate_hash=h, environment_version=self.envs.version(), **common)
                v = run_candidate(workdir, cand.module, spec, baseline_dir, s.files,
                                  self.settings.project_dir(pid) / "inputs", self.envs, self.settings.limits,
                                  lambda p: self._started(jid, p.cancel))
                body = json.loads(v.model_dump_json())
                run.metrics(**v.metrics_for_mlflow())
                run.tags(passed=v.passed, stage=v.stage)
                run.artifact_text(json.dumps(body, indent=1), "validation.json")
                run.artifact(workdir / "candidate.py", "code")
                if (workdir / "output").is_dir():
                    run.artifact(workdir / "output", "output")
                run.artifact(workdir / "step.log")
                vid, _ = self.store.record(pid, "validation", {
                    **common, "candidate_id": cid, "candidate_hash": h, "environment_version": self.envs.version(),
                    **body, "mlflow_run_id": run.run_id})
            self.store.finish_job(jid, CandidateValidated(
                bindings=Bindings(spec_version=s.spec_version, baseline_id=s.baseline_id, candidate_hash=h,
                                  validation_id=vid), passed=v.passed))
        except StepCancelled:
            self.store.cancel_job(jid)
        except Exception as e:
            self.store.fail_job(jid, str(e))
        finally:
            self._close(jid)
        return jid

    def _repair(self, pid: str, diagnosis_id: str | None, attempt: int, limit: int) -> dict | None:
        """What a repair attempt's Translator gets: the previous candidate and the diagnosis of it."""
        if diagnosis_id is None:
            return None
        d = self.store.evidence(pid, diagnosis_id)["body"]
        module = self.store.evidence(pid, d["candidate_id"])["body"]["module"]
        return {"attempt": attempt, "limit": limit, "module": module, "diagnosis": d}

    # ------------------------------------------------------------ step 5: diagnosis
    def diagnosis(self, pid: str) -> str:
        """The Diagnostician explains why the candidate failed validation or was rejected and tells the Translator
        what to change, or says the fix needs a new spec version, which goes to the DS."""
        s = self.store.state(pid)
        if (s.step, s.status) != (Step.REPAIR, Status.WORKING):
            raise ValueError(f"project is at step {s.step.value} ({s.status.value}), not diagnosing")
        spec = self.store.spec(pid, s.spec_version)
        baseline_dir = self.settings.data_dir / self.store.evidence(pid, s.baseline_id)["body"]["dir"]
        validation = self.store.evidence(pid, s.validation_id)["body"]
        cand = self.store.evidence(pid, validation["candidate_id"])["body"]
        jid, stream = self._open(pid, "diagnosis")
        common = dict(job_id=jid, spec_version=s.spec_version, baseline_id=s.baseline_id,
                      candidate_id=validation["candidate_id"], candidate_hash=s.candidate_hash,
                      validation_id=s.validation_id)
        try:
            with self.tracker.step(pid, "diagnosis", f"diagnosis v{s.spec_version}") as run:
                model_ref, model_label = self.endpoints.for_role("diagnostician")
                run.params(model=model_label, repairs=s.repairs, retry_limit=s.retry_limit, **common)
                analyst = self._analyst_inputs(pid, s.files)
                diag = Diagnostician(spec, baseline_dir, analyst.columns, cand, validation, s.ds_comment,
                                     self._earlier(pid, s.spec_version, s.baseline_id), s.repairs, s.retry_limit)
                turn = Turn("submit_diagnosis", diag.check,
                            "Accepted. When you end your turn it goes to the Translator, or to the DS if it needs a "
                            "spec change; now summarise it in a few sentences.",
                            {"dataset_profile": lambda a: json.dumps(analyst.profile(a.get("name")), indent=1),
                             "read_rulebook": lambda a: rulebook()})
                result = self._agent(jid, "diagnostician", model_ref, [], diag.kickoff(), Diagnosis, turn, stream)
                run.metrics(tool_calls=turn.tool_calls, invalid_submissions=turn.invalid)
                run.artifact_text(result["text"], "reply.md")
                run.artifact_text(json.dumps(result["messages"], indent=1), "transcript.json")
                d: Diagnosis | None = turn.result
                if d is None:
                    raise AgentServiceError("the Diagnostician ended its turn without an accepted diagnosis: "
                                            + (result["text"].strip()[:500] or "(no reply)"))
                run.tags(spec_change_needed=d.spec_change_needed)
                run.artifact_text(d.model_dump_json(indent=2), "diagnosis.json")
                did, _ = self.store.record(pid, "diagnosis", {
                    **common, **d.model_dump(mode="json"), "ds_comment": s.ds_comment,
                    "failed_checks": failed_checks(validation), "reply": result["text"], "model": model_label,
                    "mlflow_run_id": run.run_id})
            event = DiagnosisSubmitted(spec_version=s.spec_version, candidate_hash=s.candidate_hash, diagnosis_id=did,
                                       spec_change_needed=d.spec_change_needed)

            def commit(db):
                if d.spec_change_needed:
                    finding = {"specification": "the spec needs to change before the migration can match R",
                               "not_reproducible": "no faithful Python implementation can reproduce R",
                               "model_fitness": "the migration is faithful; the concern is the model itself"}[d.category]
                    self.store.add_chat(db, pid, "platform", (
                        f"The Diagnostician found that {finding} (spec v{s.spec_version}).\nRoot cause: "
                        f"{d.root_cause}\n" + (f"Limitation: {d.limitation}\n" if d.limitation else "")
                        + f"Proposed change: {d.proposed_spec_change}\nWrite to the Analyst to draft a new spec "
                        "version, or resume the repair to keep the current spec."), jid, s.spec_version)
                return event
            self.store.finish_job(jid, commit)
        except StepCancelled:
            self.store.cancel_job(jid)
        except Exception as e:
            self.store.fail_job(jid, str(e))
        finally:
            self._close(jid)
        return jid

    def _earlier(self, pid: str, spec_version: int, baseline_id: str) -> list[dict]:
        """The diagnoses already made for this spec and baseline, oldest first."""
        bodies = (self.store.evidence(pid, e["id"])["body"] for e in self.store.evidence_list(pid, "diagnosis"))
        return [d for d in bodies if (d["spec_version"], d["baseline_id"]) == (spec_version, baseline_id)]

    # ------------------------------------------------------------ step 6: report
    def report(self, pid: str) -> str:
        """The Reporter writes the narrative of the approved translation from the evidence catalog; code checks its
        citations, fills in its numbers and adds the evidence tables."""
        s = self.store.state(pid)
        if (s.step, s.status) != (Step.REPORT, Status.WORKING):
            raise ValueError(f"project is at step {s.step.value} ({s.status.value}), not reporting")
        jid, stream = self._open(pid, "report")
        common = dict(job_id=jid, **s.approval.model_dump())
        try:
            with self.tracker.step(pid, "report", f"report v{s.spec_version}") as run:
                model_ref, model_label = self.endpoints.for_role("reporter")
                run.params(model=model_label, **common)
                reporter = Reporter(gather(self.store, self.settings, pid))
                turn = Turn("submit_report", reporter.check,
                            "Accepted. The platform fills in the numbers and adds the evidence tables when you end "
                            "your turn; now end it with one sentence.")
                result = self._agent(jid, "reporter", model_ref, [], reporter.kickoff(), ReportNarrative, turn,
                                     stream)
                run.metrics(tool_calls=turn.tool_calls, invalid_submissions=turn.invalid)
                run.artifact_text(json.dumps(result["messages"], indent=1), "transcript.json")
                narrative: ReportNarrative | None = turn.result
                if narrative is None:
                    raise AgentServiceError("the Reporter ended its turn without an accepted report: "
                                            + (result["text"].strip()[:500] or "(no reply)"))
                markdown = reporter.render(narrative)
                run.artifact_text(narrative.model_dump_json(indent=2), "narrative.json")
                run.artifact_text(markdown, "report.md")
                rid, _ = self.store.record(pid, "report", {
                    **common, "spec_version": s.spec_version, "narrative": narrative.model_dump(mode="json"),
                    "catalog": reporter.entries, "markdown": markdown, "model": model_label,
                    "mlflow_run_id": run.run_id})
            self.store.finish_job(jid, ReportBuilt(spec_version=s.spec_version, report_id=rid))
        except StepCancelled:
            self.store.cancel_job(jid)
        except Exception as e:
            self.store.fail_job(jid, str(e))
        finally:
            self._close(jid)
        return jid

    # ------------------------------------------------------------ step 7: deliverables
    def deliverables(self, pid: str) -> str:
        s = self.store.state(pid)
        if (s.step, s.status) != (Step.DELIVERABLES, Status.WORKING):
            raise ValueError(f"project is at step {s.step.value} ({s.status.value}), not building deliverables")
        jid, _ = self._open(pid, "deliverables")
        dest = self.settings.project_dir(pid) / "deliverables" / jid
        common = dict(job_id=jid, report_id=s.report_id, **s.approval.model_dump())
        try:
            with self.tracker.step(pid, "deliverables", f"deliverables v{s.spec_version}") as run:
                run.params(environment_version=self.envs.version(), **common)
                record = gather(self.store, self.settings, pid)
                manifest = build_bundle(dest, record, self.store.evidence(pid, s.report_id)["body"]["markdown"],
                                        self.envs)
                archive = dest.with_suffix(".zip")
                run.artifact(dest, "bundle")
                run.artifact(archive)
                bid, _ = self.store.record(pid, "bundle", {
                    **common, "spec_version": s.spec_version, "dir": str(dest.relative_to(self.settings.data_dir)),
                    "archive": str(archive.relative_to(self.settings.data_dir)), "archive_sha256": sha256_file(archive),
                    "manifest": manifest, "mlflow_run_id": run.run_id})
            self.store.finish_job(jid, DeliverablesBuilt(spec_version=s.spec_version, bundle_id=bid))
        except StepCancelled:
            self.store.cancel_job(jid)
        except Exception as e:
            self.store.fail_job(jid, str(e))
        finally:
            self._close(jid)
        return jid

    def tag_decision(self, pid: str, bindings: Bindings, decision: str, comment: str | None = None) -> None:
        """Mirror a DS approval or rejection onto the candidate's and validation's MLflow runs. The transition row
        is the record; a tracking failure is logged, not raised."""
        try:
            v = self.store.evidence(pid, bindings.validation_id)["body"]
            c = self.store.evidence(pid, v["candidate_id"])["body"]
            tags = {"decision": decision} | ({"ds_comment": comment[:5000]} if comment else {})
            for run_id in (c["mlflow_run_id"], v["mlflow_run_id"]):
                StepRun(self.tracker.client, run_id).tags(**tags)
        except Exception:
            log.exception("could not tag the %s decision for project %s in MLflow", decision, pid)
