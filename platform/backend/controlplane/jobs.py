"""Static steps run as jobs: open the job, run the step under a tracked MLflow run, then apply its result event,
or fail or cancel the job. One job per project at a time (enforced by the store)."""
from __future__ import annotations

import threading

from .baseline import run_baseline
from .environments import Environments
from .runner import Process, StepCancelled
from .schemas import canonical_hash
from .settings import Settings
from .tracking import Tracker
from .workflow import Status, Step, Store
from .workflow.machine import BaselineSucceeded


class Jobs:
    def __init__(self, store: Store, settings: Settings, envs: Environments, tracker: Tracker):
        self.store, self.settings, self.envs, self.tracker = store, settings, envs, tracker
        self._procs: dict[str, Process] = {}
        self._cancel_requested: set[str] = set()
        self._lock = threading.Lock()

    def cancel(self, jid: str) -> None:
        """Kill the job's process group; the job thread then closes the job as cancelled."""
        with self._lock:
            self._cancel_requested.add(jid)
            proc = self._procs.get(jid)
        if proc:
            proc.cancel()

    def _started(self, jid: str, proc: Process) -> None:
        with self._lock:
            self._procs[jid] = proc
            if jid in self._cancel_requested:
                proc.cancel()

    def baseline(self, pid: str) -> str:
        s = self.store.state(pid)
        if (s.step, s.status) != (Step.BASELINE, Status.WORKING):
            raise ValueError(f"project is at step {s.step.value} ({s.status.value}), not running the baseline")
        spec = self.store.spec(pid, s.spec_version)
        jid = self.store.start_job(pid, "r_baseline")
        workdir = self.settings.project_dir(pid) / "runs" / jid
        try:
            with self.tracker.step(pid, "baseline", f"baseline v{s.spec_version}", workdir / "step.log") as run:
                run.params(job_id=jid, spec_version=s.spec_version, spec_hash=canonical_hash(spec),
                           baseline_key=canonical_hash(spec.baseline_key()), environment_version=self.envs.version())
                b = run_baseline(workdir, spec, s.files, self.settings.project_dir(pid) / "inputs", self.envs,
                                 self.settings.limits, lambda p: self._started(jid, p))
                run.params(baseline_hash=b.hash, r_version=b.meta["r_version"])
                run.metrics(runtime_s=b.runtime_s,
                            **{k: v for k, v in b.fit.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
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
        except Exception as e:
            self.store.fail_job(jid, str(e))
        finally:
            with self._lock:
                self._procs.pop(jid, None)
                self._cancel_requested.discard(jid)
        return jid
