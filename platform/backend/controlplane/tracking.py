"""MLflow as a mirror of the recorded work, never on a job's path.

A job describes each run it would log as a `Run` (params, metrics, curves, tags, dataset inputs, texts and local
artifact paths) and queues it in the store's `tracking` table once its evidence is recorded, or once it failed or
was cancelled. The `Mirror` thread replays queued rows into MLflow: one experiment per project; every step run
nested under the project's workflow parent run, and the runs a step adds (model fits, cross-validation folds,
challengers) under that step's run. A row's parent and lineage tags name other rows; a row waits until those are
mirrored. A failure is recorded on the row (failed, attempts, last_error) and retried only on request
(`Store.retry_tracking`).

Retrying a row that created its run before failing deletes that half-made run (ended FAILED, tagged
`mirror_superseded`) and logs a new one, so MLflow never shows two copies of one step. Files larger than
`Settings.mlflow_max_file_mb` are not uploaded; the run's `skipped_artifacts` tag lists them, and the local files
remain the record."""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")        # rows are retried by the mirror, not per request

from mlflow.entities import Dataset, DatasetInput, InputTag, Metric, Param, RunTag  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME  # noqa: E402

from .runner import StepCancelled  # noqa: E402

log = logging.getLogger(__name__)
PARAM_LIMIT = 6000
TAG_LIMIT = 5000

Ref = int | str          # a tracking row id, or (evidence recorded before the mirror) an MLflow run id


class Run:
    """What one MLflow run will hold. Building it touches nothing outside this object."""

    def __init__(self, kind: str, name: str, parent: Ref | None = None):
        self.kind, self.name, self.parent = kind, name, parent
        self.status = "FINISHED"
        self.data: dict = {"params": {}, "metrics": {}, "series": [], "tags": {}, "links": {}, "datasets": [],
                           "texts": {}, "artifacts": []}

    def params(self, **values) -> None:
        self.data["params"] |= {k: str(v)[:PARAM_LIMIT] for k, v in values.items()}

    def metrics(self, **values: float) -> None:
        self.data["metrics"] |= {k: float(v) for k, v in values.items()}

    def series(self, key: str, values: list[float], steps: list[int]) -> None:
        """One metric over meaningful steps (iterations, folds), so MLflow draws it as a curve."""
        self.data["series"].append([key, [float(v) for v in values], [int(s) for s in steps]])

    def datasets(self, items: list[tuple[str, str, str, str]]) -> None:
        """(name, digest, source, context) per data split, as MLflow dataset inputs."""
        self.data["datasets"] += [list(i) for i in items]

    def tags(self, **values) -> None:
        self.data["tags"] |= {k: str(v)[:TAG_LIMIT] for k, v in values.items()}

    def links(self, **refs: Ref | None) -> None:
        """Tags whose values are the MLflow run ids of other rows (lineage), resolved when mirrored."""
        self.data["links"] |= {k: v for k, v in refs.items() if v is not None}

    def artifact_text(self, text: str, name: str) -> None:
        self.data["texts"][name] = text

    def artifact(self, path: Path, folder: str | None = None) -> None:
        self.data["artifacts"].append([str(path), folder])

    def failed(self, e: BaseException, log_path: Path | None = None) -> None:
        self.status = "KILLED" if isinstance(e, StepCancelled) else "FAILED"
        self.tags(error=str(e) or type(e).__name__)
        self.artifact_text(traceback.format_exc(), "error.txt")
        if log_path is not None and log_path.is_file():
            self.artifact(log_path)

    def payload(self, data_dir: Path, start: str | None = None) -> dict:
        """The row's payload; artifact paths under `data_dir` are stored relative to it."""
        def rel(p: str) -> str:
            return str(Path(p).relative_to(data_dir)) if Path(p).is_relative_to(data_dir) else p
        return {"name": self.name, "parent": self.parent, "status": self.status, "start": _ms(start),
                "end": int(time.time() * 1000),
                **self.data, "artifacts": [[rel(p), f] for p, f in self.data["artifacts"]]}


def _ms(iso: str | None) -> int | None:
    return int(datetime.fromisoformat(iso).timestamp() * 1000) if iso else None


class Tracker:
    def __init__(self, uri: str):
        self.uri = uri
        self.client = MlflowClient(uri)

    def experiment(self, pid: str) -> str:
        name = f"project-{pid}"
        e = self.client.get_experiment_by_name(name)
        if e:
            return e.experiment_id
        try:
            return self.client.create_experiment(name, tags={"project_id": pid})
        except Exception:
            if e := self.client.get_experiment_by_name(name):             # created meanwhile
                return e.experiment_id
            raise

    def workflow_run(self, pid: str) -> str:
        eid = self.experiment(pid)
        runs = self.client.search_runs([eid], "tags.kind = 'workflow'", max_results=1)
        if runs:
            return runs[0].info.run_id
        return self.client.create_run(eid, tags={"kind": "workflow", MLFLOW_RUN_NAME: "workflow"}).info.run_id


class Waiting(Exception):
    """A referenced row is not mirrored yet."""


class Mirror:
    """Drains the tracking table into MLflow on a background thread. Nothing it does reaches a job."""

    def __init__(self, store, tracker: Tracker, data_dir: Path, max_file_mb: float, poll_s: float = 30):
        self.store, self.tracker, self.data_dir = store, tracker, data_dir
        self.max_bytes = max_file_mb * 1024 * 1024
        self.poll_s = poll_s
        self._wake, self._stop = threading.Event(), threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="tracking-mirror")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(10)

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                self.sync()
            except Exception:
                log.exception("the tracking mirror pass failed")
            self._wake.wait(self.poll_s)

    def sync(self) -> int:
        """Mirror every pending row whose references are mirrored, oldest first, until none is ready. Returns how
        many rows were attempted."""
        with self._lock:
            attempted = 0
            progress = True
            while progress and not self._stop.is_set():
                progress = False
                for row in self.store.pending_tracking():
                    try:
                        refs = self._resolve(row)
                    except Waiting:
                        continue
                    attempted += 1
                    progress = True
                    self._attempt(row, refs)
            return attempted

    # ------------------------------------------------------------ one row
    def _resolve(self, row: dict) -> dict[Ref, str]:
        p = row["payload"]
        refs = [p.get("parent"), *p.get("links", {}).values(), *p.get("targets", [])]
        rows = [r for r in refs if isinstance(r, int)]
        done = self.store.tracking_run_ids(rows)
        if any(r not in done for r in rows):
            raise Waiting()
        return {r: done.get(r, r) for r in refs if r is not None}

    def _attempt(self, row: dict, refs: dict[Ref, str]) -> None:
        self.store.tracking_attempt(row["id"])
        try:
            if row["kind"] == "decision":
                p = row["payload"]
                for target in p["targets"]:
                    self.tracker.client.log_batch(refs[target], tags=[RunTag(k, v) for k, v in p["tags"].items()])
                self.store.tracking_done(row["id"], None)
            else:
                self.store.tracking_done(row["id"], self._log_run(row, refs))
        except Exception as e:
            log.warning("could not mirror tracking row %s (%s) to MLflow: %s", row["id"], row["kind"], e)
            self.store.tracking_failed(row["id"], f"{type(e).__name__}: {e}"[:TAG_LIMIT])

    def _log_run(self, row: dict, refs: dict[Ref, str]) -> str:
        c, p, pid = self.tracker.client, row["payload"], row["project_id"]
        if row["mlflow_run_id"]:
            self._supersede(row["mlflow_run_id"])
        parent = refs[p["parent"]] if p.get("parent") is not None else self.tracker.workflow_run(pid)
        run_id = c.create_run(self.tracker.experiment(pid), start_time=p.get("start"), tags={
            MLFLOW_PARENT_RUN_ID: parent, MLFLOW_RUN_NAME: p["name"], "kind": row["kind"]}).info.run_id
        self.store.tracking_run(row["id"], run_id)
        params = [Param(k, v) for k, v in p["params"].items()]
        for i in range(0, len(params), 100):                          # MLflow's batch limit for parameters
            c.log_batch(run_id, params=params[i:i + 100])
        metrics = [Metric(k, v, 0, 0) for k, v in p["metrics"].items()]
        metrics += [Metric(key, v, 0, st) for key, values, steps in p["series"] for v, st in zip(values, steps)]
        for i in range(0, len(metrics), 1000):
            c.log_batch(run_id, metrics=metrics[i:i + 1000])
        if p["datasets"]:
            c.log_inputs(run_id, [DatasetInput(Dataset(name, digest, "local", source),
                                               [InputTag("mlflow.data.context", context)])
                                  for name, digest, source, context in p["datasets"]])
        for name, text in p["texts"].items():
            c.log_text(run_id, text, name)
        skipped = []
        for path, folder in p["artifacts"]:
            skipped += self._upload(run_id, self.data_dir / path, folder)
        tags = p["tags"] | {k: refs[v] for k, v in p["links"].items()}
        if skipped:
            tags["skipped_artifacts"] = "; ".join(skipped)[:TAG_LIMIT]
        if tags:
            c.log_batch(run_id, tags=[RunTag(k, v) for k, v in tags.items()])
        c.set_terminated(run_id, p["status"], end_time=p.get("end"))
        return run_id

    def _upload(self, run_id: str, path: Path, folder: str | None) -> list[str]:
        """Log a file or a directory, leaving out files above the size limit. Returns what was left out."""
        c = self.tracker.client
        files = sorted(f for f in path.rglob("*") if f.is_file()) if path.is_dir() else [path]
        big = {f for f in files if f.stat().st_size > self.max_bytes}
        if path.is_dir() and not big:
            c.log_artifacts(run_id, str(path), folder)
            return []
        out = []
        for f in files:
            sub = f.parent.relative_to(path).as_posix() if path.is_dir() else "."
            dest = "/".join(x for x in (folder, sub if sub != "." else None) if x) or None
            if f in big:
                out.append(f"{dest + '/' if dest else ''}{f.name} ({f.stat().st_size / 2 ** 20:.1f} MB)")
            else:
                c.log_artifact(run_id, str(f), dest)
        return out

    def _supersede(self, run_id: str) -> None:
        """End and delete the run an earlier attempt left half-made."""
        c = self.tracker.client
        try:
            if c.get_run(run_id).info.lifecycle_stage == "deleted":
                return
        except Exception as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                return
            raise
        c.set_tag(run_id, "mirror_superseded", "an incomplete mirror, replaced by a retry")
        c.set_terminated(run_id, "FAILED")
        c.delete_run(run_id)
