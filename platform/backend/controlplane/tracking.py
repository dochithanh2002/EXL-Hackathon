"""MLflow: one experiment per project; every step run is nested under the project's workflow parent run, and the
runs a step adds (model fits, cross-validation folds, challengers) under that step's run."""
from __future__ import annotations

import os
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities import Dataset, DatasetInput, InputTag, Metric, Param, RunTag  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME  # noqa: E402

from .runner import StepCancelled  # noqa: E402

PARAM_LIMIT = 6000


class StepRun:
    def __init__(self, client: MlflowClient, run_id: str):
        self.client, self.run_id = client, run_id

    def params(self, **values) -> None:
        items = [Param(k, str(v)[:PARAM_LIMIT]) for k, v in values.items()]
        for i in range(0, len(items), 100):                          # MLflow's batch limit for parameters
            self.client.log_batch(self.run_id, params=items[i:i + 100])

    def metrics(self, **values: float) -> None:
        items = [Metric(k, float(v), 0, 0) for k, v in values.items()]
        for i in range(0, len(items), 1000):
            self.client.log_batch(self.run_id, metrics=items[i:i + 1000])

    def series(self, key: str, values: list[float], steps: list[int]) -> None:
        """One metric over meaningful steps (iterations, folds), so MLflow draws it as a curve."""
        self.client.log_batch(self.run_id, metrics=[Metric(key, float(v), 0, int(st)) for v, st in zip(values, steps)])

    def datasets(self, items: list[tuple[str, str, str, str]]) -> None:
        """(name, digest, source, context) per data split, as MLflow dataset inputs."""
        self.client.log_inputs(self.run_id, [DatasetInput(Dataset(name, digest, "local", source),
                                                          [InputTag("mlflow.data.context", context)])
                                             for name, digest, source, context in items])

    def tags(self, **values) -> None:
        self.client.log_batch(self.run_id, tags=[RunTag(k, str(v)) for k, v in values.items()])

    def artifact_text(self, text: str, name: str) -> None:
        self.client.log_text(self.run_id, text, name)

    def artifact(self, path: Path, folder: str | None = None) -> None:
        if path.is_dir():
            self.client.log_artifacts(self.run_id, str(path), folder)
        else:
            self.client.log_artifact(self.run_id, str(path), folder)


class Tracker:
    def __init__(self, uri: str):
        self.uri = uri
        self.client = MlflowClient(uri)

    def experiment(self, pid: str) -> str:
        name = f"project-{pid}"
        e = self.client.get_experiment_by_name(name)
        return e.experiment_id if e else self.client.create_experiment(name, tags={"project_id": pid})

    def workflow_run(self, pid: str) -> str:
        eid = self.experiment(pid)
        runs = self.client.search_runs([eid], "tags.kind = 'workflow'", max_results=1)
        if runs:
            return runs[0].info.run_id
        return self.client.create_run(eid, tags={"kind": "workflow", MLFLOW_RUN_NAME: "workflow"}).info.run_id

    @contextmanager
    def step(self, pid: str, kind: str, name: str, log: Path | None = None,
             parent: str | None = None) -> Iterator[StepRun]:
        """A nested run that ends FINISHED, or FAILED with the traceback and the step log as artifacts. It is nested
        under `parent` if given, else under the workflow run."""
        parent = parent or self.workflow_run(pid)
        eid = self.client.get_run(parent).info.experiment_id
        run = StepRun(self.client, self.client.create_run(
            eid, tags={MLFLOW_PARENT_RUN_ID: parent, MLFLOW_RUN_NAME: name, "kind": kind}).info.run_id)
        try:
            yield run
        except BaseException as e:
            run.tags(error=str(e)[:5000])
            self.client.log_text(run.run_id, traceback.format_exc(), "error.txt")
            if log is not None and log.is_file():
                run.artifact(log)
            self.client.set_terminated(run.run_id, "KILLED" if isinstance(e, StepCancelled) else "FAILED")
            raise
        self.client.set_terminated(run.run_id, "FINISHED")

    def close_interrupted(self, jobs: list[dict]) -> None:
        """Runs of jobs cut off by a restart are still RUNNING in MLflow; end them as KILLED."""
        for job in jobs:
            e = self.client.get_experiment_by_name(f"project-{job['project_id']}")
            if e is None:
                continue
            for r in self.client.search_runs([e.experiment_id], f"params.job_id = '{job['id']}' "
                                             "and attributes.status = 'RUNNING'"):
                self.client.set_tag(r.info.run_id, "error", "interrupted by a restart")
                self.client.set_terminated(r.info.run_id, "KILLED")

    def url(self, run_id: str) -> str:
        r = self.client.get_run(run_id)
        return f"{self.uri.rstrip('/')}/#/experiments/{r.info.experiment_id}/runs/{run_id}"
