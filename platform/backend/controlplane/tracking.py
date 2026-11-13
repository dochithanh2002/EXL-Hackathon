"""MLflow: one experiment per project; every step run is nested under the project's workflow parent run."""
from __future__ import annotations

import os
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities import Metric, Param, RunTag  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME  # noqa: E402

from .runner import StepCancelled  # noqa: E402


class StepRun:
    def __init__(self, client: MlflowClient, run_id: str):
        self.client, self.run_id = client, run_id

    def params(self, **values) -> None:
        self.client.log_batch(self.run_id, params=[Param(k, str(v)) for k, v in values.items()])

    def metrics(self, **values: float) -> None:
        self.client.log_batch(self.run_id, metrics=[Metric(k, float(v), 0, 0) for k, v in values.items()])

    def tags(self, **values) -> None:
        self.client.log_batch(self.run_id, tags=[RunTag(k, str(v)) for k, v in values.items()])

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
    def step(self, pid: str, kind: str, name: str, log: Path | None = None) -> Iterator[StepRun]:
        """A nested run that ends FINISHED, or FAILED with the traceback and the step log as artifacts."""
        parent = self.workflow_run(pid)
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

    def url(self, run_id: str) -> str:
        r = self.client.get_run(run_id)
        return f"{self.uri.rstrip('/')}/#/experiments/{r.info.experiment_id}/runs/{run_id}"
