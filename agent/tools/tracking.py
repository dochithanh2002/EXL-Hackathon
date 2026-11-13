"""MLflow logging for agent stages (migration, documentation, human review, run summary).

Tracking store: Azure ML when agent/config/azure.env is filled in, otherwise output/mlflow/ (mlflow_setup.py).
Experiment name comes from model.yaml `tracking.experiment`. Logging is best-effort: a failure is written to the
audit trail and never fails the agent run.
"""
from __future__ import annotations

import re
from pathlib import Path


def _key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-./ ]", "_", s)[:250]


def log_run(m, st, run_name: str, tags: dict, params: dict, metrics: dict, artifacts: dict[str, Path]) -> str | None:
    try:
        import mlflow

        from agent.tools.mlflow_setup import init_tracking
        init_tracking(m.experiment())
        with mlflow.start_run(run_name=run_name) as r:
            mlflow.set_tags({_key(k): str(v) for k, v in tags.items()})
            mlflow.log_params({_key(k): str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics({_key(k): float(v) for k, v in metrics.items() if v is not None})
            for name, path in artifacts.items():
                if path.is_dir():
                    mlflow.log_artifacts(str(path), name)
                elif path.exists():
                    mlflow.log_artifact(str(path), name)
            st.audit("mlflow_logged", run_name=run_name, mlflow_run_id=r.info.run_id, experiment=m.experiment())
            return r.info.run_id
    except Exception as e:  # tracking must never break the agent
        st.audit("mlflow_failed", run_name=run_name, error=f"{type(e).__name__}: {e}"[:500])
        return None


def tag_review(m, st, run_id: str | None, decision: dict) -> None:
    if not run_id:
        return
    try:
        from mlflow.tracking import MlflowClient

        from agent.tools.mlflow_setup import init_tracking
        init_tracking(m.experiment())
        c = MlflowClient()
        for k in ("decision", "reviewer", "comment", "ts_utc"):
            c.set_tag(run_id, f"human_review.{k}", str(decision.get(k, "")))
        c.set_tag(run_id, "status", st.state["status"])
        st.audit("mlflow_tagged", mlflow_run_id=run_id, decision=decision["decision"])
    except Exception as e:
        st.audit("mlflow_failed", run_name="human_review", error=f"{type(e).__name__}: {e}"[:500])
