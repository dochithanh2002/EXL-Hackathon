"""MLflow tracking: Azure ML workspace if agent/config/azure.env is filled in, else output/mlflow/ (local)."""
import os
from pathlib import Path

import mlflow

from agent.tools.paths import CONFIG, MLFLOW

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v:
                os.environ.setdefault(k.strip(), v.strip())


def init_tracking(experiment: str) -> str:
    _load_env(CONFIG / "azure.env")
    sub = os.getenv("AZURE_SUBSCRIPTION_ID")
    if sub:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml = MLClient(DefaultAzureCredential(), sub, os.environ["AZURE_RESOURCE_GROUP"], os.environ["AZURE_ML_WORKSPACE"])
        uri = ml.workspaces.get(ml.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        return uri
    MLFLOW.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{MLFLOW / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(MLFLOW / "artifacts" / experiment).as_uri())
    mlflow.set_experiment(experiment)
    return uri
