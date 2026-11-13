#!/usr/bin/env bash
# Start the local processes: MLflow server and control plane. Later build steps add the agent service and the UI.
set -euo pipefail
cd "$(dirname "$0")"
trap 'trap - TERM; kill 0' EXIT INT TERM
mkdir -p var/mlflow
MLFLOW_DISABLE_AGENT_HINT=1 envs/mlflow/.venv/bin/mlflow server --host 127.0.0.1 --port "${MLFLOW_PORT:-5000}" \
  --backend-store-uri "sqlite:///$PWD/var/mlflow/mlflow.db" --artifacts-destination "$PWD/var/mlflow/artifacts" &
(cd backend && CONTROLPLANE_MLFLOW_URI="http://127.0.0.1:${MLFLOW_PORT:-5000}" uv run --frozen python -m controlplane) &
wait
