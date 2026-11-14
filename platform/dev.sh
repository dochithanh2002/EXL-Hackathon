#!/usr/bin/env bash
# Start the local processes: MLflow server, agent service and control plane. A later build step adds the UI.
set -euo pipefail
cd "$(dirname "$0")"
trap 'trap - TERM; kill 0' EXIT INT TERM
mkdir -p var/mlflow var/agents
MLFLOW_DISABLE_AGENT_HINT=1 envs/mlflow/.venv/bin/mlflow server --host 127.0.0.1 --port "${MLFLOW_PORT:-5000}" \
  --backend-store-uri "sqlite:///$PWD/var/mlflow/mlflow.db" --artifacts-destination "$PWD/var/mlflow/artifacts" &
# The agent service gets no host environment: model keys reach it only inside run requests.
(cd agents && env -i PATH="$PATH" HOME="$PWD/../var/agents" AGENTS_PORT="${AGENTS_PORT:-8100}" node server.ts) &
(cd backend && CONTROLPLANE_MLFLOW_URI="http://127.0.0.1:${MLFLOW_PORT:-5000}" \
  CONTROLPLANE_AGENTS_URL="http://127.0.0.1:${AGENTS_PORT:-8100}" \
  CONTROLPLANE_URL="http://127.0.0.1:${CONTROLPLANE_PORT:-8000}" uv run --frozen python -m controlplane) &
wait
