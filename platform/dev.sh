#!/usr/bin/env bash
# Start the local processes. Later build steps add the agent service, MLflow and the UI here.
set -euo pipefail
cd "$(dirname "$0")"
trap 'kill 0' EXIT
(cd backend && uv run --frozen python -m controlplane) &
wait
