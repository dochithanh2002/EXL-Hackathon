#!/usr/bin/env bash
# Prompts for the platform login (not stored, not echoed), then captures the M4 project.
set -euo pipefail
cd "$(dirname "$0")"
read -r -p "Platform username: " R2PY_USER
read -r -s -p "Platform password: " R2PY_PASS; echo
export R2PY_USER R2PY_PASS
uv run python "${2:-capture.py}" "${1:-M4}"
