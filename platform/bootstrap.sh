#!/usr/bin/env bash
# Build the stable environments from their lockfiles (run once, and again after a lockfile changes).
#   ./bootstrap.sh                  restore envs/r, envs/python, envs/mlflow, the backend and the agent service
#   ./bootstrap.sh relock <date>    re-resolve envs/r/renv.lock against the Posit Package Manager snapshot of <date>
# Needs R (the version in renv.lock), uv, Node and a C/C++/Fortran toolchain; CMake is pinned and fetched by uv.
set -euo pipefail
cd "$(dirname "$0")"
CMAKE_VERSION=4.4.4

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
json() { uv run --no-project --python 3.12 python -c "import json,sys; d=json.load(open('$1')); print($2)"; }

# CMake is needed to build nloptr (lme4). A shim on PATH runs the pinned PyPI build.
tools=$(mktemp -d); trap 'rm -rf "$tools"' EXIT
printf '#!/bin/sh\nexec uvx --from cmake==%s cmake "$@"\n' "$CMAKE_VERSION" > "$tools/cmake"
chmod +x "$tools/cmake"
export PATH="$tools:$PATH" MAKEFLAGS="-j$(getconf _NPROCESSORS_ONLN)"

if [[ "${1:-}" == relock ]]; then
  rm -rf envs/r/library
  Rscript --vanilla envs/r/lock.R "${2:?usage: ./bootstrap.sh relock <yyyy-mm-dd>}"
  sha256 envs/r/renv.lock > envs/r/library/.lock-sha256
  exit
fi

r_want=$(json envs/r/renv.lock "d['R']['Version']")
r_have=$(Rscript --vanilla -e 'cat(as.character(getRversion()))')
[[ "$r_want" == "$r_have" ]] || { echo "renv.lock needs R $r_want, found R $r_have" >&2; exit 1; }

echo "== R environment (R $r_have)"
if [[ "$(cat envs/r/library/.lock-sha256 2>/dev/null)" != "$(sha256 envs/r/renv.lock)" ]]; then
  rm -f envs/r/library/.lock-sha256
  REPO=$(json envs/r/renv.lock "d['R']['Repositories'][0]['URL']") Rscript --vanilla - <<'R'
lockfile <- "envs/r/renv.lock"; lib <- "envs/r/library"
dir.create(lib, showWarnings = FALSE)
.libPaths(lib)
options(repos = c(CRAN = Sys.getenv("REPO")), Ncpus = parallel::detectCores())
if (!requireNamespace("renv", lib.loc = lib, quietly = TRUE)) install.packages("renv", lib = lib)
project <- tempfile("renv-restore-"); dir.create(project)
renv::restore(project = project, library = lib, lockfile = lockfile, clean = TRUE, prompt = FALSE)
R
  sha256 envs/r/renv.lock > envs/r/library/.lock-sha256
fi

echo "== Python environment"
uv sync --frozen --project envs/python
sha256 envs/python/uv.lock > envs/python/.venv/.lock-sha256

echo "== MLflow server"
uv sync --frozen --project envs/mlflow

echo "== Control plane"
(cd backend && uv sync --frozen)

echo "== Agent service"
(cd agents && npm ci --no-audit --no-fund)
