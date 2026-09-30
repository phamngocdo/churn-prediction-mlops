#!/usr/bin/env bash
#
# Run the full training pipeline: load -> split -> train -> evaluate ->
# save processed data.
#
# This wraps `python scripts/run_training.py` so that Docker, Makefile,
# CI (GitHub Actions) and Prefect can all call ONE stable entrypoint
# without needing to know about PYTHONPATH or the src-layout package
# structure underneath.
#
# Usage:
#   ./scripts/run_training.sh
#
# Env vars (all optional, see src/churn_mlops/config.py for defaults):
#   MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODEL_REGISTRY_NAME
#   RAW_DATA_DIR, PROCESSED_DATA_DIR

set -euo pipefail

# Resolve project root regardless of where the script is called from,
# so `./scripts/run_training.sh` works the same from repo root, CI, or
# inside a Docker container with a different WORKDIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[run_training] Project root: ${PROJECT_ROOT}"

# Activate a local virtualenv if present (skip silently in Docker/CI,
# where dependencies are typically installed globally in the image).
if [ -f ".venv/bin/activate" ]; then
    echo "[run_training] Activating .venv"
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Make the `churn_mlops` package importable without requiring
# `pip install -e .` — useful for quick local runs and CI caching.
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "[run_training] MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-<default from config.py>}"
echo "[run_training] Starting training pipeline..."

python src/churn_mlops/models/run_training.py "$@"

echo "[run_training] Done."