#!/usr/bin/env bash

# Run the big demo pipeline end-to-end followed by evaluation scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts/big_demo}"

DEFAULT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="${PROJECT_ROOT}/results/big_demo/logs"
mkdir -p "${log_dir}"

echo "[1/3] Running pipeline options..."
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/big_demo/pipeline_options.py" \
  --execute \
  --dataset robot \
  --log-file "${log_dir}/pipeline_${timestamp}.log"

echo "[2/3] Evaluating conceptual safeguards (sudoku)..."
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/big_demo/eval_conceptual_safeguards.py" \
  --data-names sudoku \
  --output "${PROJECT_ROOT}/results/big_demo/conceptual_safeguards_sudoku.csv"

echo "[3/3] Evaluating score intervention (robot)..."
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/big_demo/eval_score_intervention.py" \
  --data-names robot \
  --output "${PROJECT_ROOT}/results/big_demo/score_intervention_robot.csv"

echo "All stages completed successfully."

