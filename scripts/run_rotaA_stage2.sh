#!/usr/bin/env bash
# Rota A Stage 2 only. Prediction is frozen before measurement; verifier is final.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN=$PYTHON
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN=python3
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-stage2}"
"$PYTHON_BIN" scripts/run_synthetic_stage2.py predict
"$PYTHON_BIN" scripts/run_synthetic_stage2.py measure
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/verify_rotaA_gate2.py
