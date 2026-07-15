#!/usr/bin/env bash
# Rota A Stage 0 only. The verifier is deliberately the final command.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN=$PYTHON
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" scripts/make_useful_horizon_v2.py
"$PYTHON_BIN" scripts/update_results_gate.py
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/verify_rotaA_gate0.py
