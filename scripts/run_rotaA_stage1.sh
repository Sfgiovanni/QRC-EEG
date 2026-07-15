#!/usr/bin/env bash
# Rota A Stage 1 only. The verifier is deliberately the final command.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN=$PYTHON
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" scripts/run_effective_kernel_check.py
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/verify_rotaA_gate1.py
