#!/usr/bin/env bash
# One-command deterministic reproduction. The final command is the fail-high
# phase-1 gate; nothing runs after it.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN=$PYTHON
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN=python3
fi

LOG=${QRC_EEG_LOG:-results/eeg/run.log}
mkdir -p "$(dirname "$LOG")"
: >"$LOG"

run_step() {
  local name=$1
  shift
  echo "[$(date -Is)] $name" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run_step fetch_eeg "$PYTHON_BIN" scripts/fetch_eeg.py
run_step sanity_checks "$PYTHON_BIN" scripts/run_sanity_checks.py
run_step hp_search "$PYTHON_BIN" scripts/run_hp_search.py
run_step esn66_hp_search "$PYTHON_BIN" scripts/run_esn66_hp_search.py
run_step holdout_eval "$PYTHON_BIN" scripts/run_holdout_eval.py
run_step esn66_holdout "$PYTHON_BIN" scripts/run_esn66_holdout.py
run_step quadratic_capacity "$PYTHON_BIN" scripts/run_quadratic_capacity.py
run_step statistics "$PYTHON_BIN" scripts/run_statistics.py
run_step esn66_contrasts "$PYTHON_BIN" scripts/run_esn66_contrasts.py
run_step long_horizon_contrasts "$PYTHON_BIN" scripts/run_long_horizon_contrasts.py
run_step tables_figures "$PYTHON_BIN" scripts/run_tables_figures.py
run_step long_horizon_figure "$PYTHON_BIN" scripts/run_overnight_figures.py
run_step update_results "$PYTHON_BIN" scripts/update_results_fase1.py
run_step fase1_diff_report "$PYTHON_BIN" scripts/make_fase1_diff_report.py
run_step pytest "$PYTHON_BIN" -m pytest -q
echo "[$(date -Is)] pipeline_stages_complete" | tee -a "$LOG"
run_step verify_fase1 env QRC_EEG_LOG="$LOG" "$PYTHON_BIN" scripts/verify_fase1.py
