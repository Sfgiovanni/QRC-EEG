#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
LOG=results/eeg/run_bugfix.log
mkdir -p results/eeg
: > "$LOG"

run_step() {
  local name="$1"
  shift
  echo "[$(date -Is)] $name" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run_step sanity_and_leakage "$PYTHON" scripts/run_sanity_checks.py
run_step bugfix_leakage_tests "$PYTHON" -m pytest -q tests/test_bugfix_leakage.py tests/test_mechanism_checks.py::test_shuffled_target_leakage_r2_near_zero
run_step hp_search_causal "$PYTHON" scripts/run_hp_search.py
run_step esn66_hp_search_causal "$PYTHON" scripts/run_esn66_hp_search.py
run_step holdout_causal "$PYTHON" scripts/run_holdout_eval.py
run_step esn66_holdout_causal "$PYTHON" scripts/run_esn66_holdout.py
run_step capacity_validation_selected "$PYTHON" scripts/run_quadratic_capacity.py
run_step paired_statistics "$PYTHON" scripts/run_statistics.py
run_step esn66_contrasts "$PYTHON" scripts/run_esn66_contrasts.py
run_step long_horizon_contrasts "$PYTHON" scripts/run_long_horizon_contrasts.py
run_step tables_figures "$PYTHON" scripts/run_tables_figures.py
run_step long_horizon_figure "$PYTHON" scripts/run_overnight_figures.py
run_step bugfix_diff_report "$PYTHON" scripts/make_bugfix_diff_report.py
run_step pytest_full "$PYTHON" -m pytest -q
run_step bugfix_gate "$PYTHON" scripts/verify_bugfix.py
