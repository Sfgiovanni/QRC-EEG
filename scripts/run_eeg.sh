#!/usr/bin/env bash
# One-command, deterministic reproduction of the EEG study.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=results/eeg/run.log
mkdir -p results/eeg
echo "[$(date -Is)] fetch_eeg" | tee -a "$LOG"
python scripts/fetch_eeg.py 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] sanity_checks" | tee -a "$LOG"
python scripts/run_sanity_checks.py 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] hp_search" | tee -a "$LOG"
python scripts/run_hp_search.py 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] holdout_eval" | tee -a "$LOG"
python scripts/run_holdout_eval.py 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] quadratic_capacity" | tee -a "$LOG"
python scripts/run_quadratic_capacity.py 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] statistics" | tee -a "$LOG"
python scripts/run_statistics.py 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] tables_figures" | tee -a "$LOG"
python scripts/run_tables_figures.py 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] provenance" | tee -a "$LOG"
python scripts/make_provenance.py 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] pytest (full suite)" | tee -a "$LOG"
python -m pytest -q 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] done" | tee -a "$LOG"
