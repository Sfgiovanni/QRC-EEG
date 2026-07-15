#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-gate3}"

echo "Gate 2 addendum -> analytical CSV/document/figure; Gate 3 -> frozen resource table, finite-shot rows, report, tests, hashes, then stop."
python scripts/make_gate2_postgate_addendum.py
python scripts/make_physical_resource_table.py
python scripts/run_shot_sensitivity.py all
python scripts/verify_gate3.py
