#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-release}"

echo "Rebuilding repository-only CSVs, documentation indexes and figures from frozen Gate 0–3 artifacts; no simulations and no .tex writes."
.venv/bin/python scripts/build_repository_release.py data
.venv/bin/python scripts/make_final_figures.py
.venv/bin/python scripts/build_repository_release.py index
.venv/bin/python scripts/verify_repository_release.py
