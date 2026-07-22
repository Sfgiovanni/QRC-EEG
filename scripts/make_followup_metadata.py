#!/usr/bin/env python3
"""Provenance metadata for the follow-up classical distributed-memory ESN
control and crossed segment x seed inference (Section 6 of the task spec).

Writes results/eeg/followup/metadata.json: origin commit, Python/dependency
versions, OS, thread count, per-stage wall time (from artifact mtimes, since
the runs used nohup rather than `time`), and SHA-256 of every frozen/primary
artifact.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = ROOT / "results" / "eeg" / "followup"

TRACKED_FILES = [
    "docs/classical_distributed_memory_protocol.md",
    "docs/crossed_inference_protocol.md",
    "config/esn_distributed_memory_frozen.yaml",
    "results/eeg/followup/raw/esn_distributed_memory_holdout_by_segment_seed.csv",
    "results/eeg/followup/classical_control/hp_search_log.csv",
    "results/eeg/followup/classical_control/hp_selected.json",
    "results/eeg/followup/classical_control/fixed_core_hp.json",
    "results/eeg/followup/classical_control/tab_classical_distributed_memory.csv",
    "results/eeg/followup/crossed_inference/crossed_bootstrap.csv",
    "results/eeg/followup/crossed_inference/original_style_replication.csv",
    "results/eeg/followup/crossed_inference/mixed_model_results.csv",
    "results/eeg/followup/crossed_inference/mixed_model_diagnostics.json",
]

STAGE_ARTIFACTS = {
    "hp_search_retuned_core": [
        "results/eeg/followup/classical_control/hp_search_log.csv",
        "results/eeg/followup/classical_control/hp_selected.json",
        "results/eeg/followup/classical_control/fixed_core_hp.json",
    ],
    "holdout_both_modes": [
        "results/eeg/followup/raw/esn_distributed_memory_holdout_by_segment_seed.csv",
        "results/eeg/followup/classical_control/selected_alphas.csv",
        "results/eeg/followup/classical_control/tab_resource_accounting.csv",
    ],
    "aggregated_table_and_figure_1": [
        "results/eeg/followup/classical_control/tab_classical_distributed_memory.csv",
        "figures/eeg/fig_classical_distributed_memory.pdf",
    ],
    "crossed_inference": [
        "results/eeg/followup/crossed_inference/crossed_bootstrap.csv",
        "results/eeg/followup/crossed_inference/original_style_replication.csv",
        "results/eeg/followup/crossed_inference/mixed_model_results.csv",
    ],
    "figure_2": ["figures/eeg/fig_crossed_inference.pdf"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_duration_seconds(paths: list[str]) -> float | None:
    mtimes = [os.path.getmtime(ROOT / p) for p in paths if (ROOT / p).exists()]
    if not mtimes:
        return None
    return float(max(mtimes) - min(mtimes)) if len(mtimes) > 1 else 0.0


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout

    deps = {}
    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "statsmodels", "pyyaml", "patsy"):
        try:
            deps[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            deps[pkg] = None

    hashes = {p: sha256(ROOT / p) for p in TRACKED_FILES if (ROOT / p).exists()}
    missing = [p for p in TRACKED_FILES if not (ROOT / p).exists()]

    durations = {stage: stage_duration_seconds(paths) for stage, paths in STAGE_ARTIFACTS.items()}

    metadata = {
        "origin_commit": commit,
        "working_tree_dirty_at_metadata_time": bool(dirty.strip()),
        "python_version": sys.version,
        "dependency_versions": deps,
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "thread_env": {
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "stage_wall_time_seconds_approx_from_artifact_mtimes": durations,
        "artifact_sha256": hashes,
        "artifacts_missing_at_metadata_time": missing,
    }

    FOLLOWUP.mkdir(parents=True, exist_ok=True)
    (FOLLOWUP / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("wrote", FOLLOWUP / "metadata.json")
    if missing:
        print("WARNING: missing artifacts at metadata time:", missing)


if __name__ == "__main__":
    main()
