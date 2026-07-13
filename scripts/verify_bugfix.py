#!/usr/bin/env python3
"""Fail-high final gate for the causal-preprocessing bugfix rerun."""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.preprocessing import scale_set_from_training  # noqa: E402

RESULTS = ROOT / "results" / "eeg"
SNAPSHOT = RESULTS / "_prefix_snapshot"


def fail(message: str) -> None:
    raise SystemExit(f"BUGFIX GATE FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_leakage_tests() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_bugfix_leakage.py",
            "tests/test_mechanism_checks.py::test_shuffled_target_leakage_r2_near_zero",
        ],
        cwd=ROOT,
    )
    if result.returncode:
        fail("shuffled-target or causal-scaling leakage test failed")


def check_causal_normalization() -> None:
    holdout_source = (ROOT / "scripts" / "run_holdout_eval.py").read_text()
    if "scale_set_from_training" not in holdout_source or "zscore(" in holdout_source:
        fail("held-out evaluation is not exclusively using training-fitted scaling")
    raw = {"train": np.arange(8.0), "test": np.arange(8.0) + 20.0}
    scaled_a, scaler_a = scale_set_from_training(raw, ["train"])
    changed = dict(raw)
    changed["test"] = np.array([20.0, 21.0, 22.0, 23.0, 1e9, -1e9, 1e8, -1e8])
    scaled_b, scaler_b = scale_set_from_training(changed, ["train"])
    if scaler_a != scaler_b or not np.array_equal(scaled_a["test"][:4], scaled_b["test"][:4]):
        fail("future test samples changed scaling of earlier test samples")


def check_capacity_alpha_is_validation_only() -> None:
    path = ROOT / "scripts" / "run_quadratic_capacity.py"
    tree = ast.parse(path.read_text())
    select_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select_capacity_alpha"), None)
    if select_fn is None:
        fail("capacity alpha-selection function missing")
    names = {node.id for node in ast.walk(select_fn) if isinstance(node, ast.Name)}
    if "test_idx" in names:
        fail("capacity alpha selection references reserved test_idx")
    capacity_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "capacity_for_kind"), None)
    calls = [node for node in ast.walk(capacity_fn) if isinstance(node, ast.Call)] if capacity_fn else []
    if not any(isinstance(call.func, ast.Name) and call.func.id == "select_capacity_alpha" for call in calls):
        fail("capacity evaluation does not call validation-only alpha selection")


def check_snapshot_and_report() -> None:
    required = {
        "tab_eeg_endpoints.csv",
        "tab_eeg_contrasts.csv",
        "tab_esn_matched.csv",
        "tab_long_horizon_contrasts.csv",
        "quadratic_capacity.csv",
        "linear_capacity.csv",
        "capacity_demand_regression_summary.csv",
        "capacity_gain_regression_rows.csv",
        "overnight_summary.md",
    }
    missing = sorted(name for name in required if not (SNAPSHOT / name).exists())
    if missing:
        fail(f"prefix snapshot incomplete: {missing}")
    report = RESULTS / "bugfix_diff_report.md"
    if not report.exists():
        fail("bugfix diff report missing")
    text = report.read_text()
    for marker in ("before", "after", "Kernel x ESN crossover", "Kernel > AB", "Capacity-gain association"):
        if marker.lower() not in text.lower():
            fail(f"diff report lacks required marker: {marker}")
    if sum(text.count(status) for status in ("SOBREVIVEU", "ENFRAQUECEU", "SUMIU")) < 3:
        fail("diff report lacks one verdict per headline")
    if sum(character.isdigit() for character in text) < 100:
        fail("diff report does not contain enough numeric before/after evidence")


def check_outputs() -> None:
    contrasts = pd.read_csv(RESULTS / "tab_eeg_contrasts.csv")
    if "n_segments" not in contrasts or "n_patients" in contrasts:
        fail("statistical output is not labeled n_segments")
    capacity_summary = pd.read_csv(RESULTS / "capacity_demand_regression_summary.csv")
    capacity = capacity_summary.loc[capacity_summary["predictor"] == "capacity_gap"]
    if len(capacity) != 1 or int(capacity.iloc[0]["n"]) != 3:
        fail("capacity association does not use the three independent gaps")
    lengths = {sum(1 for line in path.open() if line.strip()) for path in (ROOT / "data" / "eeg" / "sets").glob("[ZFS]/*.txt")}
    if lengths != {4097}:
        fail(f"unexpected sample counts in pinned data: {sorted(lengths)}")


def write_provenance() -> None:
    result = subprocess.run([sys.executable, "scripts/make_provenance.py"], cwd=ROOT)
    if result.returncode:
        fail("provenance checksum generation failed")
    checksums = (ROOT / "provenance" / "eeg_checksums.txt").read_text()
    for relative in (
        "results/eeg/bugfix_diff_report.md",
        "results/eeg/tab_eeg_endpoints.csv",
        "results/eeg/tab_eeg_contrasts.csv",
        "results/eeg/quadratic_capacity.csv",
    ):
        if relative not in checksums:
            fail(f"checksum missing for {relative}")


def main() -> None:
    check_leakage_tests()
    check_causal_normalization()
    check_capacity_alpha_is_validation_only()
    check_snapshot_and_report()
    check_outputs()
    write_provenance()
    print("\nBUGFIX GATE: PASS")
    print("shuffled target R2: near-zero test passed")
    print("normalization: frozen training-only statistics; future-invariance passed")
    print("capacity alpha: validation-only; reserved test evaluated once")
    print("statistical unit: n_segments; capacity gaps: n=3")
    print("snapshot + numeric before/after report: present")
    for relative in (
        "results/eeg/bugfix_diff_report.md",
        "results/eeg/tab_eeg_endpoints.csv",
        "results/eeg/tab_eeg_contrasts.csv",
        "results/eeg/quadratic_capacity.csv",
    ):
        path = ROOT / relative
        print(f"{sha256(path)}  {relative}")


if __name__ == "__main__":
    main()
