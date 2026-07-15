#!/usr/bin/env python3
"""Final fail-high verifier for the preregistered causal-memory horizon gate."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "eeg"
SNAPSHOT = RESULTS / "_prefix_snapshot_gate"
CONFIG = ROOT / "config" / "eeg_frozen.yaml"


def fail(message: str) -> None:
    raise SystemExit(f"EEG GATE VERIFICATION FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_leak_and_control_tests() -> None:
    result = subprocess.run([
        sys.executable, "-m", "pytest", "-q",
        "tests/test_gate_controls.py",
        "tests/test_segment_grouped_selection.py",
        "tests/test_bugfix_leakage.py",
        "tests/test_mechanism_checks.py::test_shuffled_target_leakage_r2_near_zero",
    ], cwd=ROOT)
    if result.returncode:
        fail("K=0 distinction, grouping, causal scaling, or shuffled-target test failed")
    pipeline = (ROOT / "src/qrc_eeg/pipeline.py").read_text()
    holdout = (ROOT / "scripts/run_holdout_eval.py").read_text()
    classical = (ROOT / "scripts/run_gate_baselines.py").read_text()
    if "rng.permutation" in pipeline or "segment leakage across ridge" not in pipeline:
        fail("active ridge selection is not guarded whole-segment validation")
    if "scale_set_from_training" not in holdout or "scale_set_from_training" not in classical:
        fail("gate paths do not use causal train-fitted normalization")
    if "assert_disjoint_segment_ids(train_ids, val_ids)" not in classical:
        fail("classical HP selection lacks explicit segment grouping guard")


def check_preregistration_and_outputs() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    if cfg["readout"]["horizons"] != [1, 2, 4, 8, 16, 32, 64, 128]:
        fail("extended frozen horizon grid changed")
    prereg = ROOT / "docs/eeg_gate_preregistration.md"
    required = [
        prereg, SNAPSHOT / "tab_eeg_endpoints.csv", SNAPSHOT / "raw/eeg_holdout_by_segment_seed.csv",
        RESULTS / "raw/eeg_gate_classical_by_segment_seed.csv", RESULTS / "gate_classical_selected_hp.csv",
        RESULTS / "gate_nrmse_curves.csv", RESULTS / "gate_interactions.csv",
        RESULTS / "useful_horizon.csv", RESULTS / "gate_report.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        fail(f"snapshot or gate artifact missing: {missing}")
    if prereg.stat().st_mtime > (RESULTS / "raw/eeg_gate_classical_by_segment_seed.csv").stat().st_mtime:
        fail("preregistration timestamp is later than gate model output")
    interactions = pd.read_csv(RESULTS / "gate_interactions.csv")
    if len(interactions) != 15 or set(interactions["comparator"]) != {"QRC_K0", "AR", "NVAR2", "persistence", "tapped_delay"}:
        fail("eeg_gate family is not the frozen 15 contrasts")
    if not (interactions["n_segments"] == 20).all():
        fail("interaction unit is not 20 held-out segments per set")
    curves = pd.read_csv(RESULTS / "gate_nrmse_curves.csv")
    gate_models = {"single_kernel", "QRC_K0", "AR", "NVAR2", "persistence", "tapped_delay"}
    expected = len(gate_models) * 3 * 8
    if len(curves[curves["construction"].isin(gate_models)]) != expected:
        fail("gate NRMSE curves do not cover every model/set/horizon")
    expected_ms = curves["horizon"] * 1000.0 / float(cfg["data"]["sampling_rate_hz"])
    if not np.allclose(curves["horizon_ms"], expected_ms):
        fail("horizon milliseconds do not match sampling rate")
    report = (RESULTS / "gate_report.md").read_text()
    if report.count("Mechanical verdict: **PASS**") + report.count("Mechanical verdict: **FAIL**") != 1:
        fail("gate report lacks exactly one mechanical PASS/FAIL verdict")
    for marker in ("Useful horizon", "QRC_K0", "tapped_delay", "Frozen interaction family", "Set S"):
        if marker not in report:
            fail(f"gate report lacks required content: {marker}")


def check_results_consistency() -> None:
    text = (ROOT / "RESULTS.md").read_text()
    for stale in ("R^2 = 0.92", "R^2 = 0.97", "0.2627", "[-0.0117, 0.0013]", "competitive with"):
        if stale in text:
            fail(f"stale RESULTS.md number/claim remains: {stale}")
    report = (RESULTS / "gate_report.md").read_text()
    verdict = "PASS" if "Mechanical verdict: **PASS**" in report else "FAIL"
    if f"verdict is **{verdict}**" not in text:
        fail("RESULTS.md verdict disagrees with gate_report.md")
    interactions = pd.read_csv(RESULTS / "gate_interactions.csv")
    for set_name in ("F", "Z"):
        row = interactions[(interactions["set"] == set_name) & (interactions["comparator"] == "QRC_K0")].iloc[0]
        token = f"kernel-vs-K=0 interaction={row.interaction_comp_minus_kernel:+.6f}"
        if token not in text:
            fail(f"RESULTS.md K=0 number differs for set {set_name}")
    cap = pd.read_csv(RESULTS / "capacity_demand_regression_summary.csv")
    row = cap[cap["predictor"] == "capacity_gap"].iloc[0]
    if f"slope {row.slope:+.6f} with n={int(row['n'])}" not in text:
        fail("RESULTS.md capacity number differs from CSV")


def check_reproduction_log() -> None:
    script = (ROOT / "scripts/run_eeg.sh").read_text()
    stages = [
        "fetch_eeg", "sanity_checks", "hp_search", "holdout_eval", "gate_baselines",
        "quadratic_capacity", "statistics", "gate_report", "update_results", "pytest", "verify_gate",
    ]
    missing = [stage for stage in stages if stage not in script]
    if missing or script.rstrip().splitlines()[-1].find("verify_gate") < 0:
        fail(f"run_eeg.sh missing stages or verify_gate is not last: {missing}")
    log_path = Path(os.environ.get("QRC_EEG_LOG", RESULTS / "run_gate.log"))
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    if not log_path.exists():
        fail(f"full-run log missing: {log_path}")
    log = log_path.read_text(errors="replace")
    missing_log = [stage for stage in stages[:-1] if f"] {stage}" not in log]
    if missing_log or "] pipeline_stages_complete" not in log:
        fail(f"run_eeg.sh did not finish every pre-gate stage: {missing_log}")


def update_and_check_hashes() -> None:
    result = subprocess.run([sys.executable, "scripts/make_provenance.py"], cwd=ROOT)
    if result.returncode:
        fail("SHA256 provenance generation failed")
    checksums = (ROOT / "provenance/eeg_checksums.txt").read_text()
    for relative in (
        "docs/eeg_gate_preregistration.md", "results/eeg/gate_report.md",
        "results/eeg/gate_interactions.csv", "results/eeg/useful_horizon.csv",
        "results/eeg/raw/eeg_gate_classical_by_segment_seed.csv",
    ):
        if relative not in checksums:
            fail(f"SHA256 missing for {relative}")


def main() -> None:
    check_leak_and_control_tests()
    check_preregistration_and_outputs()
    check_results_consistency()
    check_reproduction_log()
    update_and_check_hashes()
    report = (RESULTS / "gate_report.md").read_text()
    verdict = "PASS" if "Mechanical verdict: **PASS**" in report else "FAIL"
    print(f"\nEEG GATE VERIFICATION: PASS; MECHANICAL SCIENTIFIC VERDICT: {verdict}")
    print("K=0 dynamics, segment grouping, causal scaling and shuffled target: passed")
    print("eight horizons + milliseconds + useful horizon + 15 Holm contrasts: complete")
    print("RESULTS.md matches generated CSVs; snapshot and full run log: present")
    for relative in ("results/eeg/gate_report.md", "results/eeg/gate_interactions.csv", "results/eeg/useful_horizon.csv"):
        print(f"{sha256(ROOT / relative)}  {relative}")


if __name__ == "__main__":
    main()
