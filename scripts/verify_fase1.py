#!/usr/bin/env python3
"""Fail-high final gate for segment-blocked ridge/HP selection."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "eeg"
SNAPSHOT = RESULTS / "_prefix_snapshot_hpselect"


def fail(message: str) -> None:
    raise SystemExit(f"FASE1 GATE FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_grouping_and_target_leakage() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_segment_grouped_selection.py",
            "tests/test_mechanism_checks.py::test_shuffled_target_leakage_r2_near_zero",
        ],
        cwd=ROOT,
    )
    if result.returncode:
        fail("segment grouping or shuffled-target leakage test failed")
    source = (ROOT / "src" / "qrc_eeg" / "pipeline.py").read_text()
    if "rng.permutation" in source or "segment leakage across ridge" not in source:
        fail("pipeline still contains row-random selection or lacks segment-overlap guard")


def check_results_text() -> None:
    text = (ROOT / "RESULTS.md").read_text()
    for stale in ("R^2 = 0.92", "R^2 = 0.97", "0.2627", "[-0.0117, 0.0013]", "competitive with"):
        if stale in text:
            fail(f"stale/inflated RESULTS.md claim remains: {stale}")
    matched = pd.read_csv(RESULTS / "tab_esn_matched.csv")
    fz = matched[matched["set"].isin(["F", "Z"])]
    short, long = fz[fz.horizon.isin([1, 2])], fz[fz.horizon.isin([4, 8])]
    expected = int((short.mean_diff_rmse_comparator_minus_state < 0).sum() + (long.mean_diff_rmse_comparator_minus_state > 0).sum())
    short_sig = int(((short.mean_diff_rmse_comparator_minus_state < 0) & (short.p_holm < 0.05)).sum())
    long_sig = int(((long.mean_diff_rmse_comparator_minus_state > 0) & (long.p_holm < 0.05)).sum())
    expected_sentence = f"expected ESN-short/kernel-long direction occurs in {expected}/8 cells"
    significance_sentence = f"are {short_sig} at short horizons and {long_sig} at long horizons"
    if expected_sentence not in text or significance_sentence not in text:
        fail("RESULTS.md horizon-interaction counts do not match tab_esn_matched.csv")
    ab = pd.read_csv(RESULTS / "tab_eeg_contrasts.csv")
    ab = ab[ab.comparison == "single_kernel vs AB_noaux"]
    positive = int((ab.mean_diff_rmse_comparator_minus_state > 0).sum())
    significant = int(((ab.mean_diff_rmse_comparator_minus_state > 0) & (ab.p_holm < 0.05)).sum())
    if f"lower NRMSE in {positive}/12" not in text or f"{significant}/12 Holm-significant" not in text:
        fail("RESULTS.md AB counts do not match tab_eeg_contrasts.csv")
    regression = pd.read_csv(RESULTS / "capacity_demand_regression_summary.csv")
    cap = regression[regression.predictor == "capacity_gap"].iloc[0]
    if f"slope is {cap.slope:+.6f}" not in text or f"with n={int(cap['n'])}" not in text:
        fail("RESULTS.md capacity statement does not match regression CSV")


def check_reproduction_record() -> None:
    script = (ROOT / "scripts" / "run_eeg.sh").read_text()
    required_names = [
        "fetch_eeg", "sanity_checks", "hp_search", "esn66_hp_search", "holdout_eval", "esn66_holdout",
        "quadratic_capacity", "statistics", "esn66_contrasts", "tables_figures", "fase1_diff_report",
        "pytest", "verify_fase1",
    ]
    missing_script = [name for name in required_names if name not in script]
    if missing_script:
        fail(f"run_eeg.sh misses required stages: {missing_script}")
    log_path = Path(os.environ.get("QRC_EEG_LOG", RESULTS / "run_fase1.log"))
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    if not log_path.exists():
        fail(f"reproduction log missing: {log_path}")
    log = log_path.read_text(errors="replace")
    missing_log = [name for name in required_names[:-1] if f"] {name}" not in log]
    if missing_log or "] pipeline_stages_complete" not in log:
        fail(f"full run_eeg reproduction did not complete pre-gate stages: {missing_log}")


def check_snapshot_report_and_alphas() -> None:
    required = [
        SNAPSHOT / "tab_eeg_endpoints.csv",
        SNAPSHOT / "tab_esn_matched.csv",
        SNAPSHOT / "selected_alphas.csv",
        RESULTS / "selected_alphas.csv",
        RESULTS / "selected_alphas_esn66.csv",
        RESULTS / "fase1_diff_report.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        fail(f"snapshot/report/alpha artifacts missing: {missing}")
    before = pd.read_csv(SNAPSHOT / "selected_alphas.csv")
    after = pd.concat([pd.read_csv(RESULTS / "selected_alphas.csv"), pd.read_csv(RESULTS / "selected_alphas_esn66.csv")])
    keys = ["construction", "set", "horizon", "seed"]
    if len(before.merge(after, on=keys, validate="one_to_one")) != 840:
        fail("before/after alpha grid is incomplete; expected 840 matched choices")
    report = (RESULTS / "fase1_diff_report.md").read_text()
    for marker in ("Endpoints", "Kernel vs ESN-66", "Kernel vs AB", "Selected ridge alpha", "Factual verdict"):
        if marker not in report:
            fail(f"fase1 diff report lacks section: {marker}")
    if sum(report.count(status) for status in ("SOBREVIVEU", "ENFRAQUECEU", "SUMIU")) < 2:
        fail("fase1 report lacks one verdict per headline")


def update_and_check_hashes() -> None:
    result = subprocess.run([sys.executable, "scripts/make_provenance.py"], cwd=ROOT)
    if result.returncode:
        fail("SHA256 provenance generation failed")
    checksums = (ROOT / "provenance" / "eeg_checksums.txt").read_text()
    for relative in (
        "results/eeg/fase1_diff_report.md",
        "results/eeg/selected_alphas.csv",
        "results/eeg/selected_alphas_esn66.csv",
        "results/eeg/tab_eeg_endpoints.csv",
    ):
        if relative not in checksums:
            fail(f"checksum missing for {relative}")


def main() -> None:
    check_grouping_and_target_leakage()
    check_results_text()
    check_reproduction_record()
    check_snapshot_report_and_alphas()
    update_and_check_hashes()
    print("\nFASE1 GATE: PASS")
    print("ridge/HP selection: disjoint whole segments; overlap guard passed")
    print("shuffled target: held-out R2 near-zero test passed")
    print("RESULTS.md: regenerated numbers match CSVs; stale claims absent")
    print("run_eeg.sh: all deterministic stages completed before this gate")
    print("snapshot + numeric diff + 840 matched alpha choices: present")
    for relative in (
        "results/eeg/fase1_diff_report.md",
        "results/eeg/selected_alphas.csv",
        "results/eeg/selected_alphas_esn66.csv",
    ):
        print(f"{sha256(ROOT / relative)}  {relative}")


if __name__ == "__main__":
    main()
