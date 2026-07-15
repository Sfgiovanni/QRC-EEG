#!/usr/bin/env python3
"""Fail-high verification for the frozen Rota A synthetic Gate 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/synth"


def fail(message: str) -> None:
    raise SystemExit(f"ROTA A GATE 2 FAILED: {message}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slope(group: pd.DataFrame) -> float:
    curve = group.groupby("horizon").nrmse.mean().sort_index()
    return float(np.polyfit(np.log2(curve.index.to_numpy(float)), curve.to_numpy(), 1)[0])


def check_freezes() -> tuple[dict, dict]:
    cfg = json.loads((ROOT / "config/rotaA_stage2_frozen.json").read_text())
    for line in (RESULTS / "stage2_protocol_frozen.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        if sha(ROOT / relative) != expected:
            fail(f"frozen protocol/config changed: {relative}")
    expected_prediction = (RESULTS / "stage2_predictions_frozen.sha256").read_text().split()[0]
    if sha(RESULTS / "theory_predictions_frozen.csv") != expected_prediction:
        fail("theory prediction changed after freeze")
    metadata = json.loads((RESULTS / "stage2_metadata.json").read_text())
    if metadata["prediction_sha256"] != expected_prediction:
        fail("metadata prediction hash differs")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != cfg["gate1_commit"] or metadata["git_commit"] != commit:
        fail("commit provenance differs from frozen Gate 2")
    return cfg, metadata


def check_rows_and_slopes(cfg: dict) -> pd.DataFrame:
    theory = pd.read_csv(RESULTS / "theory_predictions_raw.csv")
    measured = pd.read_csv(RESULTS / "measured_forecasts_raw.csv")
    expected = len(cfg["scenarios"]) * len(cfg["models"]) * len(cfg["seeds"]) * cfg["split"]["test"] * len(cfg["horizons"])
    for name, raw in (("theory", theory), ("measured", measured)):
        if len(raw) != expected or not np.isfinite(raw[["nrmse", "selected_alpha"]]).all().all():
            fail(f"{name} raw rows/finiteness invalid: {len(raw)} != {expected}")
        if set(raw.horizon) != set(cfg["horizons"]) or set(raw.model) != set(cfg["models"]):
            fail(f"{name} horizons/models differ from freeze")
    combined = pd.read_csv(RESULTS / "theory_predictions_vs_measured.csv")
    if len(combined) != len(cfg["scenarios"]) * len(cfg["models"]):
        fail("combined summary row count differs")
    for (scenario, model), group in theory.groupby(["scenario", "model"]):
        stored = combined[(combined.scenario == scenario) & (combined.model == model)].predicted_slope.iloc[0]
        if not np.isclose(stored, slope(group), rtol=1e-12, atol=1e-14):
            fail(f"predicted slope mismatch: {scenario}/{model}")
    for (scenario, model), group in measured.groupby(["scenario", "model"]):
        stored = combined[(combined.scenario == scenario) & (combined.model == model)].measured_slope.iloc[0]
        if not np.isclose(stored, slope(group), rtol=1e-12, atol=1e-14):
            fail(f"measured slope mismatch: {scenario}/{model}")
    return combined


def check_verdict(combined: pd.DataFrame, metadata: dict) -> str:
    stats = metadata["statistics"]
    rho = float(spearmanr(combined.predicted_slope, combined.measured_slope).statistic)
    within = combined.groupby("scenario").apply(
        lambda x: spearmanr(x.predicted_slope, x.measured_slope).statistic,
        include_groups=False,
    )
    predicted_best = combined.loc[combined.groupby("scenario").predicted_slope.idxmin()].set_index("scenario").model
    measured_best = combined.loc[combined.groupby("scenario").measured_slope.idxmin()].set_index("scenario").model
    match = float((predicted_best == measured_best).mean())
    if not np.isclose(rho, stats["aggregate_spearman"]) or not np.isclose(np.median(within), stats["median_within_scenario_spearman"]):
        fail("stored correlations differ from combined CSV")
    if not np.isclose(match, stats["best_model_match_fraction"]):
        fail("stored best-model match differs")
    verdict = "SUPPORTED" if (
        stats["aggregate_spearman_ci_low"] > 0 and np.median(within) >= 0.5 and match >= 0.6
    ) else ("PARTIAL" if rho > 0 and np.median(within) > 0 else "NOT_SUPPORTED")
    if verdict != stats["verdict"]:
        fail(f"mechanical verdict mismatch: {verdict} != {stats['verdict']}")
    if within["ar1_phi030"] >= 0 or within["nonlinear_ar1_phi085"] >= 0:
        fail("known scenario-level prediction failures disappeared from generated report")
    report = (RESULTS / "gate2_report.md").read_text()
    for marker in (verdict, "aggregate coefficient includes between-process", "not uniform agreement", "Stage 2 stops here"):
        if marker not in report:
            fail(f"report lacks honest required marker: {marker}")
    return verdict


def check_artifacts_metadata_provenance(metadata: dict, release_audit: bool = False) -> None:
    for relative, expected in metadata["artifact_sha256"].items():
        if sha(ROOT / relative) != expected:
            fail(f"artifact hash mismatch: {relative}")
    for required in (
        ROOT / "figures/synth/fig_theory_predictions_vs_measured.pdf",
        ROOT / "figures/synth/fig_theory_predictions_vs_measured.png",
        RESULTS / "theory_vs_measured_curves.csv",
    ):
        if not required.exists() or required.stat().st_size == 0:
            fail(f"missing artifact: {required}")
    forbidden = (ROOT / "results/eeg/shot_sensitivity.csv", ROOT / "docs/physical_resources.md", ROOT / "paper/manuscript.tex")
    if not release_audit and any(path.exists() for path in forbidden):
        fail("Stage 3/4 artifact exists; mandatory Stage 2 stop violated")
    if subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode:
        fail("pytest failed")
    if subprocess.run([sys.executable, "scripts/make_provenance.py"], cwd=ROOT).returncode:
        fail("provenance generation failed")
    checksums = (ROOT / "provenance/eeg_checksums.txt").read_text()
    for relative in (
        "results/synth/gate2_report.md", "results/synth/theory_predictions_vs_measured.csv",
        "results/synth/stage2_metadata.json", "figures/synth/fig_theory_predictions_vs_measured.pdf",
        "docs/synthetic_stage2_protocol.md", "scripts/run_synthetic_stage2.py",
        "tests/test_synthetic_stage2.py",
    ):
        if relative not in checksums:
            fail(f"SHA256 missing: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-audit", action="store_true",
                        help="verify frozen Gate 2 evidence after Gate 3/release artifacts exist")
    args = parser.parse_args()
    cfg, metadata = check_freezes()
    combined = check_rows_and_slopes(cfg)
    verdict = check_verdict(combined, metadata)
    check_artifacts_metadata_provenance(metadata, args.release_audit)
    stats = metadata["statistics"]
    print(f"\nROTA A GATE 2 VERIFICATION: PASS; MECHANICAL VERDICT: {verdict}")
    print(f"aggregate Spearman={stats['aggregate_spearman']:.6f} "
          f"CI=[{stats['aggregate_spearman_ci_low']:.6f}, {stats['aggregate_spearman_ci_high']:.6f}]")
    print(f"within-scenario median={stats['median_within_scenario_spearman']:.3f}; "
          f"best-model match={stats['best_model_match_fraction']:.3f}")
    print("negative ordering failures retained: ar1_phi030, nonlinear_ar1_phi085")
    print("release audit: later-stage artifacts allowed" if args.release_audit else
          "Stage 3/4 artifacts: absent; stopped for human review")


if __name__ == "__main__":
    main()
