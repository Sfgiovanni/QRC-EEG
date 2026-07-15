#!/usr/bin/env python3
"""Fail-high verifier for Rota A Gate 0; performs no Stage 1 work."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/eeg"
FROZEN_GATE_SHA = "9f1d7717c4061ee31ee19e0dc9f666a27d1c7ec9a7c743bba56a8f8820c79247"


def fail(message: str) -> None:
    raise SystemExit(f"ROTA A GATE 0 FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_lo(values: np.ndarray, seed: int, n_boot: int) -> float:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5))


def check_frozen_gate() -> None:
    report = RESULTS / "gate_report.md"
    if sha256(report) != FROZEN_GATE_SHA:
        fail("results/eeg/gate_report.md changed after the empirical gate")
    manifest = (RESULTS / "gate_report_frozen.sha256").read_text()
    if FROZEN_GATE_SHA not in manifest:
        fail("frozen gate hash manifest is inconsistent")


def check_useful_horizon_v2() -> None:
    cfg = yaml.safe_load((ROOT / "config/eeg_frozen.yaml").read_text())
    n_boot = int(cfg["statistics"]["bootstrap_resamples"])
    seed = int(cfg["split"]["seed"])
    horizons = cfg["readout"]["horizons"]
    frames = [
        pd.read_csv(RESULTS / "raw/eeg_holdout_by_segment_seed.csv"),
        pd.read_csv(RESULTS / "raw/eeg_holdout_esn66_by_segment_seed.csv"),
        pd.read_csv(RESULTS / "raw/eeg_gate_classical_by_segment_seed.csv"),
    ]
    raw = pd.concat(frames, ignore_index=True)
    per_segment = raw.groupby(["construction", "set", "horizon", "segment_id"], as_index=False)["nrmse"].mean()
    pivot = per_segment.pivot_table(index=["set", "horizon", "segment_id"], columns="construction", values="nrmse")
    out = pd.read_csv(RESULTS / "useful_horizon_v2.csv")
    expected_rows = per_segment["construction"].nunique() * len(cfg["data"]["sets"])
    if len(out) != expected_rows or out.duplicated(["construction", "set"]).any():
        fail("v2 table does not contain exactly one row per model/set")
    for row in out.itertuples(index=False):
        qualifying = []
        evidence = {}
        for horizon in horizons:
            slab = pivot.loc[(row.set, horizon)].dropna(subset=[row.construction, "persistence"])
            model = slab[row.construction].to_numpy(dtype=float)
            improvement = (slab["persistence"] - slab[row.construction]).to_numpy(dtype=float)
            lo = bootstrap_lo(improvement, seed + 101 * horizon, n_boot)
            item = (float(model.mean()), float(improvement.mean()), lo)
            evidence[horizon] = item
            if item[0] < 1.0 and lo > 0.0:
                qualifying.append(horizon)
        expected = max(qualifying) if qualifying else np.nan
        if (pd.isna(expected) and not pd.isna(row.useful_horizon)) or (
            not pd.isna(expected) and int(row.useful_horizon) != int(expected)
        ):
            fail(f"incorrect useful horizon for {row.construction}/{row.set}: {row.useful_horizon} vs {expected}")
        if not pd.isna(expected):
            mean_nrmse, improvement_mean, lo = evidence[int(expected)]
            if not np.isclose(row.nrmse_at_useful_horizon, mean_nrmse) or not np.isclose(row.persistence_improvement_mean, improvement_mean) or not np.isclose(row.persistence_improvement_ci95_lo, lo):
                fail(f"stored useful-horizon evidence differs for {row.construction}/{row.set}")
    persistence = out[out["construction"] == "persistence"]
    if persistence["useful_horizon"].notna().any():
        fail("persistence must be NA under strict improvement over itself")
    ar = out[out["construction"] == "AR"]
    if ar["useful_horizon"].isna().any():
        fail("AR is still artificially NA under a criterion that should be symmetric")


def check_text_and_commands() -> None:
    text = (ROOT / "RESULTS.md").read_text()
    required = [
        "no claim of quantum advantage", "classical models lead at short horizons",
        "S is null in the causal test", "distributed-memory class",
        "all evaluated models have mean NRMSE above 1 at h=64", "one EEG database",
        "useful_horizon_v2.csv",
    ]
    for marker in required:
        if marker not in text:
            fail(f"RESULTS.md lacks required honest framing: {marker}")
    for stale in ("R^2 = 0.92", "R^2 = 0.97", "[-0.0117, 0.0013]", "improvement over both persistence and AR"):
        if stale in text:
            fail(f"RESULTS.md contains stale text: {stale}")
    curves = pd.read_csv(RESULTS / "gate_nrmse_curves.csv")
    if not (curves[curves["horizon"] == 64]["mean_nrmse"] > 1.0).all():
        fail("claim that all h=64 endpoints lack absolute skill is false")
    script = ROOT / "scripts/run_rotaA_stage0.sh"
    result = subprocess.run(["bash", "-n", str(script)], cwd=ROOT)
    if result.returncode or "verify_rotaA_gate0.py" not in script.read_text().splitlines()[-1]:
        fail("single Stage 0 command is invalid or does not stop at Gate 0")


def tests_and_provenance() -> None:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    if result.returncode:
        fail("pytest failed")
    result = subprocess.run([sys.executable, "scripts/make_provenance.py"], cwd=ROOT)
    if result.returncode:
        fail("provenance generation failed")
    checksums = (ROOT / "provenance/eeg_checksums.txt").read_text()
    for relative in (
        "docs/rotaA_plan.md", "docs/PROVENANCE.md", "results/eeg/useful_horizon_v2.csv",
        "results/eeg/gate_report_frozen.sha256", "scripts/make_useful_horizon_v2.py",
        "scripts/run_rotaA_stage0.sh", "scripts/verify_rotaA_gate0.py",
    ):
        if relative not in checksums:
            fail(f"SHA256 missing for new artifact: {relative}")


def main() -> None:
    check_frozen_gate()
    check_useful_horizon_v2()
    check_text_and_commands()
    tests_and_provenance()
    print("\nROTA A GATE 0: PASS")
    print("useful_horizon_v2: symmetric persistence comparison independently reproduced")
    print("gate_report.md: frozen SHA256 unchanged")
    print("RESULTS.md: CSV-consistent honest framing; historical Gate 0 artifact set preserved")
    print(f"{sha256(RESULTS / 'useful_horizon_v2.csv')}  results/eeg/useful_horizon_v2.csv")


if __name__ == "__main__":
    main()
