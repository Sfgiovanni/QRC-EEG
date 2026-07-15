#!/usr/bin/env python3
"""Symmetric useful horizon: skill < 1 and bootstrap improvement over persistence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "eeg"
CONFIG = ROOT / "config" / "eeg_frozen.yaml"


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["split"]["seed"])
    n_boot = int(cfg["statistics"]["bootstrap_resamples"])
    fs = float(cfg["data"]["sampling_rate_hz"])
    frames = [
        pd.read_csv(RESULTS / "raw/eeg_holdout_by_segment_seed.csv"),
        pd.read_csv(RESULTS / "raw/eeg_holdout_esn66_by_segment_seed.csv"),
        pd.read_csv(RESULTS / "raw/eeg_gate_classical_by_segment_seed.csv"),
    ]
    raw = pd.concat(frames, ignore_index=True)
    per_segment = raw.groupby(
        ["construction", "set", "horizon", "segment_id"], as_index=False
    )["nrmse"].mean()
    pivot = per_segment.pivot_table(
        index=["set", "horizon", "segment_id"], columns="construction", values="nrmse"
    )

    rows = []
    for set_name in cfg["data"]["sets"]:
        for model in sorted(per_segment["construction"].unique()):
            evidence = []
            for horizon in cfg["readout"]["horizons"]:
                slab = pivot.loc[(set_name, horizon)].dropna(subset=[model, "persistence"])
                model_values = slab[model].to_numpy(dtype=float)
                improvement = (slab["persistence"] - slab[model]).to_numpy(dtype=float)
                lo, hi = bootstrap_ci(improvement, seed + 101 * horizon, n_boot)
                mean_nrmse = float(model_values.mean())
                qualifies = mean_nrmse < 1.0 and lo > 0.0
                evidence.append((horizon, mean_nrmse, float(improvement.mean()), lo, hi, qualifies))
            qualified = [item for item in evidence if item[-1]]
            if qualified:
                chosen = max(qualified, key=lambda item: item[0])
                h, mean_nrmse, improvement_mean, ci_lo, ci_hi, _ = chosen
            else:
                h = mean_nrmse = improvement_mean = ci_lo = ci_hi = np.nan
            rows.append({
                "construction": model,
                "set": set_name,
                "useful_horizon": h,
                "useful_horizon_ms": 1000.0 * h / fs if np.isfinite(h) else np.nan,
                "nrmse_at_useful_horizon": mean_nrmse,
                "persistence_improvement_mean": improvement_mean,
                "persistence_improvement_ci95_lo": ci_lo,
                "persistence_improvement_ci95_hi": ci_hi,
                "bootstrap_resamples": n_boot,
                "criterion": "mean NRMSE < 1 and paired-bootstrap lower CI(persistence - model) > 0",
            })

    out = pd.DataFrame(rows).sort_values(["set", "construction"]).reset_index(drop=True)
    out.to_csv(RESULTS / "useful_horizon_v2.csv", index=False)
    print(f"wrote {RESULTS / 'useful_horizon_v2.csv'} ({len(out)} rows)")
    print(out[["construction", "set", "useful_horizon", "useful_horizon_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
