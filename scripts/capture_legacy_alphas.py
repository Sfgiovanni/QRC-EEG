#!/usr/bin/env python3
"""Reconstruct pre-fase1 row-random ridge alphas for the preserved snapshot.

This script must be run before replacing ``fit_readouts_per_horizon``. It
reuses the post-normalization configuration, frozen splits and selected HPs,
but computes no test metrics and overwrites no current result table.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.eeg_data import load_set  # noqa: E402
from qrc_eeg.pipeline import HorizonFit, construction_features  # noqa: E402
from qrc_eeg.preprocessing import scale_set_from_training  # noqa: E402
from qrc_eeg.readout import fit_readout, predict_readout  # noqa: E402
from qrc_eeg.splits import load_splits  # noqa: E402
from qrc_eeg.tasks import forecast_target  # noqa: E402

CONFIG = ROOT / "config" / "eeg_frozen.yaml"
RESULTS = ROOT / "results" / "eeg"
SNAPSHOT = RESULTS / "_prefix_snapshot_hpselect"
SPLITS_DIR = ROOT / "data" / "eeg" / "splits"


def legacy_row_random_fits(features, segments, horizons, alpha_grid, washout):
    """Exact quarantined pre-fase1 selector, retained only for the snapshot."""

    fits = {}
    for horizon in horizons:
        target = np.stack([forecast_target(segment, horizon) for segment in segments])
        end = features.shape[1] - horizon
        x = features[:, washout:end, :].reshape(-1, features.shape[2])
        y = target[:, washout:end].reshape(-1)
        valid = ~np.isnan(y)
        x, y = x[valid], y[valid]
        order = np.random.default_rng(0).permutation(len(x))
        cut = max(1, int(0.8 * len(order)))
        train_rows, validation_rows = order[:cut], order[cut:]
        best_alpha, best_error = alpha_grid[0], np.inf
        for alpha in alpha_grid:
            weights = fit_readout(x[train_rows], y[train_rows], alpha=alpha)
            prediction = predict_readout(x[validation_rows], weights)
            error = float(np.sqrt(np.mean((prediction - y[validation_rows]) ** 2)))
            if error < best_error:
                best_alpha, best_error = alpha, error
        fits[horizon] = HorizonFit(
            horizon=horizon,
            alpha=float(best_alpha),
            weights=fit_readout(x, y, alpha=best_alpha),
        )
    return fits


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    sets = cfg["data"]["sets"]
    splits = load_splits(SPLITS_DIR, sets)
    raw = {name: load_set(ROOT / "data" / "eeg" / "sets" / name) for name in sets}
    scaled = {name: scale_set_from_training(raw[name], splits[name]["train"])[0] for name in sets}
    selections = json.loads((RESULTS / "hp_selected.json").read_text())
    esn66 = json.loads((RESULTS / "hp_selected_esn66.json").read_text())["ESN_66"]
    choices = [(name, choice["hp"], name) for name, choice in selections.items()]
    choices.append(("ESN", esn66["hp"], "ESN_66"))

    rows = []
    for feature_name, hp, output_name in choices:
        for set_name in sets:
            trainval_ids = splits[set_name]["train"] + splits[set_name]["val"]
            trainval = np.stack([scaled[set_name][sid] for sid in trainval_ids])
            for seed in cfg["channel"]["confirmatory_seeds"]:
                features = construction_features(feature_name, hp, seed=seed, segments=trainval)
                fits = legacy_row_random_fits(
                    features,
                    trainval,
                    cfg["readout"]["horizons"],
                    cfg["readout"]["alpha_grid"],
                    cfg["readout"]["washout"],
                )
                for horizon, fit in fits.items():
                    rows.append(
                        {
                            "construction": output_name,
                            "set": set_name,
                            "horizon": horizon,
                            "seed": seed,
                            "alpha": fit.alpha,
                            "selection_scheme": "random_temporal_rows_80_20",
                        }
                    )
                print(f"legacy alpha: {output_name} set={set_name} seed={seed}", flush=True)

    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    out = SNAPSHOT / "selected_alphas.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["construction", "set", "horizon", "seed", "alpha", "selection_scheme"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
