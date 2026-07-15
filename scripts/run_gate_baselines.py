#!/usr/bin/env python3
"""Run preregistered persistence, AR, NVAR2 and matched tapped-delay controls."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.classical_baselines import (  # noqa: E402
    diagonal_nvar2,
    evaluate_feature_model,
    evaluate_persistence,
    lag_features,
    select_ridge_blocked,
    tapped_delay_features,
)
from qrc_eeg.eeg_data import load_set  # noqa: E402
from qrc_eeg.preprocessing import scale_set_from_training  # noqa: E402
from qrc_eeg.pipeline import assert_disjoint_segment_ids  # noqa: E402
from qrc_eeg.splits import load_splits  # noqa: E402
from qrc_eeg.state_kernels import single_exponential_weights  # noqa: E402

CONFIG = ROOT / "config" / "eeg_frozen.yaml"
RESULTS = ROOT / "results" / "eeg"
RAW = RESULTS / "raw"
SPLITS = ROOT / "data" / "eeg" / "splits"


def append_metrics(rows, model, set_name, horizon, test_ids, metrics, seeds):
    for seed in seeds:
        for i, segment_id in enumerate(test_ids):
            rows.append({
                "construction": model, "set": set_name, "horizon": horizon, "seed": seed,
                "segment_id": segment_id, **{name: values[i] for name, values in metrics.items()},
            })


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    sets = cfg["data"]["sets"]
    splits = load_splits(SPLITS, sets)
    raw = {name: load_set(ROOT / "data" / "eeg" / "sets" / name) for name in sets}
    scaled = {name: scale_set_from_training(raw[name], splits[name]["train"])[0] for name in sets}
    selected_kernel = json.loads((RESULTS / "hp_selected.json").read_text())["single_kernel"]["hp"]
    kernel = single_exponential_weights(**selected_kernel)
    horizons = cfg["readout"]["horizons"]
    alpha_grid = cfg["readout"]["alpha_grid"]
    washout = cfg["readout"]["washout"]
    p_grid = cfg["eeg_gate"]["ar_lags"]
    seeds = cfg["channel"]["confirmatory_seeds"]
    rows, hp_rows = [], []

    for set_name in sets:
        t0 = time.perf_counter()
        train_ids, val_ids, test_ids = (splits[set_name][part] for part in ("train", "val", "test"))
        assert_disjoint_segment_ids(train_ids, val_ids)
        if set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
            raise RuntimeError(f"segment leakage into classical test partition for set {set_name}")
        train = np.stack([scaled[set_name][sid] for sid in train_ids])
        val = np.stack([scaled[set_name][sid] for sid in val_ids])
        test = np.stack([scaled[set_name][sid] for sid in test_ids])
        train_lags = lag_features(train, max(p_grid))
        val_lags = lag_features(val, max(p_grid))
        test_lags = lag_features(test, max(p_grid))

        for horizon in horizons:
            persistence = evaluate_persistence(test, horizon, washout)
            append_metrics(rows, "persistence", set_name, horizon, test_ids, persistence, seeds)

            candidates = []
            for p in p_grid:
                fit = select_ridge_blocked(
                    train_lags[:, :, :p], train, val_lags[:, :, :p], val,
                    horizon, washout, alpha_grid,
                )
                candidates.append((fit.validation_nrmse, p, fit))
            _, best_p, ar_fit = min(candidates, key=lambda item: item[0])
            ar_metrics = evaluate_feature_model(test_lags[:, :, :best_p], test, horizon, washout, ar_fit.weights)
            append_metrics(rows, "AR", set_name, horizon, test_ids, ar_metrics, seeds)
            hp_rows.append({"construction": "AR", "set": set_name, "horizon": horizon,
                            "p": best_p, "alpha": ar_fit.alpha, "validation_nrmse": ar_fit.validation_nrmse})

            train_nvar = diagonal_nvar2(train_lags[:, :, :best_p])
            val_nvar = diagonal_nvar2(val_lags[:, :, :best_p])
            test_nvar = diagonal_nvar2(test_lags[:, :, :best_p])
            nvar_fit = select_ridge_blocked(train_nvar, train, val_nvar, val, horizon, washout, alpha_grid)
            nvar_metrics = evaluate_feature_model(test_nvar, test, horizon, washout, nvar_fit.weights)
            append_metrics(rows, "NVAR2", set_name, horizon, test_ids, nvar_metrics, seeds)
            hp_rows.append({"construction": "NVAR2", "set": set_name, "horizon": horizon,
                            "p": best_p, "alpha": nvar_fit.alpha, "validation_nrmse": nvar_fit.validation_nrmse})
            del train_nvar, val_nvar, test_nvar

            train_tapped = tapped_delay_features(train, kernel.present, kernel.delayed)
            val_tapped = tapped_delay_features(val, kernel.present, kernel.delayed)
            test_tapped = tapped_delay_features(test, kernel.present, kernel.delayed)
            tapped_fit = select_ridge_blocked(
                train_tapped, train, val_tapped, val, horizon, washout, alpha_grid,
            )
            tapped_metrics = evaluate_feature_model(test_tapped, test, horizon, washout, tapped_fit.weights)
            append_metrics(rows, "tapped_delay", set_name, horizon, test_ids, tapped_metrics, seeds)
            hp_rows.append({"construction": "tapped_delay", "set": set_name, "horizon": horizon,
                            "p": len(kernel.delayed) + 1, "alpha": tapped_fit.alpha,
                            "validation_nrmse": tapped_fit.validation_nrmse})
            print(f"gate baseline set={set_name} h={horizon}: AR(p={best_p})/NVAR2/tapped complete", flush=True)

        print(f"gate baselines set={set_name}: {time.perf_counter() - t0:.1f}s", flush=True)

    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / "eeg_gate_classical_by_segment_seed.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "construction", "set", "horizon", "seed", "segment_id", "nrmse", "rmse", "r2", "mae",
        ])
        writer.writeheader(); writer.writerows(rows)
    hp_path = RESULTS / "gate_classical_selected_hp.csv"
    with hp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "construction", "set", "horizon", "p", "alpha", "validation_nrmse",
        ])
        writer.writeheader(); writer.writerows(hp_rows)
    print(f"wrote {raw_path} ({len(rows)} rows)")
    print(f"wrote {hp_path} ({len(hp_rows)} rows)")


if __name__ == "__main__":
    main()
