#!/usr/bin/env python3
"""HP search: train -> validation only, pooled across sets/horizons.

Writes results/eeg/hp_search_log.csv (every combo tried) and
results/eeg/hp_selected.json (winner per construction).
"""

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

from qrc_eeg.eeg_data import load_set  # noqa: E402
from qrc_eeg.pipeline import construction_features, fit_readouts_per_horizon, evaluate_segments  # noqa: E402
from qrc_eeg.preprocessing import scale_set_from_training  # noqa: E402
from qrc_eeg.splits import build_and_save_splits, load_splits  # noqa: E402

CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
SPLITS_DIR = ROOT / "data" / "eeg" / "splits"
RESULTS_DIR = ROOT / "results" / "eeg"


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    sets = cfg["data"]["sets"]
    horizons = cfg["readout"]["horizons"]
    alpha_grid = cfg["readout"]["alpha_grid"]
    washout = cfg["readout"]["washout"]

    raw = {name: load_set(ROOT / "data" / "eeg" / "sets" / name) for name in sets}
    ids_by_set = {name: list(segs.keys()) for name, segs in raw.items()}
    build_and_save_splits(
        ids_by_set,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        seed=cfg["split"]["seed"],
        out_dir=SPLITS_DIR,
    )
    splits = load_splits(SPLITS_DIR, sets)
    scaled = {name: scale_set_from_training(raw[name], splits[name]["train"])[0] for name in sets}

    sub = cfg["hp_search_subsample"]
    n_train_sub = sub["train_segments_per_set"]
    n_val_sub = sub["val_segments_per_set"]

    train_arr, val_arr, train_ids_by_set, val_ids_by_set = {}, {}, {}, {}
    for name in sets:
        train_ids = splits[name]["train"][:n_train_sub]
        val_ids = splits[name]["val"][:n_val_sub]
        train_ids_by_set[name] = train_ids
        val_ids_by_set[name] = val_ids
        train_arr[name] = np.stack([scaled[name][i] for i in train_ids])
        val_arr[name] = np.stack([scaled[name][i] for i in val_ids])

    pooled_train = np.concatenate([train_arr[n] for n in sets], axis=0)
    pooled_val = np.concatenate([val_arr[n] for n in sets], axis=0)
    pooled_train_ids = [sid for name in sets for sid in train_ids_by_set[name]]
    pooled_val_ids = [sid for name in sets for sid in val_ids_by_set[name]]

    from qrc_eeg.tasks import forecast_target

    def mean_val_nrmse(construction: str, hp: dict, seed: int) -> float:
        feats_train = construction_features(construction, hp, seed=seed, segments=pooled_train)
        feats_val = construction_features(construction, hp, seed=seed, segments=pooled_val)
        fits = fit_readouts_per_horizon(
            feats_train,
            pooled_train,
            horizons,
            alpha_grid,
            washout=washout,
            validation_features=feats_val,
            validation_segments=pooled_val,
            train_segment_ids=pooled_train_ids,
            validation_segment_ids=pooled_val_ids,
            refit_on_train_validation=False,
        )
        results = evaluate_segments(feats_val, pooled_val, fits, washout=washout)
        return float(np.nanmean([np.nanmean(v) for v in results.values()]))

    from qrc_eeg.pipeline import hp_grid_combinations

    log_rows = []
    selected = {}
    for construction, grid in cfg["hp_grids"].items():
        combos = grid if isinstance(grid, list) else hp_grid_combinations(grid)
        best_hp, best_score = None, np.inf
        for hp in combos:
            scores = []
            t0 = time.perf_counter()
            for seed in cfg["channel"]["hp_search_seeds"]:
                scores.append(mean_val_nrmse(construction, hp, seed))
            elapsed = time.perf_counter() - t0
            mean_score = float(np.mean(scores))
            log_rows.append(
                {"construction": construction, "hp": json.dumps(hp), "mean_val_nrmse": mean_score, "seconds": elapsed}
            )
            print(f"{construction} {hp} -> val NRMSE {mean_score:.4f} ({elapsed:.1f}s)", flush=True)
            if mean_score < best_score:
                best_hp, best_score = hp, mean_score
        selected[construction] = {"hp": best_hp, "val_nrmse": best_score}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "hp_search_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["construction", "hp", "mean_val_nrmse", "seconds"])
        w.writeheader()
        w.writerows(log_rows)
    (RESULTS_DIR / "hp_selected.json").write_text(json.dumps(selected, indent=2))
    print("wrote", RESULTS_DIR / "hp_search_log.csv")
    print("wrote", RESULTS_DIR / "hp_selected.json")


if __name__ == "__main__":
    main()
