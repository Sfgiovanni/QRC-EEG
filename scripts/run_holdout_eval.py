#!/usr/bin/env python3
"""Held-out evaluation: frozen HPs, full train+val refit, test-only evaluation,
10 confirmatory seeds, every set x horizon x construction.

Writes results/eeg/raw/eeg_holdout_by_segment_seed.csv.
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
from qrc_eeg.pipeline import construction_features, fit_readouts_per_horizon, evaluate_segments_full  # noqa: E402
from qrc_eeg.splits import load_splits  # noqa: E402
from qrc_eeg.tasks import zscore  # noqa: E402

CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
SPLITS_DIR = ROOT / "data" / "eeg" / "splits"
RESULTS_DIR = ROOT / "results" / "eeg"
RAW_DIR = RESULTS_DIR / "raw"


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    sets = cfg["data"]["sets"]
    horizons = cfg["readout"]["horizons"]
    alpha_grid = cfg["readout"]["alpha_grid"]
    washout = cfg["readout"]["washout"]
    selected = json.loads((RESULTS_DIR / "hp_selected.json").read_text())

    raw = {name: load_set(ROOT / "data" / "eeg" / "sets" / name) for name in sets}
    zscored = {name: {sid: zscore(np.array(v))[0] for sid, v in segs.items()} for name, segs in raw.items()}
    splits = load_splits(SPLITS_DIR, sets)

    rows = []
    for construction, choice in selected.items():
        hp = choice["hp"]
        for set_name in sets:
            trainval_ids = splits[set_name]["train"] + splits[set_name]["val"]
            test_ids = splits[set_name]["test"]
            trainval_arr = np.stack([zscored[set_name][i] for i in trainval_ids])
            test_arr = np.stack([zscored[set_name][i] for i in test_ids])

            for seed in cfg["channel"]["confirmatory_seeds"]:
                t0 = time.perf_counter()
                feats_trainval = construction_features(construction, hp, seed=seed, segments=trainval_arr)
                feats_test = construction_features(construction, hp, seed=seed, segments=test_arr)
                fits = fit_readouts_per_horizon(feats_trainval, trainval_arr, horizons, alpha_grid, washout=washout)
                results = evaluate_segments_full(feats_test, test_arr, fits, washout=washout)
                elapsed = time.perf_counter() - t0
                for h, metrics in results.items():
                    for i, seg_id in enumerate(test_ids):
                        rows.append(
                            {
                                "construction": construction,
                                "set": set_name,
                                "horizon": h,
                                "seed": seed,
                                "segment_id": seg_id,
                                "nrmse": metrics["nrmse"][i],
                                "rmse": metrics["rmse"][i],
                                "r2": metrics["r2"][i],
                                "mae": metrics["mae"][i],
                            }
                        )
                print(
                    f"{construction} set={set_name} seed={seed}: {elapsed:.1f}s "
                    f"mean_nrmse={np.nanmean([np.nanmean(v['nrmse']) for v in results.values()]):.4f}",
                    flush=True,
                )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "eeg_holdout_by_segment_seed.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["construction", "set", "horizon", "seed", "segment_id", "nrmse", "rmse", "r2", "mae"]
        )
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_path, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
