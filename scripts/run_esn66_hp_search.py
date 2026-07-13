#!/usr/bin/env python3
"""HP search for the dimension-matched ESN control (n_reservoir=66, exactly
the readout feature count of every quantum arm).

Post-freeze deviation from `config/eeg_frozen.yaml` (logged in
`results/eeg/PROVENANCE.md`): the frozen ESN grid only searches
n_reservoir=200. Here n_reservoir is fixed to 66 -- the independent variable
of this follow-up experiment, not a tuned knob -- and the *same*
spectral_radius x leak_rate grid values already in the frozen config are
reused verbatim, with the same hp_search_seeds and hp_search_subsample.
Mirrors scripts/run_hp_search.py's selection logic exactly; does not
reimplement it.

Writes results/eeg/hp_search_log_esn66.csv, results/eeg/hp_selected_esn66.json.
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
from qrc_eeg.splits import load_splits  # noqa: E402
from qrc_eeg.tasks import zscore  # noqa: E402

CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
SPLITS_DIR = ROOT / "data" / "eeg" / "splits"
RESULTS_DIR = ROOT / "results" / "eeg"

N_RESERVOIR_MATCHED = 66


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    sets = cfg["data"]["sets"]
    horizons = cfg["readout"]["horizons"]
    alpha_grid = cfg["readout"]["alpha_grid"]
    washout = cfg["readout"]["washout"]

    raw = {name: load_set(ROOT / "data" / "eeg" / "sets" / name) for name in sets}
    zscored = {name: {sid: zscore(np.array(v))[0] for sid, v in segs.items()} for name, segs in raw.items()}
    splits = load_splits(SPLITS_DIR, sets)  # frozen splits, not regenerated

    sub = cfg["hp_search_subsample"]
    n_train_sub = sub["train_segments_per_set"]
    n_val_sub = sub["val_segments_per_set"]

    train_arr, val_arr = {}, {}
    for name in sets:
        train_ids = splits[name]["train"][:n_train_sub]
        val_ids = splits[name]["val"][:n_val_sub]
        train_arr[name] = np.stack([zscored[name][i] for i in train_ids])
        val_arr[name] = np.stack([zscored[name][i] for i in val_ids])

    pooled_train = np.concatenate([train_arr[n] for n in sets], axis=0)
    pooled_val = np.concatenate([val_arr[n] for n in sets], axis=0)

    def mean_val_nrmse(hp: dict, seed: int) -> float:
        feats_train = construction_features("ESN", hp, seed=seed, segments=pooled_train)
        feats_val = construction_features("ESN", hp, seed=seed, segments=pooled_val)
        fits = fit_readouts_per_horizon(feats_train, pooled_train, horizons, alpha_grid, washout=washout)
        results = evaluate_segments(feats_val, pooled_val, fits, washout=washout)
        return float(np.nanmean([np.nanmean(v) for v in results.values()]))

    base_grid = cfg["hp_grids"]["ESN"]
    assert base_grid["n_reservoir"] == [200], "frozen ESN grid changed unexpectedly -- update this script's assumption"
    combos = [
        {
            "n_reservoir": N_RESERVOIR_MATCHED,
            "spectral_radius": sr,
            "input_scale": in_scale,
            "leak_rate": lr,
        }
        for sr in base_grid["spectral_radius"]
        for in_scale in base_grid["input_scale"]
        for lr in base_grid["leak_rate"]
    ]

    log_rows = []
    best_hp, best_score = None, np.inf
    for hp in combos:
        scores = []
        t0 = time.perf_counter()
        for seed in cfg["channel"]["hp_search_seeds"]:
            scores.append(mean_val_nrmse(hp, seed))
        elapsed = time.perf_counter() - t0
        mean_score = float(np.mean(scores))
        log_rows.append({"construction": "ESN_66", "hp": json.dumps(hp), "mean_val_nrmse": mean_score, "seconds": elapsed})
        print(f"ESN_66 {hp} -> val NRMSE {mean_score:.4f} ({elapsed:.1f}s)", flush=True)
        if mean_score < best_score:
            best_hp, best_score = hp, mean_score

    selected = {"ESN_66": {"hp": best_hp, "val_nrmse": best_score}}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "hp_search_log_esn66.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["construction", "hp", "mean_val_nrmse", "seconds"])
        w.writeheader()
        w.writerows(log_rows)
    (RESULTS_DIR / "hp_selected_esn66.json").write_text(json.dumps(selected, indent=2))
    print("wrote", RESULTS_DIR / "hp_search_log_esn66.csv")
    print("wrote", RESULTS_DIR / "hp_selected_esn66.json")


if __name__ == "__main__":
    main()
