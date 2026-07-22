#!/usr/bin/env python3
"""HP search (Analysis B, "retuned core") for the classical distributed-memory
ESN control (docs/classical_distributed_memory_protocol.md).

For each of ESN66_K0 / ESN66_AB / ESN66_kernel, the ESN core HP
(spectral_radius x input_scale x leak_rate) is selected independently from
the exact grid already frozen for the ESN in
config/eeg_frozen.yaml:hp_grids.ESN (n_reservoir fixed to 66, mirroring the
already-logged deviation in scripts/run_esn66_hp_search.py), using the same
hp_search_seeds and hp_search_subsample as every other HP search in this
repository, train/validation only. The test partition is never read here.
The memory-kernel HP (kernel_hp below) is frozen and never searched.

Also writes classical_control/fixed_core_hp.json, documenting (not
searching) Analysis A's reused ESN-66 core HP.

Writes:
  results/eeg/followup/classical_control/hp_search_log.csv
  results/eeg/followup/classical_control/hp_selected.json   (retuned_core)
  results/eeg/followup/classical_control/fixed_core_hp.json (fixed_core)
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
from qrc_eeg.esn_distributed_memory import CONSTRUCTIONS, construction_features  # noqa: E402
from qrc_eeg.pipeline import fit_readouts_per_horizon, evaluate_segments  # noqa: E402
from qrc_eeg.preprocessing import scale_set_from_training  # noqa: E402
from qrc_eeg.splits import load_splits  # noqa: E402

CONFIG_PATH = ROOT / "config" / "esn_distributed_memory_frozen.yaml"
EEG_CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
RESULTS_DIR = ROOT / "results" / "eeg" / "followup" / "classical_control"
N_RESERVOIR = 66


def main() -> None:
    fcfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg = yaml.safe_load(EEG_CONFIG_PATH.read_text())
    sets = cfg["data"]["sets"]
    horizons = cfg["readout"]["horizons"]
    alpha_grid = cfg["readout"]["alpha_grid"]
    washout = cfg["readout"]["washout"]

    raw = {name: load_set(ROOT / "data" / "eeg" / "sets" / name) for name in sets}
    splits = load_splits(ROOT / "data" / "eeg" / "splits", sets)  # frozen splits, not regenerated
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

    base_grid = cfg["hp_grids"]["ESN"]
    assert base_grid["n_reservoir"] == [200], "frozen ESN grid changed unexpectedly -- update this script's assumption"
    combos = [
        {"n_reservoir": N_RESERVOIR, "spectral_radius": sr, "input_scale": in_scale, "leak_rate": lr}
        for sr in base_grid["spectral_radius"]
        for in_scale in base_grid["input_scale"]
        for lr in base_grid["leak_rate"]
    ]

    def mean_val_nrmse(construction: str, kernel_hp: dict, esn_hp: dict, seed: int) -> float:
        feats_train = construction_features(construction, kernel_hp, esn_hp, seed=seed, segments=pooled_train)
        feats_val = construction_features(construction, kernel_hp, esn_hp, seed=seed, segments=pooled_val)
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

    log_rows = []
    selected = {}
    for construction in CONSTRUCTIONS:
        kernel_hp = fcfg["constructions"][construction]["kernel_hp"]
        best_hp, best_score = None, np.inf
        for esn_hp in combos:
            scores = []
            t0 = time.perf_counter()
            for seed in cfg["channel"]["hp_search_seeds"]:
                scores.append(mean_val_nrmse(construction, kernel_hp, esn_hp, seed))
            elapsed = time.perf_counter() - t0
            mean_score = float(np.mean(scores))
            log_rows.append({
                "construction": construction, "esn_hp": json.dumps(esn_hp),
                "kernel_hp": json.dumps(kernel_hp), "mean_val_nrmse": mean_score, "seconds": elapsed,
            })
            print(f"{construction} {esn_hp} -> val NRMSE {mean_score:.4f} ({elapsed:.1f}s)", flush=True)
            if mean_score < best_score:
                best_hp, best_score = esn_hp, mean_score
        selected[construction] = {"esn_hp": best_hp, "kernel_hp": kernel_hp, "val_nrmse": best_score}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "hp_search_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["construction", "esn_hp", "kernel_hp", "mean_val_nrmse", "seconds"])
        w.writeheader()
        w.writerows(log_rows)
    (RESULTS_DIR / "hp_selected.json").write_text(json.dumps(selected, indent=2))
    print("wrote", RESULTS_DIR / "hp_search_log.csv")
    print("wrote", RESULTS_DIR / "hp_selected.json")

    # Analysis A (fixed core): document, do not search, the reused ESN-66 core HP.
    esn66_selected = json.loads((ROOT / "results/eeg/hp_selected_esn66.json").read_text())
    fixed_core_hp = esn66_selected["ESN_66"]["hp"]
    fixed_core = {
        construction: {"esn_hp": fixed_core_hp, "kernel_hp": fcfg["constructions"][construction]["kernel_hp"]}
        for construction in CONSTRUCTIONS
    }
    fixed_core["source"] = "HEAD:results/eeg/hp_selected_esn66.json:ESN_66.hp, reused unmodified for all three arms"
    (RESULTS_DIR / "fixed_core_hp.json").write_text(json.dumps(fixed_core, indent=2))
    print("wrote", RESULTS_DIR / "fixed_core_hp.json")


if __name__ == "__main__":
    main()
