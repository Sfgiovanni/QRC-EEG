#!/usr/bin/env python3
"""Held-out evaluation for the classical distributed-memory ESN control,
both analysis modes (docs/classical_distributed_memory_protocol.md).

Frozen split, full train+val refit, test-only evaluation, 10 confirmatory
seeds, every construction x set x horizon x analysis_mode. Mirrors
scripts/run_esn66_holdout.py, extended to three constructions and two
analysis modes. Run once, after HP selection (both modes) is frozen.

Writes:
  results/eeg/followup/raw/esn_distributed_memory_holdout_by_segment_seed.csv
  results/eeg/followup/classical_control/selected_alphas.csv
  results/eeg/followup/classical_control/tab_resource_accounting.csv
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
from qrc_eeg.esn_distributed_memory import CONSTRUCTIONS, construction_features, resource_accounting  # noqa: E402
from qrc_eeg.pipeline import fit_readouts_per_horizon, evaluate_segments_full  # noqa: E402
from qrc_eeg.preprocessing import scale_set_from_training  # noqa: E402
from qrc_eeg.splits import load_splits  # noqa: E402

CONFIG_PATH = ROOT / "config" / "esn_distributed_memory_frozen.yaml"
EEG_CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
FOLLOWUP_DIR = ROOT / "results" / "eeg" / "followup"
RAW_DIR = FOLLOWUP_DIR / "raw"
CONTROL_DIR = FOLLOWUP_DIR / "classical_control"
N_RESERVOIR = 66


def main() -> None:
    fcfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg = yaml.safe_load(EEG_CONFIG_PATH.read_text())
    sets = cfg["data"]["sets"]
    horizons = cfg["readout"]["horizons"]
    alpha_grid = cfg["readout"]["alpha_grid"]
    washout = cfg["readout"]["washout"]
    seeds = cfg["channel"]["confirmatory_seeds"]

    fixed_core = json.loads((CONTROL_DIR / "fixed_core_hp.json").read_text())
    retuned_core = json.loads((CONTROL_DIR / "hp_selected.json").read_text())

    modes = {
        "fixed_core": {c: fixed_core[c]["esn_hp"] for c in CONSTRUCTIONS},
        "retuned_core": {c: retuned_core[c]["esn_hp"] for c in CONSTRUCTIONS},
    }

    splits = load_splits(ROOT / "data" / "eeg" / "splits", sets)
    raw = {name: load_set(ROOT / "data" / "eeg" / "sets" / name) for name in sets}
    scaled = {name: scale_set_from_training(raw[name], splits[name]["train"])[0] for name in sets}

    rows, alpha_rows = [], []
    for construction in CONSTRUCTIONS:
        kernel_hp = fcfg["constructions"][construction]["kernel_hp"]
        for mode_name, mode_hp in modes.items():
            esn_hp = dict(mode_hp[construction], n_reservoir=N_RESERVOIR)
            for set_name in sets:
                train_ids, val_ids, test_ids = splits[set_name]["train"], splits[set_name]["val"], splits[set_name]["test"]
                train_arr = np.stack([scaled[set_name][i] for i in train_ids])
                val_arr = np.stack([scaled[set_name][i] for i in val_ids])
                test_arr = np.stack([scaled[set_name][i] for i in test_ids])

                for seed in seeds:
                    t0 = time.perf_counter()
                    feats_train = construction_features(construction, kernel_hp, esn_hp, seed=seed, segments=train_arr)
                    feats_val = construction_features(construction, kernel_hp, esn_hp, seed=seed, segments=val_arr)
                    feats_test = construction_features(construction, kernel_hp, esn_hp, seed=seed, segments=test_arr)
                    fits = fit_readouts_per_horizon(
                        feats_train, train_arr, horizons, alpha_grid, washout=washout,
                        validation_features=feats_val, validation_segments=val_arr,
                        train_segment_ids=train_ids, validation_segment_ids=val_ids,
                    )
                    results = evaluate_segments_full(feats_test, test_arr, fits, washout=washout)
                    elapsed = time.perf_counter() - t0
                    for h, metrics in results.items():
                        alpha_rows.append({
                            "construction": construction, "analysis_mode": mode_name, "set": set_name,
                            "horizon": h, "seed": seed, "alpha": fits[h].alpha,
                            "selection_scheme": "blocked_segments_train_val",
                        })
                        for i, seg_id in enumerate(test_ids):
                            rows.append({
                                "construction": construction, "analysis_mode": mode_name, "set": set_name,
                                "horizon": h, "seed": seed, "segment_id": seg_id,
                                "nrmse": metrics["nrmse"][i], "rmse": metrics["rmse"][i],
                                "r2": metrics["r2"][i], "mae": metrics["mae"][i],
                            })
                    print(
                        f"{construction}/{mode_name} set={set_name} seed={seed}: {elapsed:.1f}s "
                        f"mean_nrmse={np.nanmean([np.nanmean(v['nrmse']) for v in results.values()]):.4f}",
                        flush=True,
                    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "esn_distributed_memory_holdout_by_segment_seed.csv"
    fieldnames = ["construction", "analysis_mode", "set", "horizon", "seed", "segment_id", "nrmse", "rmse", "r2", "mae"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_path, f"({len(rows)} rows)")

    n_cells = len({(r["construction"], r["analysis_mode"], r["set"], r["horizon"], r["seed"]) for r in rows})
    expected = len(CONSTRUCTIONS) * len(modes) * len(sets) * len(horizons) * len(seeds)
    if n_cells != expected:
        raise RuntimeError(f"incomplete distributed-memory ESN grid: {n_cells} of {expected} cells")

    alpha_path = CONTROL_DIR / "selected_alphas.csv"
    with open(alpha_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["construction", "analysis_mode", "set", "horizon", "seed", "alpha", "selection_scheme"])
        w.writeheader()
        w.writerows(alpha_rows)
    print("wrote", alpha_path, f"({len(alpha_rows)} rows)")

    resource_rows = [resource_accounting(c, fcfg["constructions"][c]["kernel_hp"], N_RESERVOIR) for c in CONSTRUCTIONS]
    resource_path = CONTROL_DIR / "tab_resource_accounting.csv"
    with open(resource_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resource_rows[0].keys()))
        w.writeheader()
        w.writerows(resource_rows)
    print("wrote", resource_path)


if __name__ == "__main__":
    main()
