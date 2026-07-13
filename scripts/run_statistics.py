#!/usr/bin/env python3
"""Paired contrasts (family 'eeg_primary'): kernel-single vs AB_noaux/ESN,
dual vs single, per set and horizon. Seeds aggregated per segment first.

Writes results/eeg/tab_eeg_contrasts.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.statistics import holm, paired_segment_summary  # noqa: E402

CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
RESULTS_DIR = ROOT / "results" / "eeg"

CONTRASTS = [
    ("single_kernel", "AB_noaux"),
    ("single_kernel", "ESN"),
    ("dual_kernel", "single_kernel"),
]


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    df = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_by_segment_seed.csv")

    per_segment = df.groupby(["construction", "set", "horizon", "segment_id"], as_index=False)["nrmse"].mean()

    rows = []
    for set_name in cfg["data"]["sets"]:
        for horizon in cfg["readout"]["horizons"]:
            slab = per_segment[(per_segment["set"] == set_name) & (per_segment["horizon"] == horizon)]
            for state_name, comparator_name in CONTRASTS:
                a = slab[slab["construction"] == state_name].set_index("segment_id")["nrmse"]
                b = slab[slab["construction"] == comparator_name].set_index("segment_id")["nrmse"]
                if a.empty or b.empty:
                    continue
                summary = paired_segment_summary(a, b, state_name, comparator_name, seed=cfg["split"]["seed"])
                summary["set"] = set_name
                summary["horizon"] = horizon
                rows.append(summary)

    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result["p_wilcoxon"].to_numpy())
    result = result.sort_values(["set", "horizon", "comparison"]).reset_index(drop=True)

    out_path = RESULTS_DIR / "tab_eeg_contrasts.csv"
    result.to_csv(out_path, index=False)
    print("wrote", out_path, f"({len(result)} rows)")
    print(result[["comparison", "set", "horizon", "mean_diff_rmse_comparator_minus_state", "p_holm", "win_fraction_state"]])


if __name__ == "__main__":
    main()
