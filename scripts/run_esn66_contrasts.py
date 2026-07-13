#!/usr/bin/env python3
"""Paired contrasts, family 'eeg_esn_matched': single_kernel vs ESN-66 (the
dimension-equalized comparison, 66 readout features on both sides), per set
x horizon. Reuses qrc_eeg.statistics exactly as scripts/run_statistics.py
does for the eeg_primary family; Holm correction is recomputed fresh within
this 12-test subfamily (not extracted from eeg_primary's p_holm).

Keeps the original ESN-200 endpoint (nrmse mean/sem per set x horizon,
dimension-unmatched) alongside for transparency.

Writes results/eeg/tab_esn_matched.csv, paper/tab_esn_matched.tex.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.statistics import holm, paired_segment_summary  # noqa: E402

CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
RESULTS_DIR = ROOT / "results" / "eeg"


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    primary = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_by_segment_seed.csv")
    esn66 = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_esn66_by_segment_seed.csv")
    df = pd.concat([primary, esn66], ignore_index=True)

    per_segment = df.groupby(["construction", "set", "horizon", "segment_id"], as_index=False)["nrmse"].mean()

    rows = []
    for set_name in cfg["data"]["sets"]:
        for horizon in cfg["readout"]["horizons"]:
            slab = per_segment[(per_segment["set"] == set_name) & (per_segment["horizon"] == horizon)]
            a = slab[slab["construction"] == "single_kernel"].set_index("segment_id")["nrmse"]
            b = slab[slab["construction"] == "ESN_66"].set_index("segment_id")["nrmse"]
            if a.empty or b.empty:
                continue
            summary = paired_segment_summary(a, b, "single_kernel", "ESN_66", seed=cfg["split"]["seed"])
            summary["set"] = set_name
            summary["horizon"] = horizon
            rows.append(summary)

    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result["p_wilcoxon"].to_numpy())  # fresh Holm, family eeg_esn_matched (12 tests)
    result = result.sort_values(["set", "horizon"]).reset_index(drop=True)

    # ESN-200 endpoints alongside, labeled unmatched, for transparency (per set x horizon).
    esn200_endpoint = (
        primary[primary["construction"] == "ESN"]
        .groupby(["set", "horizon"])["nrmse"]
        .agg(esn200_nrmse_mean_unmatched="mean", esn200_nrmse_sem_unmatched=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))))
        .reset_index()
    )
    esn66_endpoint = (
        esn66.groupby(["set", "horizon"])["nrmse"]
        .agg(esn66_nrmse_mean_matched="mean", esn66_nrmse_sem_matched=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))))
        .reset_index()
    )
    result = result.merge(esn200_endpoint, on=["set", "horizon"], how="left")
    result = result.merge(esn66_endpoint, on=["set", "horizon"], how="left")

    out_path = RESULTS_DIR / "tab_esn_matched.csv"
    result.to_csv(out_path, index=False)
    print("wrote", out_path, f"({len(result)} rows)")
    print(
        result[
            [
                "set",
                "horizon",
                "mean_diff_rmse_comparator_minus_state",
                "ci95_lo",
                "ci95_hi",
                "p_holm",
                "win_fraction_state",
                "esn66_nrmse_mean_matched",
                "esn200_nrmse_mean_unmatched",
            ]
        ]
    )

    PAPER_DIR = ROOT / "paper"
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "set",
        "horizon",
        "mean_diff_rmse_comparator_minus_state",
        "ci95_lo",
        "ci95_hi",
        "p_holm",
        "cohen_dz",
        "win_fraction_state",
        "esn66_nrmse_mean_matched",
        "esn200_nrmse_mean_unmatched",
    ]
    tex_df = result[cols].round(4)
    lines = [
        r"\begin{table}",
        r"\caption{Single-kernel vs ESN-66 (dimension-matched, 66 readout features both sides), "
        r"family eeg\_esn\_matched, Holm-corrected within this 12-test family. "
        r"ESN-200 columns (unmatched, 200 readout features) shown for transparency only.}",
        r"\label{tab:esn_matched}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        " & ".join(cols) + r" \\",
    ]
    for _, row in tex_df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    (PAPER_DIR / "tab_esn_matched.tex").write_text("\n".join(lines))
    print("wrote", PAPER_DIR / "tab_esn_matched.tex")


if __name__ == "__main__":
    main()
