#!/usr/bin/env python3
"""Long-horizon (h=4, h=8) paired contrasts, family 'eeg_long'.

h=4/h=8 metrics already exist in results/eeg/raw/eeg_holdout_by_segment_seed.csv
(the held-out grid covers all four frozen horizons) -- this script does not
re-run anything for single_kernel/AB_noaux/ESN-200, it only extracts and
re-summarizes. It does NOT reuse eeg_primary's stored p_holm values, because
that column was Holm-corrected across a 36-test family; here the family is
redefined to the 8 tests below (2 comparisons x 2 horizons x 3 sets is 12
raw p-values... actually 2 comparisons x 2 horizons x 3 sets = 12), so Holm
must be recomputed from the underlying p_wilcoxon within this subfamily.

Reports two comparisons:
  - single_kernel vs AB_noaux (clean, no dimension confound)
  - single_kernel vs ESN_66   (the valid matched-dimension comparison; requires
    results/eeg/raw/eeg_holdout_esn66_by_segment_seed.csv from Item 2)
ESN-200 (dimension-unmatched) is included as a labeled-unmatched reference
column only, not as part of the eeg_long hypothesis family.

Writes results/eeg/tab_long_horizon_contrasts.csv, paper/tab_long_horizon_contrasts.tex.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.statistics import holm, paired_patient_summary  # noqa: E402

CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
RESULTS_DIR = ROOT / "results" / "eeg"

LONG_HORIZONS = [4, 8]
CONTRASTS = [
    ("single_kernel", "AB_noaux"),
    ("single_kernel", "ESN_66"),
]


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    primary = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_by_segment_seed.csv")
    esn66_path = RESULTS_DIR / "raw" / "eeg_holdout_esn66_by_segment_seed.csv"
    if not esn66_path.exists():
        raise FileNotFoundError(
            f"{esn66_path} missing -- run scripts/run_esn66_holdout.py (Item 2) before Item 1's "
            "single_kernel vs ESN_66 long-horizon extraction, the ESN-66 comparison depends on it"
        )
    esn66 = pd.read_csv(esn66_path)
    df = pd.concat([primary, esn66], ignore_index=True)

    per_segment = df.groupby(["construction", "set", "horizon", "segment_id"], as_index=False)["nrmse"].mean()

    rows = []
    for set_name in cfg["data"]["sets"]:
        for horizon in LONG_HORIZONS:
            slab = per_segment[(per_segment["set"] == set_name) & (per_segment["horizon"] == horizon)]
            for state_name, comparator_name in CONTRASTS:
                a = slab[slab["construction"] == state_name].set_index("segment_id")["nrmse"]
                b = slab[slab["construction"] == comparator_name].set_index("segment_id")["nrmse"]
                if a.empty or b.empty:
                    continue
                summary = paired_patient_summary(a, b, state_name, comparator_name, seed=cfg["split"]["seed"])
                summary["set"] = set_name
                summary["horizon"] = horizon
                rows.append(summary)

    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result["p_wilcoxon"].to_numpy())  # fresh, family eeg_long (12 tests)
    result = result.sort_values(["set", "horizon", "comparison"]).reset_index(drop=True)

    # ESN-200 unmatched reference (mean NRMSE per set/horizon), transparency only, not hypothesis-tested here.
    esn200_ref = (
        primary[(primary["construction"] == "ESN") & (primary["horizon"].isin(LONG_HORIZONS))]
        .groupby(["set", "horizon"])["nrmse"]
        .mean()
        .rename("esn200_nrmse_mean_unmatched_reference_only")
        .reset_index()
    )
    result = result.merge(esn200_ref, on=["set", "horizon"], how="left")

    out_path = RESULTS_DIR / "tab_long_horizon_contrasts.csv"
    result.to_csv(out_path, index=False)
    print("wrote", out_path, f"({len(result)} rows)")
    print(
        result[
            ["comparison", "set", "horizon", "mean_diff_rmse_comparator_minus_state", "ci95_lo", "ci95_hi", "p_holm", "win_fraction_state"]
        ]
    )

    PAPER_DIR = ROOT / "paper"
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["comparison", "set", "horizon", "mean_diff_rmse_comparator_minus_state", "ci95_lo", "ci95_hi", "p_holm", "cohen_dz", "win_fraction_state"]
    tex_df = result[cols].round(4)
    lines = [
        r"\begin{table}",
        r"\caption{Long-horizon ($h=4,8$) paired contrasts, family eeg\_long, Holm-corrected within this 12-test family.}",
        r"\label{tab:long_horizon_contrasts}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        " & ".join(cols) + r" \\",
    ]
    for _, row in tex_df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    (PAPER_DIR / "tab_long_horizon_contrasts.tex").write_text("\n".join(lines))
    print("wrote", PAPER_DIR / "tab_long_horizon_contrasts.tex")


if __name__ == "__main__":
    main()
