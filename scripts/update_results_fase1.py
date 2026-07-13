#!/usr/bin/env python3
"""Write RESULTS.md exclusively from the regenerated phase-1 CSVs."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "eeg"


def main() -> None:
    matched = pd.read_csv(RESULTS / "tab_esn_matched.csv")
    ab = pd.read_csv(RESULTS / "tab_eeg_contrasts.csv")
    ab = ab[ab.comparison == "single_kernel vs AB_noaux"]
    cap = pd.read_csv(RESULTS / "quadratic_capacity.csv").set_index("construction")
    regression = pd.read_csv(RESULTS / "capacity_demand_regression_summary.csv")
    cap_reg = regression[regression.predictor == "capacity_gap"].iloc[0]

    fz = matched[matched["set"].isin(["F", "Z"])]
    short = fz[fz.horizon.isin([1, 2])]
    long = fz[fz.horizon.isin([4, 8])]
    expected = int((short.mean_diff_rmse_comparator_minus_state < 0).sum() + (long.mean_diff_rmse_comparator_minus_state > 0).sum())
    short_sig = int(((short.mean_diff_rmse_comparator_minus_state < 0) & (short.p_holm < 0.05)).sum())
    long_sig = int(((long.mean_diff_rmse_comparator_minus_state > 0) & (long.p_holm < 0.05)).sum())
    ab_positive = int((ab.mean_diff_rmse_comparator_minus_state > 0).sum())
    ab_sig = int(((ab.mean_diff_rmse_comparator_minus_state > 0) & (ab.p_holm < 0.05)).sum())

    lines = [
        "# EEG Results",
        "",
        "These results use causal training-only normalization and segment-blocked ridge/HP validation. "
        "All numeric statements below are regenerated from the result CSVs.",
        "",
        "## Bottom line",
        "",
        "Performance depends on forecast horizon. In the matched 66-feature comparison on F and Z, "
        f"the expected ESN-short/kernel-long direction occurs in {expected}/8 cells; Holm-significant "
        f"cells in that direction are {short_sig} at short horizons and {long_sig} at long horizons. "
        "This is a horizon interaction, not a claim of overall superiority or generic competitiveness.",
        "",
        f"Against AB-noaux, the kernel has lower NRMSE in {ab_positive}/12 set-by-horizon cells, with "
        f"{ab_sig}/12 Holm-significant kernel-favoring cells.",
        "",
        "The quadratic iid measure used here showed no detectable positive association with EEG gains "
        f"in the evaluated configurations. The descriptive comparison-level slope is {cap_reg.slope:+.6f} "
        f"with n={int(cap_reg['n'])} independent capacity gaps; no inferential confidence interval is reported.",
        "",
        "Corrected quadratic-capacity means are: " + ", ".join(
            f"{name}={cap.loc[name, 'quadratic_capacity_mean']:.4f}" for name in cap.index
        ) + ".",
        "",
        "## Scope limitation",
        "",
        "The statistical unit is the held-out segment, not the patient. The Bonn distribution randomized "
        "and withheld the segment-to-subject mapping, so there is no subject-separated evaluation and no "
        "claim of generalization between patients. Z is surface EEG; F and S are intracranial EEG. "
        "Inference is limited to benchmark segments.",
        "",
        "## Outputs",
        "",
        "- `results/eeg/tab_eeg_endpoints.csv`",
        "- `results/eeg/tab_eeg_contrasts.csv`",
        "- `results/eeg/tab_esn_matched.csv`",
        "- `results/eeg/quadratic_capacity.csv`",
        "- `results/eeg/fase1_diff_report.md`",
        "",
        "## Reproduction",
        "",
        "```bash",
        "bash scripts/run_eeg.sh",
        "```",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote RESULTS.md from regenerated CSVs")


if __name__ == "__main__":
    main()
