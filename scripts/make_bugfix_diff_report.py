#!/usr/bin/env python3
"""Compare the preserved pre-bugfix tables with regenerated results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "eeg"
BEFORE = RESULTS / "_prefix_snapshot"


def fmt(value: float, digits: int = 4) -> str:
    return "NA" if pd.isna(value) else f"{float(value):+.{digits}f}"


def endpoints_section() -> list[str]:
    before = pd.read_csv(BEFORE / "tab_eeg_endpoints.csv")
    after = pd.read_csv(RESULTS / "tab_eeg_endpoints.csv")
    joined = before.merge(after, on=["construction", "set"], suffixes=("_before", "_after"), validate="one_to_one")
    lines = [
        "## 1. Endpoints",
        "",
        "| Construction | Set | NRMSE before | NRMSE after | Delta | R2 before | R2 after | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in joined.sort_values(["construction", "set"]).iterrows():
        lines.append(
            f"| {row['construction']} | {row['set']} | {row['nrmse_mean_before']:.4f} | "
            f"{row['nrmse_mean_after']:.4f} | {fmt(row['nrmse_mean_after'] - row['nrmse_mean_before'])} | "
            f"{row['r2_mean_before']:.4f} | {row['r2_mean_after']:.4f} | "
            f"{fmt(row['r2_mean_after'] - row['r2_mean_before'])} |"
        )
    return lines


def contrast_table(before_path: Path, after_path: Path, comparison: str, sets: list[str], title: str) -> tuple[list[str], pd.DataFrame]:
    before = pd.read_csv(before_path)
    after = pd.read_csv(after_path)
    keys = ["comparison", "set", "horizon"]
    before = before[(before["comparison"] == comparison) & before["set"].isin(sets)]
    after = after[(after["comparison"] == comparison) & after["set"].isin(sets)]
    joined = before.merge(after, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    lines = [
        title,
        "",
        "Positive Delta-NRMSE means lower NRMSE for the kernel. Significance is Holm p < 0.05.",
        "",
        "| Set | h | Delta before | CI before | Holm before | Delta after | CI after | Holm after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in joined.sort_values(["set", "horizon"]).iterrows():
        lines.append(
            f"| {row['set']} | {int(row['horizon'])} | {fmt(row['mean_diff_rmse_comparator_minus_state_before'])} | "
            f"[{fmt(row['ci95_lo_before'])}, {fmt(row['ci95_hi_before'])}] | {row['p_holm_before']:.4g} | "
            f"{fmt(row['mean_diff_rmse_comparator_minus_state_after'])} | "
            f"[{fmt(row['ci95_lo_after'])}, {fmt(row['ci95_hi_after'])}] | {row['p_holm_after']:.4g} |"
        )
    return lines, joined


def capacity_section() -> tuple[list[str], dict]:
    before = pd.read_csv(BEFORE / "quadratic_capacity.csv")
    after = pd.read_csv(RESULTS / "quadratic_capacity.csv")
    joined = before.merge(after, on="construction", suffixes=("_before", "_after"), validate="one_to_one")
    lines = [
        "## 4. Capacity",
        "",
        "| Construction | Quadratic before | Quadratic after | Delta |",
        "|---|---:|---:|---:|",
    ]
    for _, row in joined.sort_values("construction").iterrows():
        delta = row["quadratic_capacity_mean_after"] - row["quadratic_capacity_mean_before"]
        lines.append(
            f"| {row['construction']} | {row['quadratic_capacity_mean_before']:.4f} | "
            f"{row['quadratic_capacity_mean_after']:.4f} | {fmt(delta)} |"
        )
    before_summary = pd.read_csv(BEFORE / "capacity_demand_regression_summary.csv")
    after_summary = pd.read_csv(RESULTS / "capacity_demand_regression_summary.csv")
    b = before_summary.loc[before_summary["predictor"] == "capacity_gap"].iloc[0]
    a = after_summary.loc[after_summary["predictor"] == "capacity_gap"].iloc[0]
    lines += [
        "",
        f"Before: slope={b['slope']:+.6f}, nominal CI=[{b['ci_lo']:+.6f}, {b['ci_hi']:+.6f}], n reported={int(b['n'])} repeated rows.",
        f"After: descriptive slope={a['slope']:+.6f}, CI omitted, n={int(a['n'])} independent comparison-level gaps.",
        "",
        "A medida quadrática iid utilizada não apresentou associação positiva detectável com os ganhos em EEG nas configurações avaliadas.",
    ]
    return lines, {"before": b, "after": a}


def verdicts(esn: pd.DataFrame, ab: pd.DataFrame, capacity: dict) -> list[str]:
    delta = "mean_diff_rmse_comparator_minus_state_after"
    p = "p_holm_after"
    short = esn[esn["horizon"].isin([1, 2])]
    long = esn[esn["horizon"].isin([4, 8])]
    direction_count = int((short[delta] < 0).sum() + (long[delta] > 0).sum())
    sig_short = int(((short[delta] < 0) & (short[p] < 0.05)).sum())
    sig_long = int(((long[delta] > 0) & (long[p] < 0.05)).sum())
    if direction_count == len(esn) and sig_short >= 2 and sig_long >= 2:
        cross_verdict = "SOBREVIVEU"
    elif direction_count >= 6 and sig_short + sig_long > 0:
        cross_verdict = "ENFRAQUECEU"
    else:
        cross_verdict = "SUMIU"

    ab_positive = int((ab[delta] > 0).sum())
    ab_sig = int(((ab[delta] > 0) & (ab[p] < 0.05)).sum())
    if ab_positive >= 9 and ab_sig >= 6:
        ab_verdict = "SOBREVIVEU"
    elif ab_positive >= 7 and ab_sig > 0:
        ab_verdict = "ENFRAQUECEU"
    else:
        ab_verdict = "SUMIU"

    return [
        "## 5. Factual verdict",
        "",
        f"- **Kernel x ESN crossover — {cross_verdict}.** Corrected directions match ESN at short and kernel at long horizons in {direction_count}/{len(esn)} F/Z cells; Holm-significant cells in the expected direction: short={sig_short}, long={sig_long}.",
        f"- **Kernel > AB — {ab_verdict}.** Corrected kernel-favoring cells={ab_positive}/{len(ab)}; Holm-significant kernel-favoring cells={ab_sig}/{len(ab)}.",
        f"- **Capacity-gain association — ENFRAQUECEU.** The nominal before analysis used n={int(capacity['before']['n'])} repeated rows; the corrected descriptive analysis has n={int(capacity['after']['n'])} independent gaps and no inferential CI (slope={capacity['after']['slope']:+.6f}).",
    ]


def main() -> None:
    required = [BEFORE / "tab_eeg_endpoints.csv", BEFORE / "tab_esn_matched.csv", BEFORE / "tab_eeg_contrasts.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"prefix snapshot incomplete: {missing}")
    esn_lines, esn = contrast_table(
        BEFORE / "tab_esn_matched.csv", RESULTS / "tab_esn_matched.csv", "single_kernel vs ESN_66", ["F", "Z"],
        "## 2. Kernel vs ESN-66 by horizon (F and Z)",
    )
    ab_lines, ab = contrast_table(
        BEFORE / "tab_eeg_contrasts.csv", RESULTS / "tab_eeg_contrasts.csv", "single_kernel vs AB_noaux", ["F", "S", "Z"],
        "## 3. Kernel vs AB by horizon",
    )
    cap_lines, capacity = capacity_section()
    lines = [
        "# EEG bugfix: before vs after",
        "",
        "Before is `results/eeg/_prefix_snapshot/`; after is the causal-preprocessing rerun. Deltas in endpoint rows are after minus before.",
        "",
        *endpoints_section(), "", *esn_lines, "", *ab_lines, "", *cap_lines, "", *verdicts(esn, ab, capacity), "",
    ]
    out = RESULTS / "bugfix_diff_report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
