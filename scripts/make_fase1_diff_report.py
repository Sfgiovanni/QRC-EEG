#!/usr/bin/env python3
"""Generate the phase-1 before/after report, including ridge-alpha changes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "eeg"
BEFORE = RESULTS / "_prefix_snapshot_hpselect"


def signed(value: float) -> str:
    return f"{float(value):+.4f}"


def endpoint_lines() -> list[str]:
    before = pd.read_csv(BEFORE / "tab_eeg_endpoints.csv")
    after = pd.read_csv(RESULTS / "tab_eeg_endpoints.csv")
    rows = before.merge(after, on=["construction", "set"], suffixes=("_before", "_after"), validate="one_to_one")
    out = [
        "## 1. Endpoints",
        "",
        "| Construction | Set | NRMSE before | NRMSE after | Difference | R2 before | R2 after | Difference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.sort_values(["construction", "set"]).iterrows():
        out.append(
            f"| {row.construction} | {row['set']} | {row.nrmse_mean_before:.4f} | {row.nrmse_mean_after:.4f} | "
            f"{signed(row.nrmse_mean_after-row.nrmse_mean_before)} | {row.r2_mean_before:.4f} | "
            f"{row.r2_mean_after:.4f} | {signed(row.r2_mean_after-row.r2_mean_before)} |"
        )
    return out


def contrast_lines(before_file: str, after_file: str, comparison: str, sets: list[str], heading: str):
    before = pd.read_csv(BEFORE / before_file)
    after = pd.read_csv(RESULTS / after_file)
    keys = ["comparison", "set", "horizon"]
    before = before[(before.comparison == comparison) & before["set"].isin(sets)]
    after = after[(after.comparison == comparison) & after["set"].isin(sets)]
    rows = before.merge(after, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    out = [
        heading,
        "",
        "Positive Delta-NRMSE means lower NRMSE for the kernel.",
        "",
        "| Set | h | Delta before | CI before | Holm before | Delta after | CI after | Holm after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.sort_values(["set", "horizon"]).iterrows():
        out.append(
            f"| {row['set']} | {int(row.horizon)} | {signed(row.mean_diff_rmse_comparator_minus_state_before)} | "
            f"[{signed(row.ci95_lo_before)}, {signed(row.ci95_hi_before)}] | {row.p_holm_before:.4g} | "
            f"{signed(row.mean_diff_rmse_comparator_minus_state_after)} | "
            f"[{signed(row.ci95_lo_after)}, {signed(row.ci95_hi_after)}] | {row.p_holm_after:.4g} |"
        )
    return out, rows


def alpha_distribution(values: pd.Series) -> str:
    counts = values.value_counts().sort_index()
    return ", ".join(f"{alpha:g} ({count})" for alpha, count in counts.items())


def alpha_lines():
    before = pd.read_csv(BEFORE / "selected_alphas.csv")
    after = pd.concat(
        [pd.read_csv(RESULTS / "selected_alphas.csv"), pd.read_csv(RESULTS / "selected_alphas_esn66.csv")],
        ignore_index=True,
    )
    keys = ["construction", "set", "horizon", "seed"]
    matched = before.merge(after, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    out = [
        "## 4. Selected ridge alpha",
        "",
        f"Exact matched choices changed in {(matched.alpha_before != matched.alpha_after).sum()}/{len(matched)} construction/set/horizon/seed cells.",
        "",
        "Counts below pool the four horizons and ten seeds within each construction/set (40 choices per row).",
        "",
        "| Construction | Set | Before alpha (count) | After alpha (count) | Changed cells |",
        "|---|---:|---|---|---:|",
    ]
    for (construction, set_name), group in matched.groupby(["construction", "set"], sort=True):
        out.append(
            f"| {construction} | {set_name} | {alpha_distribution(group.alpha_before)} | "
            f"{alpha_distribution(group.alpha_after)} | {(group.alpha_before != group.alpha_after).sum()}/40 |"
        )
    return out, matched


def verdict_lines(esn: pd.DataFrame, ab: pd.DataFrame) -> list[str]:
    delta, p = "mean_diff_rmse_comparator_minus_state_after", "p_holm_after"
    short = esn[esn.horizon.isin([1, 2])]
    long = esn[esn.horizon.isin([4, 8])]
    expected = int((short[delta] < 0).sum() + (long[delta] > 0).sum())
    short_sig = int(((short[delta] < 0) & (short[p] < 0.05)).sum())
    long_sig = int(((long[delta] > 0) & (long[p] < 0.05)).sum())
    interaction = "SOBREVIVEU" if expected == 8 and short_sig >= 2 and long_sig >= 2 else "ENFRAQUECEU" if expected >= 6 else "SUMIU"
    ab_positive = int((ab[delta] > 0).sum())
    ab_sig = int(((ab[delta] > 0) & (ab[p] < 0.05)).sum())
    ab_verdict = "SOBREVIVEU" if ab_positive >= 9 and ab_sig >= 6 else "ENFRAQUECEU" if ab_positive >= 7 else "SUMIU"
    return [
        "## 5. Factual verdict",
        "",
        f"- **Kernel x ESN horizon interaction — {interaction}.** Expected direction in {expected}/8 F/Z cells; Holm-significant expected-direction cells: short={short_sig}, long={long_sig}.",
        f"- **Kernel > AB — {ab_verdict}.** Kernel-favoring cells={ab_positive}/12; Holm-significant kernel-favoring cells={ab_sig}/12.",
    ]


def main() -> None:
    required = [BEFORE / "selected_alphas.csv", BEFORE / "tab_eeg_endpoints.csv", BEFORE / "tab_esn_matched.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"hpselect prefix snapshot incomplete: {missing}")
    esn_text, esn = contrast_lines(
        "tab_esn_matched.csv", "tab_esn_matched.csv", "single_kernel vs ESN_66", ["F", "Z"],
        "## 2. Kernel vs ESN-66 by horizon (F and Z)",
    )
    ab_text, ab = contrast_lines(
        "tab_eeg_contrasts.csv", "tab_eeg_contrasts.csv", "single_kernel vs AB_noaux", ["F", "S", "Z"],
        "## 3. Kernel vs AB by horizon",
    )
    alpha_text, _ = alpha_lines()
    verdict = verdict_lines(esn, ab)
    text = [
        "# Phase 1: segment-blocked ridge selection diff",
        "",
        "Before is the post-causal-normalization snapshot; after uses whole disjoint segments for ridge/HP validation.",
        "",
        *endpoint_lines(), "", *esn_text, "", *ab_text, "", *alpha_text, "", *verdict, "",
    ]
    out = RESULTS / "fase1_diff_report.md"
    out.write_text("\n".join(text))
    print(f"wrote {out}")
    for line in verdict[2:]:
        print(line)


if __name__ == "__main__":
    main()
