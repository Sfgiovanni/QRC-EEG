#!/usr/bin/env python3
"""fig_long_horizon and fig_ictal_auroc (APS style, PDF + 600dpi PNG,
colorblind-safe palette -- reuses src/qrc_eeg/style/aps.mplstyle exactly as
scripts/run_tables_figures.py does).

Writes figures/eeg/fig_long_horizon.{pdf,png}, figures/eeg/fig_ictal_auroc.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "results" / "eeg"
FIGURES_DIR = ROOT / "figures" / "eeg"
STYLE_PATH = ROOT / "src" / "qrc_eeg" / "style" / "aps.mplstyle"

# Okabe-Ito colorblind-safe palette (consistent choice across both figures).
COLOR_AB = "#0072B2"
COLOR_ESN = "#D55E00"
COLOR_KERNEL = "#009E73"
COLOR_OTHER = "#CC79A7"

CONSTRUCTION_COLORS = {
    "single_kernel": COLOR_KERNEL,
    "dual_kernel": COLOR_OTHER,
    "AB_noaux": COLOR_AB,
    "ESN_66": COLOR_ESN,
}
CONSTRUCTION_LABELS = {
    "single_kernel": "Single kernel",
    "dual_kernel": "Dual kernel",
    "AB_noaux": "AB (noaux)",
    "ESN_66": "ESN-66 (matched)",
}


def fig_long_horizon() -> None:
    df = pd.read_csv(RESULTS_DIR / "tab_long_horizon_contrasts.csv")
    sets = ["Z", "F", "S"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    markers = {"single_kernel vs AB_noaux": ("s", COLOR_AB), "single_kernel vs ESN_66": ("^", COLOR_ESN)}
    for ax, set_name in zip(axes, sets):
        sub_set = df[df["set"] == set_name]
        for comparison, (marker, color) in markers.items():
            sub = sub_set[sub_set["comparison"] == comparison].sort_values("horizon")
            if sub.empty:
                continue
            x = sub["horizon"].to_numpy()
            y = sub["mean_diff_rmse_comparator_minus_state"].to_numpy()
            yerr_lo = y - sub["ci95_lo"].to_numpy()
            yerr_hi = sub["ci95_hi"].to_numpy() - y
            label = comparison.replace("single_kernel vs ", "vs ").replace("ESN_66", "ESN-66")
            ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], marker=marker, color=color, linestyle="-", label=label, capsize=2)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks([4, 8])
        ax.set_xlabel("Horizon $h$")
        ax.set_title(set_name)
    axes[0].set_ylabel(r"$\Delta$NRMSE (comparator $-$ kernel)")
    axes[-1].legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_long_horizon.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig_long_horizon (pdf, png)")


def fig_ictal_auroc() -> None:
    df = pd.read_csv(RESULTS_DIR / "tab_ictal_classification.csv")
    order = [c for c in CONSTRUCTION_LABELS if c in df["construction"].values]
    df = df.set_index("construction").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y_pos = range(len(df))
    colors = [CONSTRUCTION_COLORS[c] for c in df["construction"]]
    xerr_lo = df["auroc"] - df["auroc_ci_lo"]
    xerr_hi = df["auroc_ci_hi"] - df["auroc"]
    ax.barh(y_pos, df["auroc"], color=colors, alpha=0.35, height=0.6)
    ax.errorbar(df["auroc"], y_pos, xerr=[xerr_lo, xerr_hi], fmt="o", color="black", capsize=3, markersize=4)
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", label="chance")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([CONSTRUCTION_LABELS[c] for c in df["construction"]])
    ax.set_xlabel("AUROC (ictal vs non-ictal)")
    ax.set_xlim(0.0, 1.05)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_ictal_auroc.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig_ictal_auroc (pdf, png)")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use(str(STYLE_PATH))
    fig_long_horizon()
    fig_ictal_auroc()


if __name__ == "__main__":
    main()
