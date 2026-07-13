#!/usr/bin/env python3
"""fig_long_horizon (APS style, PDF + 600dpi PNG, colorblind-safe palette --
reuses src/qrc_eeg/style/aps.mplstyle exactly as scripts/run_tables_figures.py
does).

Writes figures/eeg/fig_long_horizon.{pdf,png}.
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

# Okabe-Ito colorblind-safe palette.
COLOR_AB = "#0072B2"
COLOR_ESN = "#D55E00"


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


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use(str(STYLE_PATH))
    fig_long_horizon()


if __name__ == "__main__":
    main()
