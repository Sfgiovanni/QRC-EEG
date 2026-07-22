#!/usr/bin/env python3
"""Forest-plot figure for the crossed segment x seed sensitivity analysis
(docs/crossed_inference_protocol.md).

Writes figures/eeg/fig_crossed_inference.{pdf,png}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CROSSED_DIR = ROOT / "results" / "eeg" / "followup" / "crossed_inference"
FIGDIR = ROOT / "figures" / "eeg"

# Okabe-Ito colorblind-safe palette
ORIGINAL_COLOR = "#0072B2"  # blue
CROSSED_COLOR = "#D55E00"   # vermillion
QRC_MARKER = "o"
ESN_MARKER = "s"


def main() -> None:
    boot = pd.read_csv(CROSSED_DIR / "crossed_bootstrap.csv")
    orig = pd.read_csv(CROSSED_DIR / "original_style_replication.csv")

    boot["row_label"] = boot["set"] + " | " + boot["comparison"]
    orig["row_label"] = orig["set"] + " | " + orig["kernel"] + " vs " + orig["comparator"] + orig["analysis_mode"].apply(
        lambda m: f" ({m})" if m != "not_applicable" else ""
    )
    merged = boot.merge(
        orig[["row_label", "observed_mean", "ci95_lo", "ci95_hi"]].rename(
            columns={"observed_mean": "orig_mean", "ci95_lo": "orig_lo", "ci95_hi": "orig_hi"}
        ),
        on="row_label", how="left",
    )
    merged["is_qrc"] = merged["kernel"] == "single_kernel"
    merged = merged.sort_values(["is_qrc", "set", "comparison"], ascending=[False, True, True]).reset_index(drop=True)

    n = len(merged)
    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.42 * n)))
    y = np.arange(n)[::-1]

    for yi, (_, row) in zip(y, merged.iterrows()):
        marker = QRC_MARKER if row["is_qrc"] else ESN_MARKER
        ax.plot([row["orig_lo"], row["orig_hi"]], [yi + 0.14, yi + 0.14], color=ORIGINAL_COLOR, lw=2.2, solid_capstyle="round")
        ax.plot(row["orig_mean"], yi + 0.14, marker=marker, color=ORIGINAL_COLOR, ms=6, mec="white", mew=0.6)
        ax.plot([row["ci95_lo"], row["ci95_hi"]], [yi - 0.14, yi - 0.14], color=CROSSED_COLOR, lw=2.2, solid_capstyle="round")
        ax.plot(row["bootstrap_mean"], yi - 0.14, marker=marker, color=CROSSED_COLOR, ms=6, mec="white", mew=0.6)

    ax.axvline(0.0, color="0.2", lw=1.2, ls="-")
    ax.set_yticks(y)
    ax.set_yticklabels(merged["row_label"], fontsize=8)
    ax.set_xlabel("Interaction (comparator degradation - kernel degradation), h=2 -> h=64")
    ax.set_title("Crossed segment x seed sensitivity: original vs. crossed bootstrap", fontsize=12)
    ax.grid(axis="x", alpha=0.25)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=ORIGINAL_COLOR, lw=2.2, marker="o", label="Original (seed-avg -> segment bootstrap)"),
        Line2D([], [], color=CROSSED_COLOR, lw=2.2, marker="o", label="Crossed (segment x seed bootstrap)"),
        Line2D([], [], color="0.3", marker=QRC_MARKER, ls="", label="QRC contrast"),
        Line2D([], [], color="0.3", marker=ESN_MARKER, ls="", label="ESN66 contrast"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="best", framealpha=0.9)
    fig.tight_layout()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_crossed_inference.{suffix}", dpi=220)
    plt.close(fig)
    print("wrote", FIGDIR / "fig_crossed_inference.pdf")
    print("wrote", FIGDIR / "fig_crossed_inference.png")


if __name__ == "__main__":
    main()
