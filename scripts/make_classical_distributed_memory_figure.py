#!/usr/bin/env python3
"""Aggregated table and figure for the classical distributed-memory ESN control
(docs/classical_distributed_memory_protocol.md).

Writes:
  results/eeg/followup/classical_control/tab_classical_distributed_memory.csv
  figures/eeg/fig_classical_distributed_memory.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FOLLOWUP_DIR = ROOT / "results" / "eeg" / "followup"
FIGDIR = ROOT / "figures" / "eeg"
EEG_CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"

# Okabe-Ito colorblind-safe palette (matches scripts/make_gate1b_figure.py)
COLORS = {"ESN66_K0": "#0072B2", "ESN66_AB": "#E69F00", "ESN66_kernel": "#009E73"}
MODE_LS = {"fixed_core": "-", "retuned_core": "--"}
DISPLAY = {"ESN66_K0": "ESN66 K=0", "ESN66_AB": "ESN66 AB (concentrated)", "ESN66_kernel": "ESN66 kernel (distributed)"}


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    cfg = yaml.safe_load(EEG_CONFIG_PATH.read_text())
    seed = int(cfg["split"]["seed"])
    df = pd.read_csv(FOLLOWUP_DIR / "raw" / "esn_distributed_memory_holdout_by_segment_seed.csv")

    per_segment = df.groupby(["construction", "analysis_mode", "set", "horizon", "segment_id"], as_index=False)["nrmse"].mean()

    rows = []
    for keys, group in per_segment.groupby(["construction", "analysis_mode", "set", "horizon"], sort=True):
        values = group["nrmse"].to_numpy()
        lo, hi = bootstrap_ci(values, seed + int(keys[3]))
        rows.append({
            "construction": keys[0], "analysis_mode": keys[1], "set": keys[2], "horizon": int(keys[3]),
            "n_segments": len(values), "mean_nrmse": float(values.mean()), "ci95_lo": lo, "ci95_hi": hi,
        })
    curves = pd.DataFrame(rows)

    # Difference from ESN66_K0, same mode/set/horizon.
    k0 = curves[curves.construction == "ESN66_K0"].set_index(["analysis_mode", "set", "horizon"])["mean_nrmse"]
    curves["diff_from_k0"] = curves.apply(
        lambda r: r["mean_nrmse"] - k0.loc[(r["analysis_mode"], r["set"], r["horizon"])], axis=1
    )

    out_dir = FOLLOWUP_DIR / "classical_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_dir / "tab_classical_distributed_memory.csv", index=False)
    print("wrote", out_dir / "tab_classical_distributed_memory.csv", f"({len(curves)} rows)")

    sets = cfg["data"]["sets"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex="col")
    for col, set_name in enumerate(sets):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        for construction in ("ESN66_K0", "ESN66_AB", "ESN66_kernel"):
            for mode in ("fixed_core", "retuned_core"):
                slab = curves[
                    (curves.set == set_name) & (curves.construction == construction) & (curves.analysis_mode == mode)
                ].sort_values("horizon")
                if slab.empty:
                    continue
                ax_top.plot(
                    slab.horizon, slab.mean_nrmse, color=COLORS[construction], ls=MODE_LS[mode],
                    marker="o" if mode == "fixed_core" else "^", ms=4,
                    label=f"{DISPLAY[construction]} ({mode})",
                )
                ax_top.fill_between(slab.horizon, slab.ci95_lo, slab.ci95_hi, color=COLORS[construction], alpha=0.12)

                diff_slab = curves[
                    (curves.set == set_name) & (curves.construction == construction) & (curves.analysis_mode == mode)
                ].sort_values("horizon")
                ax_bot.plot(
                    diff_slab.horizon, diff_slab.diff_from_k0, color=COLORS[construction], ls=MODE_LS[mode],
                    marker="o" if mode == "fixed_core" else "^", ms=4,
                )
        ax_top.set_xscale("log", base=2)
        ax_bot.set_xscale("log", base=2)
        ax_bot.axhline(0.0, color="0.3", lw=1.0, ls=":")
        ax_top.set_title(f"Set {set_name}")
        ax_bot.set_xlabel("Horizon (samples)")
        if col == 0:
            ax_top.set_ylabel("NRMSE")
            ax_bot.set_ylabel("NRMSE - ESN66_K0")
        ax_top.grid(alpha=0.25)
        ax_bot.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Classical distributed-memory ESN control: NRMSE vs. horizon (top) and difference from ESN66_K0 (bottom)", fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_classical_distributed_memory.{suffix}", dpi=220)
    plt.close(fig)
    print("wrote", FIGDIR / "fig_classical_distributed_memory.pdf")
    print("wrote", FIGDIR / "fig_classical_distributed_memory.png")


if __name__ == "__main__":
    main()
