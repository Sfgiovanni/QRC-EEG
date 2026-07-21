#!/usr/bin/env python3
"""APS-style figure for Gate 1B — post-gate robustness of the effective-kernel mechanism.

Four panels:
  A  distributions of the four confirmatory errors, tangent vs separable, log scale;
  B  tangent all-four pass fraction by r and u0;
  C  companion spectral radius by seed, r and u0, with the stability line at 1;
  D  tangent error vs amplitude epsilon, median and inter-seed 10-90% band.

Palette: Okabe-Ito colorblind-safe. r=0.7 blue (#0072B2), r=0.9 vermillion (#D55E00).
Frozen tolerances are drawn; outliers are not hidden; no truncated/misleading axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results/eeg/gate1b_robustness"
FIGDIR = ROOT / "figures/eeg"

R_COLORS = {0.7: "#0072B2", 0.9: "#D55E00"}  # Okabe-Ito blue / vermillion
U0_MARKERS = {-0.5: "o", 0.0: "s", 0.5: "^"}
TANGENT_COLOR = "#009E73"   # bluish green
SEPARABLE_COLOR = "#CC79A7"  # reddish purple
METRIC_ORDER = ["impulse_relative_frobenius", "step_relative_frobenius",
                "frequency_relative_frobenius", "memory_function_l1"]
METRIC_LABEL = {"impulse_relative_frobenius": "impulse", "step_relative_frobenius": "step",
                "frequency_relative_frobenius": "FFT", "memory_function_l1": "memory L1"}
TOL = {"impulse_relative_frobenius": 0.01, "step_relative_frobenius": 0.01,
       "frequency_relative_frobenius": 0.01, "memory_function_l1": 0.02}


def _positive(values: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    return values[values > 0]


def panel_a(ax, metrics: pd.DataFrame) -> None:
    """Log-scale error distributions: tangent vs separable across the four metrics."""
    valid = metrics[metrics.valid.astype(bool)]
    positions, data, colors = [], [], []
    xticks, xticklabels = [], []
    for i, metric in enumerate(METRIC_ORDER):
        for j, (theory, color) in enumerate((("tangent_recurrence", TANGENT_COLOR),
                                             ("separable_W_times_R", SEPARABLE_COLOR))):
            pos = i * 3 + j
            vals = _positive(valid[(valid.theory == theory) & (valid.metric == metric)].value.to_numpy(float))
            positions.append(pos); data.append(vals if len(vals) else np.array([np.nan])); colors.append(color)
        xticks.append(i * 3 + 0.5); xticklabels.append(METRIC_LABEL[metric])
    bp = ax.boxplot(data, positions=positions, widths=0.75, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor(color)
    for element in ("whiskers", "caps", "medians"):
        for k, line in enumerate(bp[element]):
            line.set_color(colors[k // 2] if element != "medians" else colors[k])
    ax.axhline(0.01, color="0.35", ls="--", lw=1.0, zorder=0)
    ax.axhline(0.02, color="0.55", ls=":", lw=1.0, zorder=0)
    ax.set_yscale("log")
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
    ax.set_ylabel("relative error (log)")
    ax.set_title("A  confirmatory error distributions", loc="left", fontsize=10)
    ax.text(0.98, 0.01 * 1.15, "tol 0.01", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6.5, color="0.35")
    handles = [Patch(facecolor=TANGENT_COLOR, alpha=0.55, label="tangent"),
               Patch(facecolor=SEPARABLE_COLOR, alpha=0.55, label="separable")]
    ax.legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)


def panel_b(ax, joint: pd.DataFrame) -> None:
    """Tangent all-four pass fraction by r and u0 (grouped bars)."""
    u0s = sorted(joint.u0.unique())
    rs = sorted(joint.r.unique())
    width = 0.38
    x = np.arange(len(u0s))
    for k, r in enumerate(rs):
        fracs = []
        for u0 in u0s:
            sub = joint[(joint.r == r) & (joint.u0 == u0) & (joint.valid)]
            fracs.append(sub.tangent_all4.mean() if len(sub) else np.nan)
        ax.bar(x + (k - 0.5) * width, fracs, width, color=R_COLORS[r], alpha=0.85,
               label=f"r={r}", edgecolor="white", linewidth=0.8)
    ax.axhline(0.9, color="0.35", ls="--", lw=1.0, zorder=0)
    ax.text(len(u0s) - 0.5, 0.905, "ROBUST >=0.90", ha="right", va="bottom", fontsize=6.5, color="0.35")
    ax.set_xticks(x); ax.set_xticklabels([f"u0={u0:g}" for u0 in u0s])
    ax.set_ylim(0, 1.05); ax.set_ylabel("tangent all-four pass fraction")
    ax.set_title("B  pass fraction by r and u0", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)


def panel_c(ax, spectrum: pd.DataFrame) -> None:
    """Companion spectral radius by seed, r and u0, with the stability line at 1."""
    finite = spectrum[np.isfinite(spectrum.companion_spectral_radius.astype(float))]
    for r in sorted(finite.r.unique()):
        for u0 in sorted(finite.u0.unique()):
            sub = finite[(finite.r == r) & (finite.u0 == u0)]
            ax.scatter(sub.seed, sub.companion_spectral_radius, s=34, color=R_COLORS[r],
                       marker=U0_MARKERS[u0], alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(1.0, color="#000000", ls="-", lw=1.2, zorder=2)
    ax.text(spectrum.seed.max(), 1.005, "stability limit", ha="right", va="bottom", fontsize=6.5)
    ax.set_xlabel("channel seed"); ax.set_ylabel("companion spectral radius")
    ax.set_xticks(sorted(spectrum.seed.unique()))
    ax.set_title("C  spectral radius by seed, r, u0", loc="left", fontsize=10)
    handles = [Line2D([], [], marker="s", color=R_COLORS[r], ls="", label=f"r={r}") for r in sorted(finite.r.unique())]
    handles += [Line2D([], [], marker=m, color="0.4", ls="", label=f"u0={u0:g}") for u0, m in U0_MARKERS.items()]
    ax.legend(handles=handles, fontsize=6.5, loc="best", ncol=2, framealpha=0.9)


def panel_d(ax, sweep: pd.DataFrame, epsilons: list[float]) -> None:
    """Tangent impulse error vs epsilon: median and inter-seed 10-90% band, per r."""
    valid = sweep[sweep.valid.astype(bool)]
    for r in sorted(valid.r.unique()):
        med, lo, hi, xs = [], [], [], []
        for eps in epsilons:
            sub = valid[(valid.r == r) & (np.isclose(valid.epsilon, eps))]
            vals = _positive(sub.tangent_impulse_relative_frobenius.to_numpy(float))
            if len(vals):
                xs.append(eps); med.append(np.median(vals))
                lo.append(np.quantile(vals, 0.10)); hi.append(np.quantile(vals, 0.90))
        if xs:
            ax.plot(xs, med, "-o", color=R_COLORS[r], ms=4, lw=1.6, label=f"r={r}", zorder=3)
            ax.fill_between(xs, lo, hi, color=R_COLORS[r], alpha=0.18, zorder=1)
    ax.axhline(0.01, color="0.35", ls="--", lw=1.0, zorder=0)
    ax.text(epsilons[0], 0.0115, "tol 0.01", ha="left", va="bottom", fontsize=6.5, color="0.35")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("amplitude epsilon"); ax.set_ylabel("tangent impulse error (median, 10-90%)")
    ax.set_title("D  error vs epsilon", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)


def build_joint(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, r, u0), group in metrics.groupby(["seed", "r", "u0"]):
        valid = bool(group.valid.astype(bool).iloc[0])

        def all4(theory):
            sub = group[(group.theory == theory) & (group.metric.isin(METRIC_ORDER))]
            return len(sub) == 4 and bool(sub["pass"].astype(str).str.lower().eq("true").all())
        rows.append({"seed": seed, "r": r, "u0": u0, "valid": valid,
                     "tangent_all4": all4("tangent_recurrence"),
                     "separable_all4": all4("separable_W_times_R")})
    return pd.DataFrame(rows)


def main() -> None:
    metrics = pd.read_csv(OUTDIR / "metrics_by_configuration.csv")
    spectrum = pd.read_csv(OUTDIR / "spectrum_by_configuration.csv")
    sweep = pd.read_csv(OUTDIR / "amplitude_sweep.csv")
    joint = build_joint(metrics)
    epsilons = sorted(sweep.epsilon.unique())

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "axes.axisbelow": True,
                         "figure.dpi": 120, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    panel_a(axes[0, 0], metrics)
    panel_b(axes[0, 1], joint)
    panel_c(axes[1, 0], spectrum)
    panel_d(axes[1, 1], sweep, epsilons)
    fig.suptitle("Gate 1B — post-gate robustness of the effective-kernel mechanism "
                 "(prespecified grid frozen before execution)", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig_gate1b_robustness.pdf")
    fig.savefig(FIGDIR / "fig_gate1b_robustness.png", dpi=200)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'fig_gate1b_robustness.pdf'} and .png")


if __name__ == "__main__":
    main()
