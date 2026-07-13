#!/usr/bin/env python3
"""Generate tab_eeg_endpoints, tab_quadratic_capacity (REVTeX + csv), and
fig_eeg_demand_gradient / fig_eeg_capacity_vs_gain (APS style, PDF+PNG).
Also runs the capacity-vs-gain regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "results" / "eeg"
FIGURES_DIR = ROOT / "figures" / "eeg"
PAPER_DIR = ROOT / "paper"
STYLE_PATH = ROOT / "src" / "qrc_eeg" / "style" / "aps.mplstyle"

CONSTRUCTION_LABELS = {
    "AB_noaux": "AB (noaux)",
    "single_kernel": "Single kernel",
    "dual_kernel": "Dual kernel",
    "triangular": "Triangular",
    "uniform": "Uniform",
    "ESN": "ESN",
}


def revtex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = " & ".join(str(c) for c in df.columns)
    lines = [
        r"\begin{table}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{" + "l" * len(df.columns) + "}",
        cols + r" \\",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    return "\n".join(lines)


def build_endpoints_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_by_segment_seed.csv")
    per_segment = df.groupby(["construction", "set", "segment_id"], as_index=False)[
        ["nrmse", "rmse", "r2", "mae"]
    ].mean()
    agg = per_segment.groupby(["construction", "set"]).agg(
        nrmse_mean=("nrmse", "mean"),
        nrmse_sem=("nrmse", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
        rmse_mean=("rmse", "mean"),
        rmse_sem=("rmse", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
        r2_mean=("r2", "mean"),
        mae_mean=("mae", "mean"),
    ).reset_index()
    agg["construction"] = agg["construction"].map(CONSTRUCTION_LABELS).fillna(agg["construction"])
    for col in ["nrmse_mean", "nrmse_sem", "rmse_mean", "rmse_sem", "r2_mean", "mae_mean"]:
        agg[col] = agg[col].round(4)
    return agg


def build_capacity_gain_regression() -> pd.DataFrame:
    contrasts = pd.read_csv(RESULTS_DIR / "tab_eeg_contrasts.csv")
    capacity = pd.read_csv(RESULTS_DIR / "quadratic_capacity.csv")
    demand = pd.read_csv(RESULTS_DIR / "nonlinear_demand.csv")

    rows = []
    for _, row in contrasts.iterrows():
        state_name, comparator = row["comparison"].split(" vs ")
        state_capacity_row = capacity.loc[capacity["construction"] == state_name, "quadratic_capacity_mean"]
        comp_capacity_row = capacity.loc[capacity["construction"] == comparator, "quadratic_capacity_mean"]
        if state_capacity_row.empty or comp_capacity_row.empty:
            continue
        capacity_gap = float(state_capacity_row.iloc[0]) - float(comp_capacity_row.iloc[0])
        demand_row = demand.loc[demand["set"] == row["set"], "nonlinear_demand"]
        rows.append(
            {
                "comparison": row["comparison"],
                "set": row["set"],
                "horizon": row["horizon"],
                "delta_nrmse": row["mean_diff_rmse_comparator_minus_state"],
                "capacity_gap": capacity_gap,
                "nonlinear_demand": float(demand_row.iloc[0]) if not demand_row.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ols_with_ci(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 3 or np.std(x) < 1e-12:
        return {"slope": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": n}
    x_design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    resid = y - x_design @ beta
    dof = max(n - 2, 1)
    sigma2 = float(np.sum(resid**2) / dof)
    cov = sigma2 * np.linalg.inv(x_design.T @ x_design)
    se_slope = float(np.sqrt(cov[1, 1]))
    slope = float(beta[1])
    return {"slope": slope, "ci_lo": slope - 1.96 * se_slope, "ci_hi": slope + 1.96 * se_slope, "n": n}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    endpoints = build_endpoints_table()
    endpoints.to_csv(RESULTS_DIR / "tab_eeg_endpoints.csv", index=False)
    (PAPER_DIR / "tab_eeg_endpoints.tex").write_text(
        revtex_table(endpoints, "Held-out EEG forecasting endpoints by construction and set.", "tab:eeg_endpoints")
    )

    capacity = pd.read_csv(RESULTS_DIR / "quadratic_capacity.csv")
    capacity_display = capacity[["construction", "quadratic_capacity_mean", "quadratic_capacity_std", "n_qubits_or_units", "n_features"]].copy()
    capacity_display["construction"] = capacity_display["construction"].map(CONSTRUCTION_LABELS).fillna(capacity_display["construction"])
    (PAPER_DIR / "tab_quadratic_capacity.tex").write_text(
        revtex_table(
            capacity_display,
            "Frozen synthetic quadratic capacity and register size by construction.",
            "tab:quadratic_capacity",
        )
    )

    contrasts_path = RESULTS_DIR / "tab_eeg_contrasts.csv"
    if contrasts_path.exists():
        contrasts = pd.read_csv(contrasts_path)
        cols = ["comparison", "set", "horizon", "mean_diff_rmse_comparator_minus_state", "ci95_lo", "ci95_hi", "p_holm", "cohen_dz", "win_fraction_state"]
        (PAPER_DIR / "tab_eeg_contrasts.tex").write_text(
            revtex_table(contrasts[cols].round(4), "Paired contrasts, family eeg\\_primary, Holm-corrected.", "tab:eeg_contrasts")
        )

    plt.style.use(str(STYLE_PATH))

    demand = pd.read_csv(RESULTS_DIR / "nonlinear_demand.csv")
    contrasts = pd.read_csv(contrasts_path) if contrasts_path.exists() else pd.DataFrame()
    if not contrasts.empty:
        # NOTE: the pre-registered continuous nonlinear-demand score is
        # near-degenerate (~0 for all sets; see docs/eeg_preregistration.md
        # amendment), so it cannot serve as a meaningful continuous x-axis
        # (three points stacked at x~0 is not a gradient, it's noise on a
        # zero-width axis). Plotting the categorical clinical-state ordering
        # (Z -> F -> S) instead, labeled explicitly as categorical, not the
        # pre-registered continuous score. single_kernel vs ESN is included
        # only as a caveated secondary curve (dimension-confounded: ESN used
        # 200 readout features vs 66 for every quantum arm; see RESULTS.md).
        fig, ax = plt.subplots()
        set_order = ["Z", "F", "S"]
        markers = {"single_kernel vs AB_noaux": "s", "single_kernel vs ESN": "^"}
        x_pos = {s: i for i, s in enumerate(set_order)}
        for comparison, marker in markers.items():
            sub = contrasts[contrasts["comparison"] == comparison]
            sub = sub[sub["horizon"] == 1]
            present = [s for s in set_order if s in sub["set"].values]
            x = [x_pos[s] for s in present]
            y = [sub.loc[sub["set"] == s, "mean_diff_rmse_comparator_minus_state"].iloc[0] for s in present]
            yerr_lo = [sub.loc[sub["set"] == s, "mean_diff_rmse_comparator_minus_state"].iloc[0] - sub.loc[sub["set"] == s, "ci95_lo"].iloc[0] for s in present]
            yerr_hi = [sub.loc[sub["set"] == s, "ci95_hi"].iloc[0] - sub.loc[sub["set"] == s, "mean_diff_rmse_comparator_minus_state"].iloc[0] for s in present]
            label = comparison.replace("single_kernel vs ", "vs ")
            if comparison == "single_kernel vs ESN":
                label += " (dimension-confounded, see text)"
            ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], marker=marker, linestyle="-", label=label, capsize=2)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(set_order)
        ax.set_xlabel("Clinical state (categorical: healthy -> interictal -> ictal)")
        ax.set_ylabel(r"$\Delta$NRMSE (comparator $-$ kernel), $h=1$")
        ax.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"fig_eeg_demand_gradient.{ext}")
        plt.close(fig)
        print("wrote fig_eeg_demand_gradient (pdf, png) -- categorical x-axis, not the degenerate continuous score")

    reg_df = build_capacity_gain_regression()
    reg_df.to_csv(RESULTS_DIR / "capacity_gain_regression_rows.csv", index=False)
    reg_demand = ols_with_ci(reg_df["nonlinear_demand"].to_numpy(), reg_df["delta_nrmse"].to_numpy())
    reg_capacity = ols_with_ci(reg_df["capacity_gap"].to_numpy(), reg_df["delta_nrmse"].to_numpy())
    pd.DataFrame(
        [
            {"predictor": "nonlinear_demand", **reg_demand},
            {"predictor": "capacity_gap", **reg_capacity},
        ]
    ).to_csv(RESULTS_DIR / "capacity_demand_regression_summary.csv", index=False)
    print("capacity/demand regression:", reg_demand, reg_capacity)

    if not reg_df.empty:
        fig, ax = plt.subplots()
        ax.scatter(reg_df["capacity_gap"], reg_df["delta_nrmse"], s=14)
        if not np.isnan(reg_capacity["slope"]):
            xs = np.linspace(reg_df["capacity_gap"].min(), reg_df["capacity_gap"].max(), 50)
            x_design = np.column_stack([np.ones(len(xs)), xs])
            x_full = np.column_stack([np.ones(len(reg_df)), reg_df["capacity_gap"]])
            beta, *_ = np.linalg.lstsq(x_full, reg_df["delta_nrmse"], rcond=None)
            ax.plot(xs, x_design @ beta, color="black", linewidth=1.0)
        ax.set_xlabel("Quadratic-capacity gap (kernel $-$ comparator)")
        ax.set_ylabel(r"$\Delta$NRMSE (comparator $-$ kernel)")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"fig_eeg_capacity_vs_gain.{ext}")
        plt.close(fig)
        print("wrote fig_eeg_capacity_vs_gain (pdf, png)")


if __name__ == "__main__":
    main()
