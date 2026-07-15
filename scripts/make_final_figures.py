#!/usr/bin/env python3
"""Generate the four canonical repository figures from frozen artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/final"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def effective_kernel() -> None:
    data = np.load(ROOT / "results/eeg/theory_vs_sim_responses.npz")
    meta = json.loads((ROOT / "results/eeg/theory_vs_sim_metadata.json").read_text())
    fig = plt.figure(figsize=(11.8, 7.2))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
    ax = fig.add_subplot(grid[:, 0]); ax.axis("off")
    boxes = [(0.08, .82, "state-history buffer\nρ(t),…,ρ(t−K)"), (.08, .57, "weighted state mixture\nΣ wτ ρ(t−τ)"),
             (.08, .32, "input-dependent channel\nE_u and feedback"), (.08, .07, "66 Pauli expectations\nCρ(t)")]
    for x, y, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), .82, .13, fc="#E6F2F8", ec="#0072B2", lw=1.2))
        ax.text(x+.41, y+.065, label, ha="center", va="center")
    for y in (.78, .53, .28):
        ax.annotate("", xy=(.49, y-.06), xytext=(.49, y+.04), arrowprops={"arrowstyle": "->", "color": "#333333"})
    ax.text(.49, .985, r"$H_{actual}(z)=C[zI-AW_K(z)]^{-1}B$", ha="center", va="top", fontsize=10)
    ax.text(.49, .005, r"$H_{sep}(z)=W_K(z)R(z)$: falsified", ha="center", color="#D55E00")
    ax.set_title("(a) State history inside recurrent feedback", loc="left")

    t = np.arange(data["measured_impulse"].shape[0])
    norm = lambda name: np.linalg.norm(data[name], axis=1)
    ax = fig.add_subplot(grid[0, 1])
    for name, label, color, style in (("measured_impulse", "nonlinear", COLORS[6], "-"),
                                       ("tangent_impulse", "tangent", COLORS[0], "--"),
                                       ("separable_impulse", "separable", COLORS[1], ":")):
        ax.plot(t, norm(name), label=label, color=color, ls=style, lw=1.2)
    ax.set(xlabel="lag (samples)", ylabel="response Frobenius norm", title="(b) Impulse response")
    ax.set_yscale("log"); ax.legend()

    ax = fig.add_subplot(grid[0, 2])
    for name, label, color, style in (("measured_step", "nonlinear", COLORS[6], "-"),
                                       ("tangent_step", "tangent", COLORS[0], "--"),
                                       ("separable_step", "separable", COLORS[1], ":")):
        ax.plot(t, norm(name), label=label, color=color, ls=style, lw=1.2)
    ax.set(xlabel="time (samples)", ylabel="response Frobenius norm", title="(c) Step response")
    ax.set_yscale("log"); ax.legend()

    eig = data["companion_eigenvalues"]
    ax = fig.add_subplot(grid[1, 1])
    theta = np.linspace(0, 2*np.pi, 400); ax.plot(np.cos(theta), np.sin(theta), color=".5", lw=.8)
    ax.scatter(eig.real, eig.imag, s=3, alpha=.25, color=COLORS[0], rasterized=True)
    ax.set_aspect("equal"); ax.set(xlabel="Re(λ)", ylabel="Im(λ)", title="(d) Companion spectrum")
    ax.text(.03, .03, f"radius={meta['companion']['spectral_radius']:.4f}", transform=ax.transAxes)

    metrics = pd.read_csv(ROOT / "results/eeg/theory_vs_sim_check.csv")
    pivot = metrics.pivot(index="metric", columns="theory", values="value")
    ax = fig.add_subplot(grid[1, 2])
    x = np.arange(len(pivot)); width=.34
    ax.bar(x-width/2, pivot["tangent_recurrence"], width, color=COLORS[0], label="tangent")
    ax.bar(x+width/2, pivot["separable_W_times_R"], width, color=COLORS[1], label="separable")
    ax.axhline(.01, color=".3", ls="--", lw=.9, label="Frobenius tolerance")
    ax.set_yscale("log"); ax.set_xticks(x, [m.replace("_relative_frobenius", "").replace("memory_function_l1", "memory L1") for m in pivot.index], rotation=25, ha="right")
    ax.set(ylabel="error", title="(e) Frozen local checks"); ax.legend()
    fig.tight_layout(); save(fig, "fig_effective_kernel_mechanism")


def synthetic_validation() -> None:
    table = pd.read_csv(ROOT / "results/synth/theory_predictions_vs_measured.csv")
    meta = json.loads((ROOT / "results/synth/stage2_metadata.json").read_text())["statistics"]
    table["pred_center"] = table.predicted_slope - table.groupby("scenario").predicted_slope.transform("mean")
    table["meas_center"] = table.measured_slope - table.groupby("scenario").measured_slope.transform("mean")
    scenarios = sorted(table.scenario.unique()); cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    for i, (scenario, group) in enumerate(table.groupby("scenario")):
        axes[0].scatter(group.predicted_slope, group.measured_slope, s=24, color=cmap(i), label=scenario)
        axes[1].scatter(group.pred_center, group.meas_center, s=24, color=cmap(i))
    all_values = np.r_[table.predicted_slope, table.measured_slope]
    axes[0].plot([all_values.min(), all_values.max()], [all_values.min(), all_values.max()], color=".4", ls="--")
    axes[0].set(xlabel="predicted NRMSE slope", ylabel="measured NRMSE slope",
                title=f"(a) Across processes (ρ={meta['aggregate_spearman']:.3f})")
    axes[1].axhline(0, color=".5", lw=.7); axes[1].axvline(0, color=".5", lw=.7)
    axes[1].set(xlabel="predicted slope, scenario-centered", ylabel="measured slope, scenario-centered",
                title="(b) Within-scenario deviations")
    rho = pd.Series(meta["scenario_spearman"]).reindex(scenarios)
    axes[2].barh(np.arange(len(rho)), rho, color=[COLORS[1] if x < 0 else COLORS[0] for x in rho])
    axes[2].axvline(0, color=".3", lw=.8); axes[2].set_yticks(np.arange(len(rho)), rho.index, fontsize=6.5)
    axes[2].set(xlabel="within-scenario Spearman", title="(c) Ordering agreement and failures", xlim=(-1.05,1.05))
    axes[0].legend(fontsize=5.2, ncol=2, loc="upper left")
    fig.tight_layout(); save(fig, "fig_synthetic_theory_validation")


def eeg_horizon() -> None:
    curves = pd.read_csv(ROOT / "results/eeg/gate_nrmse_curves.csv")
    useful = pd.read_csv(ROOT / "results/eeg/useful_horizon_v2.csv")
    interactions = pd.read_csv(ROOT / "results/eeg/gate_interactions.csv")
    models = ["single_kernel", "QRC_K0", "AB_noaux", "AR", "NVAR2", "persistence"]
    labels = {"single_kernel":"kernel", "QRC_K0":"QRC K=0", "AB_noaux":"AB", "AR":"AR", "NVAR2":"NVAR2", "persistence":"persistence"}
    color = dict(zip(models, COLORS))
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2)); axes = axes.ravel()
    for ax, set_name in zip(axes[:3], ("F", "Z", "S")):
        for model in models:
            group = curves[(curves["set"] == set_name) & (curves.construction == model)].sort_values("horizon_ms")
            if group.empty: continue
            ax.plot(group.horizon_ms, group.mean_nrmse, marker="o", ms=3, color=color[model], label=labels[model])
            ax.fill_between(group.horizon_ms, group.ci95_lo, group.ci95_hi, color=color[model], alpha=.08)
        ax.axhline(1, color=".35", ls="--", lw=.8); ax.set_xscale("log", base=2)
        ax.set(xlabel="forecast horizon (ms)", ylabel="NRMSE", title=f"({chr(97+list(('F','Z','S')).index(set_name))}) Set {set_name}")
        causal = interactions[(interactions["set"] == set_name) & (interactions.comparator == "QRC_K0")].iloc[0]
        label = "null" if causal.ci95_lo <= 0 <= causal.ci95_hi else "supported"
        ax.text(.03,.94,f"kernel×horizon vs K0: {label}",transform=ax.transAxes,va="top",fontsize=7)
    axes[0].legend(ncol=2)
    ax = axes[3]
    uh = useful[useful.construction.isin(models) & useful["set"].isin(["F","Z","S"])].copy()
    x = np.arange(3); width=.12
    for i, model in enumerate(models):
        values = uh[uh.construction == model].set_index("set").reindex(["F","Z","S"]).useful_horizon_ms
        ax.bar(x+(i-2.5)*width, values, width, label=labels[model], color=color[model])
    ax.set_xticks(x, ["F","Z","S"]); ax.set(ylabel="useful horizon (ms)", title="(d) Symmetric useful horizon")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); save(fig, "fig_eeg_horizon_dependence")


def resources_shots() -> None:
    resources = pd.read_csv(ROOT / "results/resources/qrc_resource_table.csv")
    levels = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_classification.csv")
    strata = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_strata_classification.csv")
    summary = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_summary.csv")
    principal = ["QRC_K0", "AB_noaux", "single_kernel"]
    res = resources[resources.construction.isin(principal)].set_index("construction").reindex(principal)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ax=axes[0,0]; ax.bar(["K0","AB","K=15"],res.dense_buffer_bytes/1024,color=COLORS[:3])
    ax.set(ylabel="dense buffer (KiB, complex128)",title="(a) Simulated state-buffer memory")
    ax=axes[0,1]; ax.bar(["K0","AB","K=15"],res.classical_complex_ops_proxy_per_step,color=COLORS[:3],label="classical proxy")
    ax.set(ylabel="complex-operation proxy / step",title="(b) Dense simulator cost")
    ax.text(.03,.94,f"66 groups × N shots / step",transform=ax.transAxes,va="top")
    ax=axes[1,0]; ax.plot(levels.shots,100*levels["median"],marker="o",color=COLORS[0],label="median")
    ax.plot(levels.shots,100*levels.p90,marker="s",color=COLORS[1],label="P90")
    ax.axhline(5,color=".4",ls="--",lw=.8); ax.axhline(10,color=".4",ls=":",lw=.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set(xlabel="shots / Pauli observable",ylabel="relative NRMSE inflation (%)",title="(c) Median and upper tail")
    ax.legend()
    ax=axes[1,1]; passing=strata.groupby("shots").stratum_pass.mean().reindex(levels.shots)
    ax.plot(passing.index,100*passing,marker="o",color=COLORS[2],label="passing set×horizon strata")
    for set_name,c in zip(("F","Z","S"),COLORS[:3]):
        values=summary[summary["set"]==set_name].groupby("shots").p90_relative_nrmse_inflation.median()
        ax.plot(values.index,100*values,ls="--",color=c,label=f"{set_name} median cell P90")
    ax.set_xscale("log"); ax.set(xlabel="shots / observable",ylabel="percent (%)",title="(d) Heterogeneity across strata")
    ax.legend(fontsize=6)
    fig.text(.5,.005,"Measurement counts assume separate ensembles; shot noise is not a complete hardware model.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.025,1,1)); save(fig,"fig_resources_and_shots")


def inventory() -> None:
    entries = [
        ("Effective-kernel mechanism", "fig_effective_kernel_mechanism", "Gate 1 NPZ/CSV/metadata",
         "State history changes recurrent feedback; tangent local response passes while external factorization fails.",
         "Local linearization only; r is not asserted as a finite-system pole.",
         "State-history recurrence, nonlinear/tangent/separable responses, and companion spectrum for the frozen local check."),
        ("Synthetic theory validation", "fig_synthetic_theory_validation", "Gate 2 combined CSV/metadata",
         "Strong between-process association coexists with moderate and sometimes negative within-process ordering.",
         "Frozen synthetic scenarios only; no universal best-kernel prediction.",
         "Predicted versus measured degradation slopes across processes and after within-scenario centering; negative bars expose ordering failures."),
        ("EEG horizon dependence", "fig_eeg_horizon_dependence", "Gate EEG curves/interactions/useful horizon v2",
         "Horizon dependence is supported in F/Z and null in S; useful horizon is the absolute-skill headline.",
         "Segment-level single database; h=64 is sensitivity endpoint with NRMSE>1.",
         "Held-out NRMSE curves and symmetric useful horizons for F, Z and S; the S-null is retained."),
        ("Resources and shots", "fig_resources_and_shots", "Gate 3 resources/shot summaries",
         "Finite-shot sensitivity is mixed, with moderate median and large tail alongside high measurement cost.",
         "Independent Pauli sampling is not decoherence, backaction or hardware execution.",
         "Density-matrix buffer cost, operation proxy, shot-noise inflation and passing-stratum fraction."),
    ]
    lines=["# Figure inventory", "", "All figures are generated by `scripts/make_final_figures.py` as vector PDF and 600-dpi PNG.", ""]
    for title,name,sources,message,limits,caption in entries:
        lines += [f"## {title}", "", f"- Files: `figures/final/{name}.pdf` and `.png`", "- Script: `scripts/make_final_figures.py`",
                  f"- Sources: {sources}", f"- Permitted message: {message}", f"- Limitation: {limits}", f"- Suggested caption: {caption}", ""]
    (ROOT / "docs/figure_inventory.md").write_text("\n".join(lines))


def main() -> None:
    effective_kernel(); synthetic_validation(); eeg_horizon(); resources_shots(); inventory()
    print("wrote four canonical PDF/PNG figures; no .tex files read or written")


if __name__ == "__main__":
    main()
