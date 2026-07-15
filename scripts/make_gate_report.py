#!/usr/bin/env python3
"""Apply the frozen EEG model-by-horizon gate mechanically."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.statistics import holm  # noqa: E402

CONFIG = ROOT / "config" / "eeg_frozen.yaml"
RESULTS = ROOT / "results" / "eeg"
FIGURES = ROOT / "figures" / "eeg"
COMPARATORS = ["QRC_K0", "AR", "NVAR2", "persistence", "tapped_delay"]
FITTED_CLASSICAL = ["AR", "NVAR2", "tapped_delay"]
DISPLAY = {
    "single_kernel": "QRC kernel", "QRC_K0": "QRC K=0", "AR": "AR",
    "NVAR2": "NVAR2", "persistence": "Persistence", "tapped_delay": "Tapped delay",
    "AB_noaux": "AB noaux", "dual_kernel": "QRC dual kernel",
    "triangular": "QRC triangular", "uniform": "QRC uniform",
    "ESN": "ESN-200", "ESN_66": "ESN-66",
}


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_interaction(per_segment, set_name, comparator, h_short, h_long, seed):
    slab = per_segment[
        (per_segment["set"] == set_name)
        & (per_segment["construction"].isin(["single_kernel", comparator]))
        & (per_segment["horizon"].isin([h_short, h_long]))
    ]
    pivot = slab.pivot(index="segment_id", columns=["construction", "horizon"], values="nrmse").dropna()
    kernel_d = pivot[("single_kernel", h_long)] - pivot[("single_kernel", h_short)]
    comp_d = pivot[(comparator, h_long)] - pivot[(comparator, h_short)]
    interaction = (comp_d - kernel_d).to_numpy()
    lo, hi = bootstrap_ci(interaction, seed)
    p = float(st.wilcoxon(interaction).pvalue) if np.any(np.abs(interaction) > 1e-12) else 1.0
    return {
        "comparison": f"single_kernel vs {comparator}", "comparator": comparator, "set": set_name,
        "h_short": h_short, "h_long": h_long, "n_segments": len(interaction),
        "kernel_degradation": float(kernel_d.mean()), "comparator_degradation": float(comp_d.mean()),
        "interaction_comp_minus_kernel": float(interaction.mean()), "ci95_lo": lo, "ci95_hi": hi,
        "p_wilcoxon": p,
    }


def fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.4f}"


def markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""

    table = frame.reset_index() if include_index else frame.copy()
    headers = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("NA")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    fs = float(cfg["data"]["sampling_rate_hz"])
    h_short, h_long = cfg["eeg_gate"]["h_short"], cfg["eeg_gate"]["h_long"]
    seed = int(cfg["split"]["seed"])
    primary = pd.read_csv(RESULTS / "raw" / "eeg_holdout_by_segment_seed.csv")
    classical = pd.read_csv(RESULTS / "raw" / "eeg_gate_classical_by_segment_seed.csv")
    frames = [primary, classical]
    esn66_path = RESULTS / "raw" / "eeg_holdout_esn66_by_segment_seed.csv"
    if esn66_path.exists():
        frames.append(pd.read_csv(esn66_path))
    raw = pd.concat(frames, ignore_index=True)
    per_segment = raw.groupby(["construction", "set", "horizon", "segment_id"], as_index=False)["nrmse"].mean()

    curve_rows = []
    for keys, group in per_segment.groupby(["construction", "set", "horizon"], sort=True):
        values = group["nrmse"].to_numpy()
        lo, hi = bootstrap_ci(values, seed + int(keys[2]))
        curve_rows.append({
            "construction": keys[0], "set": keys[1], "horizon": int(keys[2]),
            "horizon_ms": 1000.0 * int(keys[2]) / fs, "n_segments": len(values),
            "mean_nrmse": float(values.mean()), "ci95_lo": lo, "ci95_hi": hi,
        })
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(RESULTS / "gate_nrmse_curves.csv", index=False)

    interactions = pd.DataFrame([
        paired_interaction(per_segment, set_name, comparator, h_short, h_long, seed)
        for set_name in cfg["data"]["sets"] for comparator in COMPARATORS
    ])
    interactions["p_holm"] = holm(interactions["p_wilcoxon"].to_numpy())
    interactions["expected_direction"] = interactions["interaction_comp_minus_kernel"] > 0
    interactions["significant_expected"] = (
        interactions["expected_direction"] & (interactions["ci95_lo"] > 0) & (interactions["p_holm"] < 0.05)
    )
    interactions.to_csv(RESULTS / "gate_interactions.csv", index=False)

    useful_rows = []
    pivot = per_segment.pivot_table(index=["set", "horizon", "segment_id"], columns="construction", values="nrmse")
    for set_name in cfg["data"]["sets"]:
        for model in sorted(per_segment["construction"].unique()):
            qualifying = []
            detail = []
            for horizon in cfg["readout"]["horizons"]:
                try:
                    slab = pivot.loc[(set_name, horizon)].dropna(subset=[model, "persistence", "AR"])
                except KeyError:
                    continue
                model_values = slab[model].to_numpy()
                diff_p = (slab["persistence"] - slab[model]).to_numpy()
                diff_ar = (slab["AR"] - slab[model]).to_numpy()
                p_lo, p_hi = bootstrap_ci(diff_p, seed + horizon + 11)
                ar_lo, ar_hi = bootstrap_ci(diff_ar, seed + horizon + 29)
                qualifies = float(model_values.mean()) < 1.0 and p_lo > 0 and ar_lo > 0
                if qualifies:
                    qualifying.append(horizon)
                detail.append((horizon, float(model_values.mean()), p_lo, p_hi, ar_lo, ar_hi, qualifies))
            useful = max(qualifying) if qualifying else np.nan
            useful_rows.append({
                "construction": model, "set": set_name, "useful_horizon": useful,
                "useful_horizon_ms": 1000.0 * useful / fs if not pd.isna(useful) else np.nan,
                "criterion": "mean NRMSE < 1; bootstrap lower CI for persistence-model and AR-model > 0",
            })
    useful_df = pd.DataFrame(useful_rows)
    useful_df.to_csv(RESULTS / "useful_horizon.csv", index=False)

    decisions, failures = [], []
    for set_name in ["F", "Z"]:
        k0 = interactions[(interactions["set"] == set_name) & (interactions["comparator"] == "QRC_K0")].iloc[0]
        classical_rows = interactions[(interactions["set"] == set_name) & interactions["comparator"].isin(FITTED_CLASSICAL)]
        strongest = classical_rows.sort_values("comparator_degradation", ascending=True).iloc[0]
        k0_ok, classical_ok = bool(k0.significant_expected), bool(strongest.significant_expected)
        decisions.append((set_name, k0, strongest, k0_ok, classical_ok))
        if not k0_ok:
            failures.append(f"{set_name}: kernel did not significantly degrade more slowly than QRC K=0")
        if not classical_ok:
            failures.append(f"{set_name}: kernel did not significantly beat strongest classical {strongest.comparator}")
    verdict = "PASS" if not failures else "FAIL"

    gate_models = [m for m in DISPLAY if m in set(curves.construction)]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, set_name in zip(axes, ["F", "Z", "S"]):
        for model in gate_models:
            slab = curves[(curves["set"] == set_name) & (curves["construction"] == model)].sort_values("horizon")
            ax.plot(slab["horizon_ms"], slab["mean_nrmse"], marker="o", label=DISPLAY[model])
        ax.set_xscale("log", base=2); ax.set_title(f"Set {set_name}"); ax.set_xlabel("Horizon (ms)"); ax.grid(alpha=.25)
    axes[0].set_ylabel("NRMSE")
    axes[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig_eeg_gate_curves.{suffix}", dpi=220)
    plt.close(fig)

    lines = [
        "# EEG causal-memory gate report", "", f"## Mechanical verdict: **{verdict}**", "",
        "Frozen rule: both F and Z must significantly favor the kernel over QRC K=0 and over the fitted classical comparator with the slowest degradation.", "",
        "| Set | K=0 interaction | K=0 Holm p | K=0 condition | Strongest classical | Classical interaction | Classical Holm p | Classical condition |",
        "|---|---:|---:|---|---|---:|---:|---|",
    ]
    for set_name, k0, strongest, k0_ok, classical_ok in decisions:
        lines.append(
            f"| {set_name} | {k0.interaction_comp_minus_kernel:.4f} [{k0.ci95_lo:.4f}, {k0.ci95_hi:.4f}] | {k0.p_holm:.4g} | {'PASS' if k0_ok else 'FAIL'} | "
            f"{strongest.comparator} | {strongest.interaction_comp_minus_kernel:.4f} [{strongest.ci95_lo:.4f}, {strongest.ci95_hi:.4f}] | {strongest.p_holm:.4g} | {'PASS' if classical_ok else 'FAIL'} |"
        )
    if failures:
        lines += ["", "Conditions that failed:"] + [f"- {failure}." for failure in failures]
    lines += [
        "", "## NRMSE curves", "",
        f"Horizons are in samples and milliseconds at fs={fs:.2f} Hz. Primary interaction endpoints are h={h_short} and h={h_long}.", "",
        "![Gate NRMSE curves](../../figures/eeg/fig_eeg_gate_curves.png)", "",
    ]
    for set_name in ["F", "Z", "S"]:
        table = curves[(curves["set"] == set_name) & curves["construction"].isin(gate_models)].pivot(index="construction", columns="horizon_ms", values="mean_nrmse")
        table = table.rename(index=DISPLAY)
        table.columns = [f"{value:.1f} ms" for value in table.columns]
        lines += [f"### Set {set_name}", "", markdown_table(table.round(4), include_index=True), ""]
    lines += ["## Useful horizon", "", markdown_table(useful_df), ""]
    lines += [
        "## Frozen interaction family", "",
        markdown_table(interactions[["set", "comparator", "kernel_degradation", "comparator_degradation", "interaction_comp_minus_kernel", "ci95_lo", "ci95_hi", "p_holm", "significant_expected"]]), "",
        "## Deviations", "", "None. The degree-2 NVAR uses the preregistered selected AR lag window with linear coordinates and their diagonal quadratic powers.", "",
        "The gate concerns held-out benchmark segments, not patients; Bonn segment-to-subject mapping is unavailable and no subject-level generalization is claimed.", "",
    ]
    report_path = RESULTS / "gate_report.md"
    rendered = "\n".join(lines)
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                "gate_report.md is frozen; regenerated content differs. Preserve it and write any new analysis elsewhere."
            )
        print("gate_report.md frozen and unchanged")
    else:
        report_path.write_text(rendered, encoding="utf-8")
    print(f"GATE VERDICT: {verdict}")
    for failure in failures:
        print(f"FAILED CONDITION: {failure}")
    print("wrote gate_nrmse_curves.csv, gate_interactions.csv, useful_horizon.csv, gate_report.md")


if __name__ == "__main__":
    main()
