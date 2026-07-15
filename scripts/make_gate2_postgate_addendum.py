#!/usr/bin/env python3
"""Post-gate diagnostics from frozen Gate 2 CSVs only; never reruns a reservoir."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/synth"
FIGURES = ROOT / "figures/synth"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen() -> None:
    for manifest in (RESULTS / "stage2_protocol_frozen.sha256", RESULTS / "stage2_predictions_frozen.sha256"):
        for line in manifest.read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            if sha(ROOT / relative) != expected:
                raise SystemExit(f"INVALID_PROVENANCE: frozen Gate 2 hash differs: {relative}")
    metadata = json.loads((RESULTS / "stage2_metadata.json").read_text())
    if metadata["statistics"]["verdict"] != "SUPPORTED":
        raise SystemExit("INVALID_PROVENANCE: frozen Gate 2 verdict is not SUPPORTED")


def main() -> None:
    verify_frozen()
    table = pd.read_csv(RESULTS / "theory_predictions_vs_measured.csv")
    raw = pd.read_csv(RESULTS / "measured_forecasts_raw.csv")
    table["predicted_centered"] = table.predicted_slope - table.groupby("scenario").predicted_slope.transform("mean")
    table["measured_centered"] = table.measured_slope - table.groupby("scenario").measured_slope.transform("mean")
    centered_pearson = pearsonr(table.predicted_centered, table.measured_centered)
    centered_spearman = spearmanr(table.predicted_centered, table.measured_centered)
    beta = float(np.dot(table.predicted_centered, table.measured_centered) / np.dot(table.predicted_centered, table.predicted_centered))
    scenario_intercepts = {
        scenario: float(group.measured_slope.mean() - beta * group.predicted_slope.mean())
        for scenario, group in table.groupby("scenario")
    }

    rows = [
        {"analysis": "centered_correlation", "scenario": "ALL", "metric": "pearson_r", "value": centered_pearson.statistic},
        {"analysis": "centered_correlation", "scenario": "ALL", "metric": "spearman_rho", "value": centered_spearman.statistic},
        {"analysis": "scenario_fixed_effect_regression", "scenario": "ALL", "metric": "predicted_slope_coefficient", "value": beta},
    ]
    pair_rows, regret_rows = [], []
    for scenario, group in table.groupby("scenario"):
        indexed = group.set_index("model")
        correct = 0
        for first, second in itertools.combinations(sorted(group.model), 2):
            predicted_sign = np.sign(indexed.loc[first, "predicted_slope"] - indexed.loc[second, "predicted_slope"])
            measured_sign = np.sign(indexed.loc[first, "measured_slope"] - indexed.loc[second, "measured_slope"])
            hit = bool(predicted_sign == measured_sign)
            correct += hit
            pair_rows.append({"analysis": "pairwise", "scenario": scenario, "metric": f"{first}__vs__{second}", "value": float(hit)})
        rows.append({"analysis": "pairwise_accuracy", "scenario": scenario, "metric": "accuracy_10_pairs", "value": correct / 10})
        predicted_best = group.loc[group.predicted_slope.idxmin(), "model"]
        measured_order = group.sort_values("measured_slope").model.tolist()
        measured_best = measured_order[0]
        regret = float(indexed.loc[predicted_best, "measured_slope"] - indexed.loc[measured_best, "measured_slope"])
        top2 = float(predicted_best in measured_order[:2])
        # Paired bootstrap over the same seed/segment blocks, using per-block horizon slopes.
        subset = raw[(raw.scenario == scenario) & raw.model.isin([predicted_best, measured_best])]
        block_slopes = []
        for (seed, segment, model), block in subset.groupby(["seed", "test_segment", "model"]):
            block_slopes.append({"seed": seed, "segment": segment, "model": model,
                                 "slope": np.polyfit(np.log2(block.horizon), block.nrmse, 1)[0]})
        pivot = pd.DataFrame(block_slopes).pivot(index=["seed", "segment"], columns="model", values="slope")
        diff = (pivot[predicted_best] - pivot[measured_best]).to_numpy()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(scenario.encode()).digest()[:8], "little"))
        boots = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        indistinguishable = float(lo <= 0 <= hi)
        regret_rows.append({"scenario": scenario, "predicted_best": predicted_best, "measured_best": measured_best,
                            "regret": regret, "top2_match": top2, "paired_ci_low": lo,
                            "paired_ci_high": hi, "statistically_indistinguishable": indistinguishable})
        rows += [
            {"analysis": "predicted_model_regret", "scenario": scenario, "metric": "measured_slope_regret", "value": regret},
            {"analysis": "top2_match", "scenario": scenario, "metric": "predicted_winner_in_measured_top2", "value": top2},
            {"analysis": "ci_tie_match", "scenario": scenario, "metric": "predicted_winner_indistinguishable", "value": indistinguishable},
            {"analysis": "scenario_intercept", "scenario": scenario, "metric": "fixed_effect_intercept", "value": scenario_intercepts[scenario]},
        ]
    rows.extend(pair_rows)
    rows += [
        {"analysis": "pairwise_accuracy", "scenario": "ALL", "metric": "accuracy_100_pairs", "value": np.mean([r["value"] for r in pair_rows])},
        {"analysis": "top2_match", "scenario": "ALL", "metric": "mean", "value": np.mean([r["top2_match"] for r in regret_rows])},
        {"analysis": "ci_tie_match", "scenario": "ALL", "metric": "mean", "value": np.mean([r["statistically_indistinguishable"] for r in regret_rows])},
        {"analysis": "predicted_model_regret", "scenario": "ALL", "metric": "mean", "value": np.mean([r["regret"] for r in regret_rows])},
    ]
    for omitted in sorted(table.scenario.unique()):
        kept = table[table.scenario != omitted].copy()
        xp = kept.predicted_slope - kept.groupby("scenario").predicted_slope.transform("mean")
        yp = kept.measured_slope - kept.groupby("scenario").measured_slope.transform("mean")
        rows.append({"analysis": "leave_one_scenario_out", "scenario": omitted,
                     "metric": "centered_pearson_without_scenario", "value": pearsonr(xp, yp).statistic})

    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "gate2_postgate_sensitivity.csv", index=False)
    regret_frame = pd.DataFrame(regret_rows)
    pair_accuracy = output[(output.analysis == "pairwise_accuracy") & (output.scenario != "ALL")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for scenario, group in table.groupby("scenario"):
        axes[0].scatter(group.predicted_centered, group.measured_centered, s=28, label=scenario)
    axes[0].axhline(0, color="0.4", lw=0.7); axes[0].axvline(0, color="0.4", lw=0.7)
    axes[0].set(xlabel="Predicted slope, scenario-centered", ylabel="Measured slope, scenario-centered",
                title=f"Within-scenario Pearson={centered_pearson.statistic:.2f}")
    axes[1].barh(pair_accuracy.scenario, pair_accuracy.value, color="#4477AA")
    axes[1].set(xlim=(0, 1), xlabel="Correct pairwise ordering", title="10 model pairs / scenario")
    axes[2].barh(regret_frame.scenario, regret_frame.regret, color="#CC6677")
    axes[2].set(xlabel="Measured slope regret", title="Frozen predicted winner")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/synth/fig_gate2_within_scenario.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "centered_pearson": float(centered_pearson.statistic),
        "centered_spearman": float(centered_spearman.statistic), "fixed_effect_beta": beta,
        "pairwise_accuracy": float(np.mean([r["value"] for r in pair_rows])),
        "mean_regret": float(regret_frame.regret.mean()), "top2_match": float(regret_frame.top2_match.mean()),
        "ci_tie_match": float(regret_frame.statistically_indistinguishable.mean()),
        "loso_min": float(output[output.analysis == "leave_one_scenario_out"].value.min()),
        "loso_max": float(output[output.analysis == "leave_one_scenario_out"].value.max()),
    }
    doc = f"""# Gate 2 post-gate analytical addendum

`MECHANICAL_GATE2_VERDICT = SUPPORTED` remains unchanged. This addendum uses only already-generated
Gate 2 CSVs and does not rerun a reservoir, alter a scenario, ranking, criterion or frozen result.

After centering both slopes within scenario, Pearson correlation is `{summary['centered_pearson']:.6f}`
and Spearman is `{summary['centered_spearman']:.6f}`. The scenario-fixed-effect regression
`measured_slope ~ predicted_slope + scenario intercept` gives coefficient `{beta:.6f}`.
Pairwise ordering accuracy across all 100 comparisons is `{summary['pairwise_accuracy']:.3f}`;
top-2 match is `{summary['top2_match']:.3f}`. Mean measured-slope regret of the predicted winner is
`{summary['mean_regret']:.6g}`. In `{summary['ci_tie_match']:.3f}` of scenarios, the predicted winner
is statistically indistinguishable from the measured best under the paired bootstrap.

Leave-one-scenario-out centered Pearson correlations range from `{summary['loso_min']:.6f}` to
`{summary['loso_max']:.6f}`. The originally reported aggregate Spearman includes differences of
scale between processes. The within-scenario evidence is therefore more moderate, as already
indicated by the frozen median within-scenario Spearman of 0.60. This clarification does not alter
the mechanical Gate 2 verdict.
"""
    (ROOT / "docs/gate2_postgate_addendum.md").write_text(doc)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
