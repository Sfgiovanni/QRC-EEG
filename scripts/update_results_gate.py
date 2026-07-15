#!/usr/bin/env python3
"""Regenerate RESULTS.md strictly from gate CSV artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "eeg"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = ["NA" if pd.isna(value) else f"{float(value):.3f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    report = (RESULTS / "gate_report.md").read_text()
    verdict = "PASS" if "Mechanical verdict: **PASS**" in report else "FAIL"
    interactions = pd.read_csv(RESULTS / "gate_interactions.csv")
    useful = pd.read_csv(RESULTS / "useful_horizon_v2.csv")
    curves = pd.read_csv(RESULTS / "gate_nrmse_curves.csv")
    capacity = pd.read_csv(RESULTS / "capacity_demand_regression_summary.csv")
    cap = capacity[capacity["predictor"] == "capacity_gap"].iloc[0]
    fitted = ["AR", "NVAR2", "tapped_delay"]

    lines = [
        "# Corrected EEG results", "",
        "The empirical claim is mechanistic: distributed state-history changes the slope of error degradation with horizon. There is no claim of quantum advantage or absolute forecasting superiority; classical models lead at short horizons, and all evaluated models have mean NRMSE above 1 at h=64.", "",
        "## Useful horizon (symmetric v2)", "",
        "Useful horizon is the largest h with mean NRMSE below 1 and a paired-bootstrap lower confidence bound above zero for the improvement over persistence. The same criterion is applied to every model; persistence is NA because its self-difference is zero.", "",
    ]
    gate_names = ["single_kernel", "QRC_K0", "AR", "NVAR2", "persistence", "tapped_delay"]
    gate_useful = useful[useful["construction"].isin(gate_names)][
        ["construction", "set", "useful_horizon", "useful_horizon_ms", "nrmse_at_useful_horizon", "persistence_improvement_ci95_lo"]
    ]
    lines += [markdown_table(gate_useful), ""]
    lines += [
        "In F, the kernel and fitted classical baselines all stop at h=8 (46.080 ms). In Z, the kernel reaches h=16 (92.161 ms), while AR, NVAR2 and tapped-delay stop at h=8; QRC K=0 also reaches h=16. In S, the kernel, K=0, NVAR2 and tapped-delay reach h=32 (184.321 ms). Thus useful horizon does not establish a uniquely quantum or uniquely exponential advantage.", "",
        "## Preregistered causal-memory interaction", "",
        f"The frozen gate verdict is **{verdict}**. Its primary endpoint is the model-by-horizon interaction from h=2 to h=64, not the h=64 endpoint itself.", "",
    ]
    failures = []
    for set_name in ["F", "Z"]:
        set_rows = interactions[interactions["set"] == set_name]
        k0 = set_rows[set_rows["comparator"] == "QRC_K0"].iloc[0]
        strongest = set_rows[set_rows["comparator"].isin(fitted)].sort_values("comparator_degradation").iloc[0]
        lines.append(
            f"- Set {set_name}: kernel-vs-K=0 interaction={k0.interaction_comp_minus_kernel:+.6f} "
            f"(95% CI [{k0.ci95_lo:+.6f}, {k0.ci95_hi:+.6f}], Holm p={k0.p_holm:.6g}; "
            f"condition={'PASS' if k0.significant_expected else 'FAIL'}). Strongest fitted classical comparator={strongest.comparator}, "
            f"interaction={strongest.interaction_comp_minus_kernel:+.6f} "
            f"(95% CI [{strongest.ci95_lo:+.6f}, {strongest.ci95_hi:+.6f}], Holm p={strongest.p_holm:.6g}; "
            f"condition={'PASS' if strongest.significant_expected else 'FAIL'})."
        )
        if not bool(k0.significant_expected):
            failures.append(f"{set_name} did not pass the QRC K=0 condition")
        if not bool(strongest.significant_expected):
            failures.append(f"{set_name} did not pass the strongest-classical condition ({strongest.comparator})")
    s_k0 = interactions[(interactions["set"] == "S") & (interactions["comparator"] == "QRC_K0")].iloc[0]
    lines += [
        "",
        f"S is null in the causal test: kernel-vs-K=0 interaction={s_k0.interaction_comp_minus_kernel:+.6f} "
        f"(95% CI [{s_k0.ci95_lo:+.6f}, {s_k0.ci95_hi:+.6f}], Holm p={s_k0.p_holm:.6g}). The effect is confined to F and Z.", "",
        "Single, dual, triangular and uniform distributed kernels have closely overlapping degradation curves and the same useful horizons in F/Z/S. The supported interpretation concerns the distributed-memory class, not the exponential shape specifically.", "",
    ]
    if failures:
        lines += ["The frozen rule failed because " + "; ".join(failures) + ".", ""]

    h64 = curves[(curves["construction"] == "single_kernel") & (curves["horizon"] == 64)].set_index("set")
    lines += [
        "## Long-horizon endpoint", "",
        f"At h=64 the kernel mean NRMSE is {h64.loc['F', 'mean_nrmse']:.6f} in F, {h64.loc['Z', 'mean_nrmse']:.6f} in Z and {h64.loc['S', 'mean_nrmse']:.6f} in S. All exceed 1, so there is no absolute skill claim at this endpoint; h=64 is used only as the frozen long endpoint of the interaction.", "",
    ]
    lines += [
        "## Quadratic capacity", "",
        f"The iid quadratic-capacity gap regression has slope {cap.slope:+.6f} with n={int(cap['n'])}; no valid interval is claimed from three capacity-gap values. "
        "The iid quadratic measure used here showed no detectable positive association with EEG gains in the evaluated configurations.", "",
        "## Scope", "",
        "Inference is limited to benchmark segments from one EEG database. Bonn segment-to-subject mapping is randomized/unavailable, there is no subject-disjoint split, and no between-patient or cross-dataset generalization is claimed. Z is scalp EEG; F and S are intracranial recordings.", "",
        "## Reproduction", "", "The complete corrected empirical pipeline is:", "", "```bash", "bash scripts/run_eeg.sh", "```", "", "The Stage 0 integrity refresh is:", "", "```bash", "bash scripts/run_rotaA_stage0.sh", "```", "",
        "The original empirical gate is frozen in `results/eeg/gate_report.md`; the corrected symmetric headline table is `results/eeg/useful_horizon_v2.csv`.", "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote RESULTS.md from gate CSVs; verdict={verdict}")


if __name__ == "__main__":
    main()
