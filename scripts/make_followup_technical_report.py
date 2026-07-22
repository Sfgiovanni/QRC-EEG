#!/usr/bin/env python3
"""Assemble results/eeg/followup/technical_report.md mechanically from the
generated CSVs/JSON. Answers the eight questions of Section 7 of the task
spec with numbers and artifact paths, not prose alone.

Must be run after: run_esn_distributed_memory_hp_search.py,
run_esn_distributed_memory_holdout.py, make_classical_distributed_memory_figure.py,
run_crossed_inference.py, make_crossed_inference_figure.py, verify_esn_distributed_memory.py,
make_followup_metadata.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FOLLOWUP = ROOT / "results" / "eeg" / "followup"
CONTROL = FOLLOWUP / "classical_control"
CROSSED = FOLLOWUP / "crossed_inference"


def md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: "NA" if pd.isna(v) else float_fmt.format(v))
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def question_1_k0_reproduces_esn66() -> str:
    """K0 reproduces the existing ESN-66? Compare fixed_core/ESN66_K0 against
    the existing ESN_66 held-out raw CSV, cell by cell."""

    followup = pd.read_csv(FOLLOWUP / "raw" / "esn_distributed_memory_holdout_by_segment_seed.csv")
    k0 = followup[(followup.construction == "ESN66_K0") & (followup.analysis_mode == "fixed_core")]
    esn66 = pd.read_csv(ROOT / "results/eeg/raw/eeg_holdout_esn66_by_segment_seed.csv")
    merged = k0.merge(esn66, on=["set", "horizon", "seed", "segment_id"], suffixes=("_k0", "_esn66"))
    if merged.empty:
        return "**Could not compare** -- no overlapping cells between ESN66_K0/fixed_core and ESN_66."
    max_err = float(np.max(np.abs(merged["nrmse_k0"] - merged["nrmse_esn66"])))
    n = len(merged)
    verdict = "YES" if max_err < 1e-8 else "NO -- see docs/classical_distributed_memory_protocol.md Deviations"
    return (
        f"**{verdict}.** Compared {n} matched (set, horizon, seed, segment) cells between "
        f"`ESN66_K0`/`fixed_core` and the existing committed `ESN_66` arm "
        f"(`results/eeg/raw/eeg_holdout_esn66_by_segment_seed.csv`). Max absolute NRMSE "
        f"difference = `{max_err:.3e}` (unit test tolerance `tests/test_esn_distributed_memory.py::"
        f"test_k0_reproduces_existing_esn_implementation` uses `1e-10` on the underlying feature "
        f"trajectories directly; this end-to-end comparison additionally passes through the ridge "
        f"readout fit, hence the slightly looser but still effectively-exact tolerance here)."
    )


def question_2_and_3_curve_shape_and_ab() -> str:
    curves = pd.read_csv(CONTROL / "tab_classical_distributed_memory.csv")
    lines = []
    for mode in ("fixed_core", "retuned_core"):
        for set_name in sorted(curves["set"].unique()):
            slab = curves[(curves.analysis_mode == mode) & (curves.set == set_name)]
            k0_2 = slab[(slab.construction == "ESN66_K0") & (slab.horizon == 2)]["mean_nrmse"]
            k0_64 = slab[(slab.construction == "ESN66_K0") & (slab.horizon == 64)]["mean_nrmse"]
            kernel_2 = slab[(slab.construction == "ESN66_kernel") & (slab.horizon == 2)]["mean_nrmse"]
            kernel_64 = slab[(slab.construction == "ESN66_kernel") & (slab.horizon == 64)]["mean_nrmse"]
            ab_2 = slab[(slab.construction == "ESN66_AB") & (slab.horizon == 2)]["mean_nrmse"]
            ab_64 = slab[(slab.construction == "ESN66_AB") & (slab.horizon == 64)]["mean_nrmse"]
            if k0_2.empty or kernel_2.empty or ab_2.empty:
                continue
            d_k0 = float(k0_64.iloc[0] - k0_2.iloc[0])
            d_kernel = float(kernel_64.iloc[0] - kernel_2.iloc[0])
            d_ab = float(ab_64.iloc[0] - ab_2.iloc[0])
            lines.append(
                f"- {mode}/{set_name}: D(K0)={d_k0:+.4f}, D(AB)={d_ab:+.4f}, D(kernel)={d_kernel:+.4f} "
                f"(kernel {'<' if d_kernel < d_k0 else '>='} K0: {'slower degradation' if d_kernel < d_k0 else 'not slower'}; "
                f"kernel {'<' if d_kernel < d_ab else '>='} AB: {'beats concentrated delay' if d_kernel < d_ab else 'does not beat concentrated delay'})"
            )
    return "\n".join(lines)


def question_4_fixed_vs_retuned() -> str:
    selected = json.loads((CONTROL / "hp_selected.json").read_text())
    fixed = json.loads((CONTROL / "fixed_core_hp.json").read_text())
    rows = []
    for c in ("ESN66_K0", "ESN66_AB", "ESN66_kernel"):
        rows.append({
            "construction": c,
            "fixed_core_esn_hp": json.dumps(fixed[c]["esn_hp"]),
            "retuned_core_esn_hp": json.dumps(selected[c]["esn_hp"]),
            "same_hp": fixed[c]["esn_hp"] == selected[c]["esn_hp"],
        })
    df = pd.DataFrame(rows)
    all_same = bool(df["same_hp"].all())
    verdict = (
        "All three arms' independently retuned core HP matched the fixed-core (existing ESN-66) HP "
        "exactly, so the fixed_core and retuned_core analyses coincide in this run -- the conclusion "
        "does not depend on which mode is used."
        if all_same else
        "At least one arm's retuned core HP differs from the fixed-core HP; fixed_core and retuned_core "
        "conclusions must be compared explicitly (see the aggregated table) rather than assumed equal."
    )
    return f"{verdict}\n\n{md_table(df)}"


def question_5_original_survives_crossed() -> str:
    boot = pd.read_csv(CROSSED / "crossed_bootstrap.csv")
    orig = pd.read_csv(CROSSED / "original_style_replication.csv")
    merged = boot.merge(
        orig, on=["set", "kernel", "comparator", "analysis_mode"], suffixes=("_crossed", "_orig")
    )
    merged["orig_significant"] = merged["ci95_lo_orig"] > 0
    merged["crossed_significant"] = merged["significant_expected"]
    weakened = merged[(merged["orig_significant"]) & (~merged["crossed_significant"])]
    stable = merged[(merged["orig_significant"]) & (merged["crossed_significant"])]
    n_orig_sig = int(merged["orig_significant"].sum())
    lines = [
        f"Of {len(merged)} tests in the `eeg_followup_crossed_sensitivity` family, "
        f"{n_orig_sig} were significant in the original-style (seed-averaged) replication at the raw "
        f"95% CI level; of those, {len(stable)} remained significant under the crossed "
        f"bootstrap (Holm-adjusted) and {len(weakened)} did not.",
        "",
    ]
    if len(weakened):
        lines.append("Weakened under the crossed design:")
        lines.append(md_table(weakened[["set", "kernel", "comparator", "analysis_mode", "bootstrap_mean_crossed", "ci95_lo_crossed", "ci95_hi_crossed", "p_holm"]]))
    return "\n".join(lines)


def question_6_mixed_model() -> str:
    diagnostics = json.loads((CROSSED / "mixed_model_diagnostics.json").read_text())
    n = len(diagnostics)
    n_converged = sum(1 for d in diagnostics if d.get("converged"))
    n_boundary = sum(1 for d in diagnostics if d.get("boundary_hit"))
    n_singular = sum(1 for d in diagnostics if d.get("singular"))
    n_error = sum(1 for d in diagnostics if d.get("error"))

    mixed = pd.read_csv(CROSSED / "mixed_model_results.csv")
    boot = pd.read_csv(CROSSED / "crossed_bootstrap.csv")
    merged = mixed.merge(boot, on=["set", "kernel", "comparator", "analysis_mode"], suffixes=("_mixed", "_boot"))
    same_sign = merged[np.isfinite(merged["interaction_comp_minus_kernel"])]
    agree = int((np.sign(same_sign["interaction_comp_minus_kernel"]) == np.sign(same_sign["bootstrap_mean"])).sum())

    return (
        f"{n_converged}/{n} cells converged; {n_boundary}/{n} flagged a variance-component boundary "
        f"(near-zero segment or seed variance, or a solver boundary warning); {n_singular}/{n} had a "
        f"non-finite/singular covariance matrix; {n_error}/{n} raised an exception during fitting "
        f"(all diagnostics recorded verbatim, never hidden, in `mixed_model_diagnostics.json`). "
        f"Where a finite point estimate was obtained, its sign agreed with the crossed bootstrap's "
        f"sign in {agree}/{len(same_sign)} cells. Per the frozen protocol, the crossed bootstrap "
        f"(Section 3) remains the primary sensitivity analysis regardless of mixed-model convergence; "
        f"statsmodels' crossed-random-effects variance-components approximation is used because no "
        f"R/lme4 is available in this environment (documented in "
        f"`docs/crossed_inference_protocol.md` Section 5)."
    )


def question_7_claims_to_revise() -> str:
    boot = pd.read_csv(CROSSED / "crossed_bootstrap.csv")
    weakened = boot[~boot["significant_expected"]]
    if weakened.empty:
        return "No test in the `eeg_followup_crossed_sensitivity` family failed its expected-direction/CI/Holm condition; no existing claim requires downgrading on this basis alone."
    rows = weakened[["set", "kernel", "comparator", "analysis_mode", "bootstrap_mean", "ci95_lo", "ci95_hi", "p_holm"]]
    return (
        f"{len(weakened)}/{len(boot)} tests did not meet the expected-direction + CI-excludes-zero + "
        f"Holm-p<0.05 condition under the crossed bootstrap:\n\n{md_table(rows)}\n\n"
        f"Per `docs/crossed_inference_protocol.md` Section 9, any canonical claim resting on these "
        f"specific cells should be reported with reduced strength; claims C1-C7 in "
        f"`docs/claims_registry.md` are otherwise based on the canonical seed-averaged gate analysis, "
        f"which is untouched by this sensitivity check."
    )


def main() -> None:
    sections = [
        "# Follow-up technical report: classical distributed-memory ESN control and crossed segment x seed inference",
        "",
        "Additive follow-up to the canonical QRC-EEG repository. Does not modify any canonical "
        "Gate/Gate1B/Gate2/Gate3 artifact, `docs/claims_registry.md`, or `results/eeg/gate_interactions.csv`. "
        "Frozen protocol: `docs/classical_distributed_memory_protocol.md`, "
        "`docs/crossed_inference_protocol.md`, `config/esn_distributed_memory_frozen.yaml` "
        "(hashes in `results/eeg/followup/PROTOCOL_HASHES.sha256`).",
        "",
        "## 1. Does K0 reproduce the existing ESN-66?",
        "",
        question_1_k0_reproduces_esn66(),
        "",
        "## 2-3. Does distributed memory change the degradation curve in the ESN, and does it beat concentrated delay?",
        "",
        "`D(construction) = NRMSE(h=64) - NRMSE(h=2)`, per mode/set (lower magnitude = slower degradation = the direction expected of the mechanism if it is generic, not substrate-specific).",
        "",
        question_2_and_3_curve_shape_and_ab(),
        "",
        "## 4. Does the result depend on fixed vs. retuned core HP?",
        "",
        question_4_fixed_vs_retuned(),
        "",
        "## 5. Do the original F/Z/S conclusions survive the segment x seed bootstrap?",
        "",
        question_5_original_survives_crossed(),
        "",
        "## 6. Does the mixed model converge and agree with the bootstrap?",
        "",
        question_6_mixed_model(),
        "",
        "## 7. Does any paper claim need to be reduced or reformulated?",
        "",
        question_7_claims_to_revise(),
        "",
        "## 8. Files feeding the new paper tables/figures",
        "",
        "- `results/eeg/followup/classical_control/tab_classical_distributed_memory.csv` -- aggregated NRMSE by construction x mode x set x horizon.",
        "- `results/eeg/followup/classical_control/tab_resource_accounting.csv` -- parameter/memory/op accounting.",
        "- `results/eeg/followup/crossed_inference/crossed_bootstrap.csv` -- primary sensitivity endpoint, 21-test family, Holm-corrected.",
        "- `results/eeg/followup/crossed_inference/original_style_replication.csv` -- side-by-side canonical-style replication (comparison only).",
        "- `results/eeg/followup/crossed_inference/mixed_model_results.csv` / `mixed_model_diagnostics.json` -- secondary verification + diagnostics.",
        "- `figures/eeg/fig_classical_distributed_memory.{pdf,png}`.",
        "- `figures/eeg/fig_crossed_inference.{pdf,png}`.",
        "- `results/eeg/followup/metadata.json` -- commit/dependency/OS/timing/hash provenance.",
        "",
        "## Reproduction",
        "",
        "```bash",
        ".venv/bin/python scripts/run_esn_distributed_memory_hp_search.py",
        ".venv/bin/python scripts/run_esn_distributed_memory_holdout.py",
        ".venv/bin/python scripts/make_classical_distributed_memory_figure.py",
        ".venv/bin/python scripts/run_crossed_inference.py",
        ".venv/bin/python scripts/make_crossed_inference_figure.py",
        ".venv/bin/python scripts/verify_esn_distributed_memory.py",
        ".venv/bin/python scripts/make_followup_metadata.py",
        ".venv/bin/python -m pytest tests/test_esn_distributed_memory.py tests/test_crossed_inference.py -q",
        "```",
    ]
    report = "\n".join(str(s) for s in sections)
    (FOLLOWUP / "technical_report.md").write_text(report)
    print("wrote", FOLLOWUP / "technical_report.md")


if __name__ == "__main__":
    main()
