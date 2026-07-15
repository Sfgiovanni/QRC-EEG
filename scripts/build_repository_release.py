#!/usr/bin/env python3
"""Build Stage 4 repository-only scientific indexes and canonical CSVs.

This script never runs a reservoir and never reads or writes a LaTeX file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from importlib import metadata as package_metadata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"
GATES = ROOT / "docs/gates"
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def source_line(path: str) -> str:
    target = ROOT / path
    return f"`{path}` (SHA256 `{sha(target)}`)"


def capture_preflight() -> None:
    path = FINAL / "repository_preflight.json"
    if path.exists():
        return
    versions = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "sympy", "pytest", "qrc-eeg"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    record = {
        "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
        "git_commit": COMMIT,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "working_tree_status": subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).splitlines(),
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "dependencies": versions,
        "tex_sha256_before_stage4": {
            str(path.relative_to(ROOT)): sha(path)
            for path in sorted((ROOT / "paper").glob("*.tex"))
        },
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def build_shot_decomposition() -> None:
    summary = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_summary.csv")
    strata = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_strata_classification.csv")
    contrasts = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_contrasts.csv")
    raw = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_raw.csv", low_memory=False)
    by_stratum = summary.merge(strata, on=["shots", "set", "horizon"], validate="many_to_one")
    by_stratum["stratum_pass_fraction_within_shot"] = by_stratum.groupby("shots").stratum_pass.transform("mean")
    by_stratum["principal_sign_fraction_definition"] = "F/Z kernel-vs-K0 and kernel-vs-AB only; S reported as null"
    by_stratum.to_csv(ROOT / "results/eeg/shot_sensitivity_by_stratum.csv", index=False)

    finite = raw[raw.shots > 0].copy()
    rows: list[dict] = []
    scopes = [("global", [])]
    scopes += [("set", ["set"]), ("model", ["model"]), ("horizon", ["horizon"])]
    for scope, columns in scopes:
        grouped = [((), finite)] if not columns else finite.groupby(columns[0] if len(columns) == 1 else columns)
        for key, group in grouped:
            keys = key if isinstance(key, tuple) else (key,)
            values = dict(zip(columns, keys))
            for shots, slab in group.groupby("shots"):
                rows.append({"analysis": "inflation", "scope": scope, "shots": shots,
                             "set": values.get("set", "ALL"), "model": values.get("model", "ALL"),
                             "horizon": values.get("horizon", "ALL"),
                             "median_absolute_nrmse_inflation": slab.nrmse_difference.median(),
                             "mean_absolute_nrmse_inflation": slab.nrmse_difference.mean(),
                             "median_relative_nrmse_inflation": slab.relative_nrmse_inflation.median(),
                             "p90_relative_nrmse_inflation": slab.relative_nrmse_inflation.quantile(.9),
                             "n_rows": len(slab)})
    pass_counts = strata.groupby("shots").stratum_pass.agg(["sum", "count", "mean"])
    for shots, values in pass_counts.iterrows():
        rows.append({"analysis": "stratum_pass", "scope": "global", "shots": shots,
                     "set": "ALL", "model": "ALL", "horizon": "ALL",
                     "passing_strata": int(values["sum"]), "total_strata": int(values["count"]),
                     "passing_fraction": values["mean"]})
    tail_cells = by_stratum.sort_values("p90_relative_nrmse_inflation", ascending=False).head(30)
    for row in tail_cells.itertuples():
        rows.append({"analysis": "tail_cell", "scope": "set_model_horizon", "shots": row.shots,
                     "set": row.set, "model": row.model, "horizon": row.horizon,
                     "median_absolute_nrmse_inflation": row.median_nrmse_inflation,
                     "mean_absolute_nrmse_inflation": row.mean_nrmse_inflation,
                     "median_relative_nrmse_inflation": row.median_relative_nrmse_inflation,
                     "p90_relative_nrmse_inflation": row.p90_relative_nrmse_inflation,
                     "notes": "one of 30 largest set×model×horizon P90 cells"})
    for row in contrasts.itertuples():
        magnitude_ratio = np.nan if row.shots == 0 or row.exact_interaction == 0 else row.interaction_comp_minus_kernel / row.exact_interaction
        rows.append({"analysis": "contrast", "scope": "set_comparator", "shots": row.shots,
                     "set": row.set, "model": "single_kernel", "horizon": "2_to_64",
                     "comparator": row.comparator, "interaction": row.interaction_comp_minus_kernel,
                     "interaction_change_from_exact": row.interaction_change_from_exact,
                     "magnitude_ratio_to_exact": magnitude_ratio, "sign_preserved": row.sign_preserved,
                     "notes": "sign preservation does not imply magnitude preservation"})
    pd.DataFrame(rows).to_csv(ROOT / "results/eeg/shot_sensitivity_tail_analysis.csv", index=False)


def build_claims() -> None:
    claims = [
        ("C1", "Distributed state memory acts inside feedback and changes augmented-system dynamics.", "SUPPORTED",
         "Gate 1 tangent recurrence and failed separable ansatz", "results/eeg/theory_vs_sim_check.csv", "four local response metrics", "local four-qubit configuration",
         "Local linear mechanism; not hardware realization.",
         "Distributed state memory acts inside feedback and changes the augmented-system dynamics.",
         "The kernel is only an external W(z) filter applied to K=0."),
        ("C2", "The tangent recurrence locally reproduces the simulator.", "SUPPORTED_LOCAL",
         "All frozen tangent tolerances pass", "results/eeg/theory_vs_sim_check.csv", "impulse/step/FFT/memory errors", "epsilon=1e-4",
         "Local small-signal statement only.", "The tangent recurrence reproduces the simulator locally.",
         "The linear theory explains all EEG forecasting performance."),
        ("C3", "Theory predicts between-process differences and partially recovers within-process ordering.", "SUPPORTED_WITH_LIMITS",
         "Frozen Gate 2 plus post-gate centered analysis", "results/synth/theory_predictions_vs_measured.csv;results/synth/gate2_postgate_sensitivity.csv",
         "aggregate/within correlations, 6/10 best, pairwise ordering", "10 frozen scenarios",
         "Within-scenario evidence is moderate and has explicit failures.",
         "Theory predicts process differences and partially recovers model ordering within processes.",
         "Theory universally predicts the best kernel."),
        ("C4", "Distributed memory changes horizon dependence in F and Z.", "SUPPORTED_F_Z",
         "Frozen model×horizon interaction", "results/eeg/gate_interactions.csv", "paired bootstrap CI and Holm p", "F and Z segments",
         "Single Bonn database; segment-level; h=64 lacks absolute skill.",
         "Distributed memory alters error dependence on horizon in F and Z.",
         "There is quantum advantage or universal forecasting superiority."),
        ("C5", "S is null in the primary causal test.", "NULL",
         "Kernel-vs-K0 CI crosses zero", "results/eeg/gate_interactions.csv", "interaction and Holm p", "S segments",
         "Null is a valid result and remains visible.", "S yielded a null result in the primary causal test.",
         "The effect was demonstrated in all three sets."),
        ("C6", "Distributed-memory class differs from discrete delay in some comparisons.", "PARTIAL",
         "EEG interactions and tied distributed shapes", "results/eeg/gate_nrmse_curves.csv;results/eeg/gate_interactions.csv",
         "degradation curves", "F/Z/S segments", "Exponential, triangular and uniform shapes overlap.",
         "Results favor distributed memory over discrete delay in part of the comparisons.",
         "The exponential form is universally superior to triangular and uniform forms."),
        ("C7", "Finite-shot sensitivity is heterogeneous; principal F/Z signs persist.", "MIXED_SHOT_SENSITIVITY",
         "Gate 3 finite-shot analysis", "results/eeg/shot_sensitivity_classification.csv;results/eeg/shot_sensitivity_contrasts.csv",
         "median, P90, strata and contrast signs", "F/Z principal; S reported null",
         "Large tail; readout sampling only; not hardware.",
         "Shot robustness is heterogeneous and principal F/Z signs are qualitatively preserved.",
         "The method is globally shot-robust or hardware-ready."),
    ]
    columns = ["id", "claim", "status", "evidence", "artifacts", "metrics", "sets_scope", "limitations",
               "permitted_wording", "prohibited_wording"]
    write_csv(FINAL / "claims_registry.csv", [dict(zip(columns, row)) for row in claims], columns)
    lines = ["# Claims registry", "", "This registry is normative for future writing. It does not assert quantum advantage.", ""]
    for claim in claims:
        item = dict(zip(columns, claim))
        lines += [f"## {item['id']} — {item['status']}", "", item["claim"], "",
                  f"- Permitted: “{item['permitted_wording']}”", f"- Prohibited: “{item['prohibited_wording']}”",
                  f"- Evidence: `{item['artifacts']}`", f"- Limitation: {item['limitations']}", ""]
    (ROOT / "docs/claims_registry.md").write_text("\n".join(lines))


def key_row(identifier, value, ci_low=np.nan, ci_high=np.nan, unit="", population="ALL", horizon="",
            model="", comparator="", source="", gate="", status="confirmatory", note=""):
    return {"identifier": identifier, "value": value, "ci_low": ci_low, "ci_high": ci_high,
            "unit": unit, "population_set": population, "horizon": horizon, "model": model,
            "comparator": comparator, "source": source, "gate": gate, "status": status, "note": note}


def build_key_results() -> None:
    rows = []
    theory = pd.read_csv(ROOT / "results/eeg/theory_vs_sim_check.csv")
    for _, row in theory.iterrows():
        rows.append(key_row(f"gate1_{row['theory']}_{row['metric']}", row["value"], unit="relative_error",
                            population="local_channel", model="single_kernel", source="results/eeg/theory_vs_sim_check.csv",
                            gate="Gate 1", note=f"tolerance={row['tolerance']}; pass={row['pass']}"))
    gate1meta = json.loads((ROOT / "results/eeg/theory_vs_sim_metadata.json").read_text())
    rows.append(key_row("gate1_companion_spectral_radius", gate1meta["companion"]["spectral_radius"], unit="spectral_radius",
                        population="local_channel", model="single_kernel", source="results/eeg/theory_vs_sim_metadata.json", gate="Gate 1"))
    interactions = pd.read_csv(ROOT / "results/eeg/gate_interactions.csv")
    for set_name in ("F", "Z", "S"):
        row = interactions[(interactions["set"] == set_name) & (interactions.comparator == "QRC_K0")].iloc[0]
        rows.append(key_row(f"eeg_{set_name}_kernel_vs_k0_interaction", row.interaction_comp_minus_kernel,
                            row.ci95_lo, row.ci95_hi, "delta_NRMSE_degradation", set_name,
                            f"{int(row.h_short)}_to_{int(row.h_long)}", "single_kernel", "QRC_K0",
                            "results/eeg/gate_interactions.csv", "Gate 0", note=f"Holm p={row.p_holm}"))
    useful = pd.read_csv(ROOT / "results/eeg/useful_horizon_v2.csv")
    for set_name in ("F", "Z", "S"):
        row = useful[(useful["set"] == set_name) & (useful.construction == "single_kernel")].iloc[0]
        rows.append(key_row(f"eeg_{set_name}_kernel_useful_horizon", row.useful_horizon, unit="samples",
                            population=set_name, horizon=row.useful_horizon, model="single_kernel",
                            source="results/eeg/useful_horizon_v2.csv", gate="Gate 0"))
    stage2 = json.loads((ROOT / "results/synth/stage2_metadata.json").read_text())["statistics"]
    rows += [
        key_row("gate2_aggregate_spearman", stage2["aggregate_spearman"], stage2["aggregate_spearman_ci_low"],
                stage2["aggregate_spearman_ci_high"], "Spearman_rho", "10_scenarios", source="results/synth/stage2_metadata.json", gate="Gate 2"),
        key_row("gate2_median_within_scenario_spearman", stage2["median_within_scenario_spearman"], unit="Spearman_rho",
                population="10_scenarios", source="results/synth/stage2_metadata.json", gate="Gate 2"),
        key_row("gate2_best_model_match", stage2["best_model_match_fraction"], unit="fraction",
                population="10_scenarios", source="results/synth/stage2_metadata.json", gate="Gate 2"),
    ]
    post = pd.read_csv(ROOT / "results/synth/gate2_postgate_sensitivity.csv").set_index(["analysis", "scenario", "metric"]).value
    for identifier, key in {
        "gate2_centered_pearson": ("centered_correlation", "ALL", "pearson_r"),
        "gate2_centered_spearman": ("centered_correlation", "ALL", "spearman_rho"),
        "gate2_pairwise_accuracy": ("pairwise_accuracy", "ALL", "accuracy_100_pairs"),
        "gate2_top2_match": ("top2_match", "ALL", "mean"),
    }.items():
        rows.append(key_row(identifier, post[key], unit="coefficient_or_fraction", population="10_scenarios",
                            source="results/synth/gate2_postgate_sensitivity.csv", gate="Gate 2", status="post_gate_diagnostic"))
    levels = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_classification.csv")
    for row in levels.itertuples():
        rows.append(key_row(f"gate3_shots_{row.shots}_median_relative_inflation", row.median, unit="fraction",
                            population="all_shot_cells", horizon="all", source="results/eeg/shot_sensitivity_classification.csv", gate="Gate 3"))
        rows.append(key_row(f"gate3_shots_{row.shots}_p90_relative_inflation", row.p90, unit="fraction",
                            population="all_shot_cells", horizon="all", source="results/eeg/shot_sensitivity_classification.csv", gate="Gate 3"))
    strata = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_strata_classification.csv")
    rows.append(key_row("gate3_passing_strata", int(strata.stratum_pass.sum()), unit="count", population="120_set_horizon_cells",
                        source="results/eeg/shot_sensitivity_strata_classification.csv", gate="Gate 3"))
    out = pd.DataFrame(rows)
    out.to_csv(FINAL / "key_results.csv", index=False)
    (FINAL / "key_results.json").write_text(json.dumps(out.replace({np.nan: None}).to_dict("records"), indent=2) + "\n")
    lines = ["# Canonical results summary", "", "Generated from the canonical CSV/JSON sources; values are not manually transcribed.", "",
             f"- Gate 1: `{gate1meta['automatic_verdict']}`; companion radius `{gate1meta['companion']['spectral_radius']:.6f}`.",
             f"- Gate 2: `SUPPORTED`; aggregate Spearman `{stage2['aggregate_spearman']:.6f}`, median within-scenario `{stage2['median_within_scenario_spearman']:.2f}`, best-model match `{stage2['best_model_match_fraction']:.0%}`.",
             f"- Gate 3: `MIXED_SHOT_SENSITIVITY`; `{int(strata.stratum_pass.sum())}/{len(strata)}` set×horizon strata pass, with no globally passing shot level.",
             "- EEG: the primary interaction is supported in F and Z; S is null. All models exceed mean NRMSE 1 at h=64.", "",
             "Machine-readable values: `results/final/key_results.csv` and `results/final/key_results.json`."]
    (ROOT / "docs/final_results_summary.md").write_text("\n".join(lines) + "\n")


def build_tables() -> None:
    pd.read_csv(ROOT / "results/eeg/gate_interactions.csv").to_csv(FINAL / "table_eeg_interactions.csv", index=False)
    pd.read_csv(ROOT / "results/eeg/useful_horizon_v2.csv").to_csv(FINAL / "table_useful_horizon.csv", index=False)
    pd.read_csv(ROOT / "results/synth/theory_predictions_vs_measured.csv").to_csv(FINAL / "table_synthetic_validation.csv", index=False)
    pd.read_csv(ROOT / "results/resources/qrc_resource_table.csv").to_csv(FINAL / "table_physical_resources.csv", index=False)
    pd.read_csv(ROOT / "results/eeg/shot_sensitivity_by_stratum.csv").to_csv(FINAL / "table_shot_sensitivity.csv", index=False)
    negative = []
    gate1 = pd.read_csv(ROOT / "results/eeg/theory_vs_sim_check.csv")
    for row in gate1[gate1.theory == "separable_W_times_R"].itertuples():
        negative.append({"result": "separable_factorization_failed", "scope": row.metric, "value": row.value,
                         "ci_low": np.nan, "ci_high": np.nan, "source": "results/eeg/theory_vs_sim_check.csv", "interpretation": "valid falsification"})
    eeg = pd.read_csv(ROOT / "results/eeg/gate_interactions.csv")
    row = eeg[(eeg["set"] == "S") & (eeg.comparator == "QRC_K0")].iloc[0]
    negative.append({"result": "S_primary_causal_null", "scope": "S", "value": row.interaction_comp_minus_kernel,
                     "ci_low": row.ci95_lo, "ci_high": row.ci95_hi, "source": "results/eeg/gate_interactions.csv", "interpretation": "null result"})
    stage2 = json.loads((ROOT / "results/synth/stage2_metadata.json").read_text())["statistics"]["scenario_spearman"]
    for scenario, value in stage2.items():
        if value < 0:
            negative.append({"result": "negative_within_scenario_ordering", "scope": scenario, "value": value,
                             "ci_low": np.nan, "ci_high": np.nan, "source": "results/synth/stage2_metadata.json", "interpretation": "explicit theory-ordering failure"})
    levels = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_classification.csv")
    for row in levels.itertuples():
        negative.append({"result": "no_global_shot_robustness", "scope": f"N={row.shots}", "value": row.p90,
                         "ci_low": np.nan, "ci_high": np.nan, "source": "results/eeg/shot_sensitivity_classification.csv", "interpretation": "global criterion failed"})
    write_csv(FINAL / "table_negative_null_results.csv", negative)


def build_gate_summaries() -> None:
    interactions = pd.read_csv(ROOT / "results/eeg/gate_interactions.csv")
    gate1 = pd.read_csv(ROOT / "results/eeg/theory_vs_sim_check.csv")
    gate1meta = json.loads((ROOT / "results/eeg/theory_vs_sim_metadata.json").read_text())
    gate2meta = json.loads((ROOT / "results/synth/stage2_metadata.json").read_text())
    gate3meta = json.loads((ROOT / "results/eeg/gate3_metadata.json").read_text())
    common = f"Commit: `{COMMIT}`. Status: **canonical**."
    fz = interactions[(interactions["set"].isin(["F", "Z"])) & (interactions.comparator == "QRC_K0")]
    s = interactions[(interactions["set"] == "S") & (interactions.comparator == "QRC_K0")].iloc[0]
    gate0 = f"""# Gate 0 — corrected EEG pipeline

## Scientific question

Does distributed state history change the model-by-horizon degradation curve under causal preprocessing, segment-blocked HP selection and held-out segment evaluation?

## Configuration and inputs

Frozen Bonn Z/F/S splits; h={{1,2,4,8,16,32,64,128}}; 10 channel seeds; segment is the statistical unit. Inputs: {source_line('config/eeg_frozen.yaml')}, {source_line('results/eeg/gate_interactions.csv')}.

## Scripts and artifacts

`scripts/run_eeg.sh`, `scripts/make_gate_report.py`, `scripts/make_useful_horizon_v2.py`; canonical outputs are `gate_interactions.csv`, `gate_nrmse_curves.csv`, `useful_horizon_v2.csv`, `gate_report.md` and `RESULTS.md`.

## Metrics and verdicts

Primary metric: comparator-minus-kernel change in NRMSE degradation from h=2 to h=64, paired over 20 test segments with bootstrap CI and Holm correction. F/Z kernel-vs-K0 conditions pass ({', '.join(f'{r.set}={r.interaction_comp_minus_kernel:.6f}' for r in fz.itertuples())}); S is null (`{s.interaction_comp_minus_kernel:.6f}`, CI [{s.ci95_lo:.6f}, {s.ci95_hi:.6f}]). Technical verdict: **PASS**. Scientific verdict: horizon dependence supported in F/Z, null in S.

## Limitations

One Bonn database; randomized/unavailable subject mapping; no subject-disjoint or clinical generalization; h=64 is an interaction endpoint with mean NRMSE>1, not an absolute-skill headline. {common}
"""
    tangent = gate1[gate1.theory == "tangent_recurrence"]
    separable = gate1[gate1.theory == "separable_W_times_R"]
    gate1doc = f"""# Gate 1 — effective-kernel mechanism

## Scientific question

Does the implementation-faithful tangent recurrence reproduce the local nonlinear response, and is the external factorization `H_sep=W_K R` valid?

## Configuration and inputs

Committed `single_kernel`: K={int(tangent.K.iloc[0])}, r={tangent.r.iloc[0]}, past mass={tangent.past_mass.iloc[0]}, seed={int(tangent.seed.iloc[0])}, epsilon={tangent.epsilon.iloc[0]}. Inputs: {source_line('results/eeg/theory_vs_sim_check.csv')} and {source_line('results/eeg/theory_vs_sim_responses.npz')}.

## Scripts, metrics and artifacts

`scripts/run_effective_kernel_check.py`; impulse, step and FFT relative Frobenius errors plus memory-energy L1. Tangent: {int(tangent['pass'].sum())}/{len(tangent)} metrics pass. Separable ansatz: {int(separable['pass'].sum())}/{len(separable)} pass. Companion spectral radius: {gate1meta['companion']['spectral_radius']:.6f} (locally stable).

Technical verdict: **PASS**. Scientific verdict: **{gate1meta['automatic_verdict']}**. The correct local transfer is `H_actual(z)=C[zI-AW_K(z)]^-1B`.

## Limitations and status

Local small-signal theory only; no universal T_eff law or complete EEG explanation. The r=0.9 snapshot is `INVALID_CONFIG`, exploratory only. {common}
"""
    stats = gate2meta["statistics"]
    gate2doc = f"""# Gate 2 — synthetic validation

## Scientific question

Do predictions frozen from `H_actual` anticipate degradation differences across and within synthetic processes?

## Inputs, scripts and artifacts

Ten frozen scenarios and five models; `scripts/run_synthetic_stage2.py` and analytical-only `scripts/make_gate2_postgate_addendum.py`. Sources: {source_line('results/synth/theory_predictions_vs_measured.csv')} and {source_line('results/synth/gate2_postgate_sensitivity.csv')}.

## Metrics and verdicts

Aggregate Spearman={stats['aggregate_spearman']:.6f} (95% CI [{stats['aggregate_spearman_ci_low']:.6f}, {stats['aggregate_spearman_ci_high']:.6f}]); median within-scenario Spearman={stats['median_within_scenario_spearman']:.2f}; predicted best matches {int(stats['best_model_match_fraction']*10)}/10. Technical verdict: **PASS**. Scientific/mechanical verdict: **{stats['verdict']}**.

## Limitations

Aggregate association partly reflects between-process scale. Within-process ordering is moderate; ar1_phi030 and nonlinear_ar1_phi085 are explicit negative-ordering failures. No universal T_eff-to-slope law. The post-gate addendum does not alter the freeze. {common}
"""
    levels = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_classification.csv")
    strata = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_strata_classification.csv")
    resource = pd.read_csv(ROOT / "results/resources/qrc_resource_table.csv").set_index("construction")
    gate3doc = f"""# Gate 3 — resources and finite-shot sensitivity

## Scientific question

What are the simulator/measurement resources and how heterogeneous is finite-shot readout sensitivity?

## Configuration, inputs and scripts

QRC K0, AB-noaux and single-kernel; exact plus N={{100,300,1000,3000,10000}}, 10 noise replicates, seeds 1–3, Z/F/S, eight horizons. `scripts/run_shot_sensitivity.py`, `scripts/make_physical_resource_table.py`. Sources: {source_line('results/eeg/shot_sensitivity_raw.csv')} and {source_line('results/resources/qrc_resource_table.csv')}.

## Metrics and resources

Median/P90 relative NRMSE inflation, absolute inflation, set×horizon strata, interaction sign and magnitude. No shot level passes globally; {int(strata.stratum_pass.sum())}/{len(strata)} strata pass. K0/AB/K15 buffers use {int(resource.loc['QRC_K0','dense_buffer_bytes'])}/{int(resource.loc['AB_noaux','dense_buffer_bytes'])}/{int(resource.loc['single_kernel','dense_buffer_bytes'])} bytes. There are 66 conservative measurement groups.

Technical verdict: **{gate3meta['technical_verdict']}**. Scientific verdict: **{gate3meta['scientific_classification']}**.

## Limitations

Principal sign fraction covers only preregistered F/Z contrasts; S is reported as null. Sign preservation is not magnitude preservation. h=64 is a sensitivity endpoint, not a skill headline. Shot noise omits decoherence, preparation error, drift, full backaction and physical history storage. {common}
"""
    for number, text in enumerate((gate0, gate1doc, gate2doc, gate3doc)):
        (GATES / f"gate{number}_summary.md").write_text(text)
    (GATES / "README.md").write_text("""# Gate audit index

The four summaries are generated from canonical sources by `scripts/build_repository_release.py`.

- [Gate 0](gate0_summary.md): corrected causal EEG pipeline and frozen interaction.
- [Gate 1](gate1_summary.md): local mechanism and failed separable factorization.
- [Gate 2](gate2_summary.md): frozen synthetic validation plus analytical addendum.
- [Gate 3](gate3_summary.md): physical resources and mixed shot sensitivity.

Historical snapshots and invalid configurations are preserved but are not canonical evidence.
Machine-readable classification is in `results/ARTIFACT_INDEX.csv`.
""")


def classify_path(relative: str) -> tuple[str, str, str, bool, str, str]:
    if "_invalid_config_r09_snapshot" in relative:
        return "Gate 1", "snapshot", "INVALID_CONFIG", False, "historical", "Exploratory r=0.9; never confirmatory"
    if "_prefix_snapshot" in relative:
        return "Gate 0", "snapshot", "snapshot", False, "historical", "Preserved pre-change comparison"
    if relative.startswith("results/final/") or relative.startswith("figures/final/") or relative.startswith("docs/gates/"):
        return "Stage 4", "derived_release", "canonical", True, "scripts/build_repository_release.py", "Derived from frozen sources"
    mapping = [
        ("theory_vs_sim", "Gate 1", "confirmatory"), ("effective_kernel", "Gate 1", "protocol_or_theory"),
        ("results/synth/", "Gate 2", "synthetic_result"), ("figures/synth/", "Gate 2", "figure"),
        ("shot_", "Gate 3", "shot_result"), ("gate3", "Gate 3", "report_or_metadata"),
        ("results/resources/", "Gate 3", "resource_result"), ("physical_resources", "Gate 3", "resource_doc"),
        ("gate_", "Gate 0", "eeg_gate_result"), ("useful_horizon_v2", "Gate 0", "eeg_gate_result"),
    ]
    for token, gate, kind in mapping:
        if token in relative:
            return gate, kind, "canonical", True, "see provenance", "Frozen or canonical gate artifact"
    if any(token in relative for token in ("bugfix", "fase1", "overnight", "run.log")):
        return "Legacy", "legacy", "legacy", False, "historical", "Retained for audit; not final evidence"
    return "Repository", "support", "derived", False, "various", "Supporting artifact"


def build_artifact_index() -> None:
    patterns = ("results/**/*", "figures/**/*", "docs/**/*", "config/*", "scripts/*.py", "scripts/*.sh", "src/qrc_eeg/*.py", "tests/*.py")
    excluded = {ROOT / "results/ARTIFACT_INDEX.csv", FINAL / "repository_release_report.md",
                ROOT / "results/eeg/PROVENANCE.md"}
    files = sorted({path for pattern in patterns for path in ROOT.glob(pattern)
                    if path.is_file() and path.suffix != ".tex" and path not in excluded})
    rows = []
    for path in files:
        relative = str(path.relative_to(ROOT))
        gate, kind, status, canonical, generator, note = classify_path(relative)
        rows.append({"artifact_path": relative, "gate": gate, "artifact_type": kind, "status": status,
                     "generator_script": generator, "source_inputs": "see docs/gates and provenance",
                     "git_commit": COMMIT, "sha256": sha(path), "canonical": canonical, "notes": note})
    write_csv(ROOT / "results/ARTIFACT_INDEX.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("data", "index", "all"), default="all", nargs="?")
    args = parser.parse_args()
    FINAL.mkdir(parents=True, exist_ok=True); GATES.mkdir(parents=True, exist_ok=True)
    if args.phase in ("data", "all"):
        capture_preflight()
        build_shot_decomposition()
        build_claims()
        build_key_results()
        build_tables()
        build_gate_summaries()
    if args.phase in ("index", "all"):
        build_artifact_index()
    print(f"repository release derivatives built ({args.phase}); no simulations or .tex writes")


if __name__ == "__main__":
    main()
