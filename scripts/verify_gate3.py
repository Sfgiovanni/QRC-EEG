#!/usr/bin/env python3
"""Fail-high verification for the frozen Rota A Gate 3."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/eeg"


def fail(message: str) -> None:
    raise SystemExit(f"ROTA A GATE 3 FAILED: {message}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(path: Path) -> None:
    for line in path.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        target = ROOT / relative
        if not target.is_file() or sha(target) != expected:
            fail(f"frozen hash mismatch: {relative}")


def check_configuration() -> tuple[dict, dict]:
    check_manifest(ROOT / "results/resources/gate3_protocol_frozen.sha256")
    check_manifest(ROOT / "results/synth/stage2_protocol_frozen.sha256")
    check_manifest(ROOT / "results/synth/stage2_predictions_frozen.sha256")
    cfg = json.loads((ROOT / "config/rotaA_gate3_frozen.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != cfg["git_commit"]:
        fail("HEAD differs from the frozen Gate 3 commit")
    selected = json.loads(subprocess.check_output(
        ["git", "show", f"{commit}:results/eeg/hp_selected.json"], cwd=ROOT, text=True))
    hp = {"QRC_K0": {}, **{name: selected[name]["hp"] for name in cfg["resource_models"] if name != "QRC_K0"}}
    digest = hashlib.sha256(json.dumps(hp, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != cfg["official_hp_sha256"]:
        fail("committed HP blob differs from frozen official mapping")
    for relative, expected in cfg["reference_sha256"].items():
        if sha(ROOT / relative) != expected:
            fail(f"baseline/split reference changed: {relative}")
    gate2 = json.loads((ROOT / "results/synth/stage2_metadata.json").read_text())
    if gate2["statistics"]["verdict"] != "SUPPORTED":
        fail("frozen Gate 2 verdict changed")
    metadata = json.loads((RESULTS / "gate3_metadata.json").read_text())
    if metadata["git_commit"] != commit or metadata["configuration"] != cfg:
        fail("Gate 3 metadata commit/configuration differs")
    return cfg, metadata


def check_gate2_addendum() -> None:
    addendum = pd.read_csv(ROOT / "results/synth/gate2_postgate_sensitivity.csv")
    lookup = addendum.set_index(["analysis", "scenario", "metric"]).value
    expected = {
        ("centered_correlation", "ALL", "pearson_r"): 0.9608914509,
        ("centered_correlation", "ALL", "spearman_rho"): 0.7617286915,
        ("pairwise_accuracy", "ALL", "accuracy_100_pairs"): 0.68,
        ("top2_match", "ALL", "mean"): 0.8,
        ("ci_tie_match", "ALL", "mean"): 0.9,
    }
    for key, value in expected.items():
        if key not in lookup or not np.isclose(lookup[key], value, atol=1e-10):
            fail(f"Gate 2 analytical addendum mismatch: {key}")
    text = (ROOT / "docs/gate2_postgate_addendum.md").read_text()
    for marker in ("MECHANICAL_GATE2_VERDICT = SUPPORTED", "does not alter", "within-scenario evidence"):
        if marker not in text:
            fail(f"Gate 2 addendum lacks required statement: {marker}")


def check_resources(cfg: dict) -> None:
    table = pd.read_csv(ROOT / "results/resources/qrc_resource_table.csv").set_index("construction")
    if set(table.index) != set(cfg["resource_models"]):
        fail("resource constructions differ from freeze")
    for model, row in table.iterrows():
        d, states = 2 ** int(row.n_qubits), int(row.K) + 1
        expected = {
            "dimension": d,
            "buffer_states": states,
            "independent_real_parameters": states * (d * d - 1),
            "conservative_real_scalars": states * d * d,
            "dense_buffer_bytes": states * d * d * np.dtype(np.complex128).itemsize,
            "preparations_per_step_at_10000": cfg["observables"] * 10000,
            "preparations_per_trajectory_at_10000": cfg["observables"] * 10000 * cfg["trajectory_length_for_resource_table"],
        }
        for column, value in expected.items():
            if int(row[column]) != value:
                fail(f"resource formula mismatch: {model}/{column}")
    if int(table.loc["QRC_K0", "dense_buffer_bytes"]) != 4096:
        fail("K0 buffer must be 4096 bytes")
    if int(table.loc["single_kernel", "dense_buffer_bytes"]) != 65536:
        fail("K15 buffer must be 65536 bytes")


def check_baseline() -> None:
    status = json.loads((RESULTS / "shot_baseline_reproduction_status.json").read_text())
    if status["status"] != "PASS" or status["matched_rows"] != status["expected_rows"]:
        fail("exact baseline reproduction did not pass every available row")
    if status["max_abs_nrmse_difference"] > status["tolerance"]:
        fail("exact baseline reproduction exceeds tolerance")
    rows = pd.read_csv(RESULTS / "shot_baseline_reproduction.csv")
    if len(rows) != status["expected_rows"] or rows.abs_nrmse_difference.max() > status["tolerance"]:
        fail("baseline reproduction CSV differs from status")


def check_shots(cfg: dict, metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RESULTS / "shot_sensitivity_raw.csv", low_memory=False)
    expected_exact = len(cfg["sets"]) * len(cfg["main_models"]) * len(cfg["channel_seeds"]) * len(cfg["horizons"]) * 20
    expected_finite = expected_exact * len(cfg["shots"]) * cfg["noise_replicates"]
    if len(raw) != expected_exact + expected_finite:
        fail(f"raw row count {len(raw)} != {expected_exact + expected_finite}")
    numeric = ["nrmse", "rmse", "r2", "mae", "nrmse_exact", "nrmse_difference", "relative_nrmse_inflation"]
    if not np.isfinite(raw[numeric]).all().all():
        fail("raw shot metrics contain non-finite values")
    if set(raw[raw.shots > 0].shots) != set(cfg["shots"]):
        fail("shot levels differ from freeze")
    for column, expected in (("set", cfg["sets"]), ("model", cfg["main_models"]),
                             ("horizon", cfg["horizons"]), ("channel_seed", cfg["channel_seeds"])):
        if set(raw[column]) != set(expected):
            fail(f"raw {column} differs from freeze")
    finite = raw[raw.shots > 0]
    if set(finite.noise_replicate.astype(int)) != set(range(1, cfg["noise_replicates"] + 1)):
        fail("noise replicates differ from freeze")
    summary = pd.read_csv(RESULTS / "shot_sensitivity_summary.csv")
    if len(summary) != len(cfg["sets"]) * len(cfg["main_models"]) * len(cfg["horizons"]) * len(cfg["shots"]):
        fail("shot summary row count differs")
    levels = pd.read_csv(RESULTS / "shot_sensitivity_classification.csv")
    for row in levels.itertuples():
        slab = finite[finite.shots == row.shots].relative_nrmse_inflation
        if not np.isclose(row.median, slab.median()) or not np.isclose(row.p90, slab.quantile(.9)):
            fail(f"shot-level summary differs at N={row.shots}")
        expected_pass = row.median <= .05 and row.p90 <= .10 and row.sign_fraction >= 1.0
        if bool(row.global_pass) != expected_pass:
            fail(f"global classification differs at N={row.shots}")
    strata = pd.read_csv(RESULTS / "shot_sensitivity_strata_classification.csv")
    if len(strata) != len(cfg["shots"]) * len(cfg["sets"]) * len(cfg["horizons"]):
        fail("set×horizon strata row count differs")
    if int(strata.stratum_pass.sum()) != metadata["passing_strata"] or len(strata) != metadata["total_strata"]:
        fail("stratum classification differs from metadata")
    expected_science = "MIXED_SHOT_SENSITIVITY" if strata.stratum_pass.any() and not levels.global_pass.any() else None
    if metadata["technical_verdict"] != "COMPLETE" or metadata["scientific_classification"] != expected_science:
        fail("technical/scientific Gate 3 classification differs")
    return raw, levels


def check_contrasts(cfg: dict) -> None:
    table = pd.read_csv(RESULTS / "shot_sensitivity_contrasts.csv")
    expected = (len(cfg["shots"]) + 1) * len(cfg["sets"]) * 2
    if len(table) != expected:
        fail("contrast row count differs")
    principal = table[(table.shots > 0) & table["set"].isin(["F", "Z"])]
    if not principal.sign_preserved.all():
        fail("a principal frozen contrast sign was not preserved")
    s_exact = table[(table.shots == 0) & (table["set"] == "S") & (table.comparator == "QRC_K0")].iloc[0]
    if not (s_exact.ci95_low <= 0 <= s_exact.ci95_high):
        fail("required S-null causal contrast is no longer visible")


def check_docs_and_hashes(metadata: dict) -> None:
    for relative, expected in metadata["artifact_sha256"].items():
        target = ROOT / relative
        if not target.is_file() or sha(target) != expected:
            fail(f"metadata artifact hash mismatch: {relative}")
    physical = (ROOT / "docs/physical_resources.md").read_text().lower()
    for marker in ("cannot be copied", "measurement backaction", "state-preparation error",
                   "drift", "not physical gate counts", "not an immediate hardware"):
        if marker not in physical:
            fail(f"physical-resource limitations missing: {marker}")
    report = (RESULTS / "gate3_report.md").read_text()
    for marker in ("Technical verdict: COMPLETE", "MIXED_SHOT_SENSITIVITY", "66 of 120",
                   "not a real hardware execution", "Gate 3 stops here"):
        if marker not in report:
            fail(f"Gate 3 report lacks required marker: {marker}")


def finalize_provenance() -> None:
    if subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode:
        fail("pytest failed")
    if subprocess.run([sys.executable, "scripts/make_provenance.py"], cwd=ROOT).returncode:
        fail("provenance generation failed")
    checksums = (ROOT / "provenance/eeg_checksums.txt").read_text()
    required = (
        "results/eeg/gate3_report.md", "results/eeg/gate3_metadata.json",
        "results/eeg/shot_sensitivity_raw.csv", "results/eeg/shot_sensitivity_strata_classification.csv",
        "results/resources/qrc_resource_table.csv", "results/synth/gate2_postgate_sensitivity.csv",
        "figures/eeg/fig_shot_sensitivity.pdf", "figures/synth/fig_gate2_within_scenario.pdf",
        "docs/gate3_protocol.md", "docs/physical_resources.md", "scripts/verify_gate3.py",
        "tests/test_gate3_resources_shots.py",
    )
    for relative in required:
        if relative not in checksums:
            fail(f"SHA256 missing from project provenance: {relative}")
    for line in checksums.splitlines():
        digest, relative = line.split(maxsplit=1)
        if sha(ROOT / relative) != digest:
            fail(f"project checksum mismatch: {relative}")


def main() -> None:
    cfg, metadata = check_configuration()
    check_gate2_addendum()
    check_resources(cfg)
    check_baseline()
    _, levels = check_shots(cfg, metadata)
    check_contrasts(cfg)
    check_docs_and_hashes(metadata)
    finalize_provenance()
    print("\nROTA A GATE 3 VERIFICATION: PASS")
    print(f"technical={metadata['technical_verdict']}; scientific={metadata['scientific_classification']}")
    print(f"global p90 at N=10000: {levels.loc[levels.shots == 10000, 'p90'].iloc[0]:.4%}")
    print(f"passing set×horizon strata: {metadata['passing_strata']}/{metadata['total_strata']}")
    print("Gate 1 and Gate 2 verdicts preserved; no manuscript or hardware claim introduced")


if __name__ == "__main__":
    main()
