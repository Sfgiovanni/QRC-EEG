#!/usr/bin/env python3
"""Fail-high audit of the repository-only Stage 4 preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"
REQUIRED_VERDICTS = {
    "Gate 1": "FAIL_SEPARABLE_FACTORIZATION",
    "Gate 2": "SUPPORTED",
    "Gate 3 technical": "COMPLETE",
    "Gate 3 scientific": "MIXED_SHOT_SENSITIVITY",
}


def fail(message: str, provenance: bool = False) -> None:
    verdict = "INVALID_PROVENANCE" if provenance else "INCOMPLETE"
    raise SystemExit(f"REPOSITORY RELEASE FAILED [{verdict}]: {message}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(path: Path) -> None:
    for line in path.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        target = ROOT / relative
        if not target.is_file() or sha(target) != expected:
            fail(f"frozen manifest mismatch: {relative}", provenance=True)


def check_frozen_and_verdicts() -> None:
    for manifest in (
        ROOT / "results/eeg/gate_report_frozen.sha256",
        ROOT / "results/eeg/effective_kernel_protocol_frozen.sha256",
        ROOT / "results/synth/stage2_protocol_frozen.sha256",
        ROOT / "results/synth/stage2_predictions_frozen.sha256",
        ROOT / "results/resources/gate3_protocol_frozen.sha256",
    ):
        check_manifest(manifest)
    gate1 = json.loads((ROOT / "results/eeg/theory_vs_sim_metadata.json").read_text())
    gate2 = json.loads((ROOT / "results/synth/stage2_metadata.json").read_text())
    gate3 = json.loads((ROOT / "results/eeg/gate3_metadata.json").read_text())
    observed = {
        "Gate 1": gate1["automatic_verdict"], "Gate 2": gate2["statistics"]["verdict"],
        "Gate 3 technical": gate3["technical_verdict"],
        "Gate 3 scientific": gate3["scientific_classification"],
    }
    if observed != REQUIRED_VERDICTS:
        fail(f"frozen verdict mismatch: {observed}", provenance=True)
    if "INVALID_CONFIG" not in (ROOT / "results/eeg/_invalid_config_r09_snapshot/README.md").read_text():
        fail("r=0.9 snapshot is not explicitly INVALID_CONFIG", provenance=True)


def check_required_artifacts() -> None:
    required = [
        "results/ARTIFACT_INDEX.csv", "results/final/claims_registry.csv",
        "results/final/key_results.csv", "results/final/key_results.json",
        "results/final/table_eeg_interactions.csv", "results/final/table_useful_horizon.csv",
        "results/final/table_synthetic_validation.csv", "results/final/table_physical_resources.csv",
        "results/final/table_shot_sensitivity.csv", "results/final/table_negative_null_results.csv",
        "results/eeg/shot_sensitivity_by_stratum.csv", "results/eeg/shot_sensitivity_tail_analysis.csv",
        "docs/gates/gate0_summary.md", "docs/gates/gate1_summary.md", "docs/gates/gate2_summary.md",
        "docs/gates/gate3_summary.md", "docs/claims_registry.md", "docs/figure_inventory.md",
        "docs/table_inventory.md", "docs/paper_blueprint.md", "docs/evidence_map.md",
        "docs/limitations.md", "docs/reviewer_risk_register.md", "docs/data_availability.md",
        "docs/code_availability.md", "docs/reproducibility.md", "docs/repository_structure.md",
        "RELEASE_CHECKLIST.md", "CHANGELOG.md", "requirements-lock.txt",
    ]
    for name in required:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"missing canonical artifact: {name}")


def assert_frame_equal(derived: str, source: str) -> None:
    first, second = pd.read_csv(ROOT / derived), pd.read_csv(ROOT / source)
    try:
        pd.testing.assert_frame_equal(first, second, check_dtype=False, check_exact=False, rtol=1e-13, atol=1e-15)
    except AssertionError as error:
        fail(f"derived CSV differs: {derived} <- {source}: {error}")


def check_final_csvs() -> None:
    mappings = {
        "results/final/table_eeg_interactions.csv": "results/eeg/gate_interactions.csv",
        "results/final/table_useful_horizon.csv": "results/eeg/useful_horizon_v2.csv",
        "results/final/table_synthetic_validation.csv": "results/synth/theory_predictions_vs_measured.csv",
        "results/final/table_physical_resources.csv": "results/resources/qrc_resource_table.csv",
        "results/final/table_shot_sensitivity.csv": "results/eeg/shot_sensitivity_by_stratum.csv",
    }
    for derived, source in mappings.items():
        assert_frame_equal(derived, source)
    key = pd.read_csv(FINAL / "key_results.csv").set_index("identifier")
    interactions = pd.read_csv(ROOT / "results/eeg/gate_interactions.csv")
    for set_name in ("F", "Z", "S"):
        expected = interactions[(interactions["set"] == set_name) & (interactions.comparator == "QRC_K0")].iloc[0]
        stored = key.loc[f"eeg_{set_name}_kernel_vs_k0_interaction"]
        if not np.isclose(stored.value, expected.interaction_comp_minus_kernel) or not np.isclose(stored.ci_low, expected.ci95_lo):
            fail(f"key result mismatch for EEG {set_name}")
    levels = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_classification.csv")
    for row in levels.itertuples():
        if not np.isclose(key.loc[f"gate3_shots_{row.shots}_p90_relative_inflation", "value"], row.p90):
            fail(f"key result mismatch at {row.shots} shots")
    claims = pd.read_csv(FINAL / "claims_registry.csv")
    if set(claims.id) != {f"C{i}" for i in range(1, 8)}:
        fail("claims registry does not contain exactly C1–C7")


def png_metadata(path: Path) -> tuple[int, int, float]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        fail(f"invalid PNG signature: {path}")
    width, height = struct.unpack(">II", data[16:24])
    offset, dpi = 8, 0.0
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset+4])[0]
        kind = data[offset+4:offset+8]
        payload = data[offset+8:offset+8+length]
        if kind == b"pHYs" and len(payload) == 9 and payload[8] == 1:
            dpi = struct.unpack(">I", payload[:4])[0] * .0254
            break
        offset += 12 + length
    return width, height, dpi


def check_figures() -> None:
    names = ("fig_effective_kernel_mechanism", "fig_synthetic_theory_validation",
             "fig_eeg_horizon_dependence", "fig_resources_and_shots")
    for name in names:
        pdf, png = ROOT / f"figures/final/{name}.pdf", ROOT / f"figures/final/{name}.png"
        if not pdf.is_file() or pdf.read_bytes()[:5] != b"%PDF-" or pdf.stat().st_size < 10000:
            fail(f"missing/invalid vector PDF: {name}")
        if not png.is_file():
            fail(f"missing PNG: {name}")
        width, height, dpi = png_metadata(png)
        if width < 3000 or height < 1800 or dpi < 599:
            fail(f"PNG resolution below requirement: {name} {width}x{height} dpi={dpi:.1f}")


def check_artifact_index() -> None:
    index = pd.read_csv(ROOT / "results/ARTIFACT_INDEX.csv")
    required = {"artifact_path", "gate", "artifact_type", "status", "generator_script",
                "source_inputs", "git_commit", "sha256", "canonical", "notes"}
    if not required.issubset(index.columns) or index.artifact_path.duplicated().any():
        fail("artifact index schema or uniqueness invalid")
    for row in index.itertuples():
        target = ROOT / row.artifact_path
        if not target.is_file() or sha(target) != row.sha256:
            fail(f"artifact-index hash mismatch: {row.artifact_path}", provenance=True)
    invalid = index[index.artifact_path.str.contains("_invalid_config_r09_snapshot")]
    if invalid.empty or set(invalid.status) != {"INVALID_CONFIG"} or invalid.canonical.astype(bool).any():
        fail("r=0.9 artifact-index classification invalid", provenance=True)


def check_active_text() -> None:
    active = [ROOT / "README.md", ROOT / "RESULTS.md", ROOT / "docs/final_results_summary.md",
              ROOT / "docs/limitations.md", *(ROOT / "docs/gates").glob("gate*_summary.md")]
    text = "\n".join(path.read_text() for path in active)
    for stale in ("R^2 = 0.92", "R^2 = 0.97", "[-0.0117, 0.0013]", "n_patients"):
        if stale in text:
            fail(f"stale number/unit in active documentation: {stale}")
    for bad in ("demonstrates quantum advantage", "globally robust to shots", "effect was demonstrated in all three sets",
                "r=0.9 is confirmatory", "exponential kernel is universally superior"):
        if re.search(re.escape(bad), text, flags=re.IGNORECASE):
            fail(f"prohibited active claim: {bad}")
    required = ("FAIL_SEPARABLE_FACTORIZATION", "SUPPORTED", "MIXED_SHOT_SENSITIVITY", "S is null")
    for marker in required:
        if marker not in text:
            fail(f"active documentation omits frozen result: {marker}")


def check_tex_invariant() -> None:
    preflight = json.loads((FINAL / "repository_preflight.json").read_text())
    before = preflight["tex_sha256_before_stage4"]
    current_paths = sorted(str(path.relative_to(ROOT)) for path in ROOT.rglob("*.tex") if ".git" not in path.parts)
    if sorted(before) != current_paths:
        fail("a .tex file was created or removed during Stage 4", provenance=True)
    for relative, expected in before.items():
        if sha(ROOT / relative) != expected:
            fail(f"pre-existing .tex changed during Stage 4: {relative}", provenance=True)


def check_links() -> None:
    required_links = {
        "README.md": ["docs/gates/README.md", "results/ARTIFACT_INDEX.csv", "docs/limitations.md"],
        "docs/README.md": ["gates/README.md", "claims_registry.md", "reproducibility.md"],
    }
    for relative, links in required_links.items():
        text = (ROOT / relative).read_text()
        for link in links:
            if link not in text:
                fail(f"navigation link missing: {relative} -> {link}")


def run_full_validation() -> None:
    commands = [
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "scripts/verify_rotaA_gate0.py"],
        [sys.executable, "scripts/verify_rotaA_gate1.py", "--release-audit"],
        [sys.executable, "scripts/verify_rotaA_gate2.py", "--release-audit"],
        [sys.executable, "scripts/verify_gate3.py"],
    ]
    for command in commands:
        if subprocess.run(command, cwd=ROOT).returncode:
            fail(f"validation command failed: {' '.join(command)}")


def write_report(verdict: str, quick: bool) -> None:
    preflight = json.loads((FINAL / "repository_preflight.json").read_text())
    warnings = [
        "Authorship, affiliations, ORCIDs, funding and conflicts require human input.",
        "No DOI or immutable archive exists; CITATION.cff and .zenodo.json were not invented.",
        "The working tree contains extensive preserved changes from Gates 0–3 and Stage 4.",
        "Gate 2 within-scenario ordering is moderate; Gate 3 sensitivity is mixed; S is null.",
    ]
    lines = ["# Repository release audit", "", f"**Verdict: {verdict}.**", "",
             f"Commit: `{preflight['git_commit']}`; branch: `{preflight['branch']}`.",
             f"Mode: {'quick (no tests/gate reruns)' if quick else 'full'}.", "",
             "Frozen scientific results: Gate 1 `FAIL_SEPARABLE_FACTORIZATION`; Gate 2 `SUPPORTED`; "
             "Gate 3 `COMPLETE` / `MIXED_SHOT_SENSITIVITY`; EEG F/Z support with S-null.", "",
             "## Warnings requiring human review", ""] + [f"- {item}" for item in warnings]
    lines += ["", "No `.tex` file was created, removed or modified by Stage 4. No commit, push, release, DOI, upload or manuscript was produced.", ""]
    (FINAL / "repository_release_report.md").write_text("\n".join(lines))


def update_and_check_provenance() -> None:
    if subprocess.run([sys.executable, "scripts/make_provenance.py"], cwd=ROOT).returncode:
        fail("provenance generator failed", provenance=True)
    checksums = (ROOT / "provenance/eeg_checksums.txt").read_text().splitlines()
    mapping = {relative: digest for digest, relative in (line.split(maxsplit=1) for line in checksums)}
    required = ("results/final/repository_release_report.md", "results/ARTIFACT_INDEX.csv",
                "figures/final/fig_effective_kernel_mechanism.pdf", "docs/claims_registry.md",
                "scripts/verify_repository_release.py", "tests/test_repository_release.py")
    for relative in required:
        if relative not in mapping or sha(ROOT / relative) != mapping[relative]:
            fail(f"final SHA256 missing/mismatched: {relative}", provenance=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip pytest and historical gate verifier reruns")
    args = parser.parse_args()
    check_frozen_and_verdicts(); check_required_artifacts(); check_final_csvs(); check_figures()
    check_artifact_index(); check_active_text(); check_tex_invariant(); check_links()
    if not args.quick:
        run_full_validation()
        if subprocess.run([sys.executable, "scripts/build_repository_release.py", "index"], cwd=ROOT).returncode:
            fail("artifact index refresh failed after historical gate audits", provenance=True)
        check_artifact_index()
    verdict = "REPOSITORY_READY_WITH_WARNINGS"
    write_report(verdict, args.quick)
    update_and_check_provenance()
    print(f"\nREPOSITORY RELEASE VERIFICATION: {verdict}")
    print("frozen verdicts and canonical CSVs: consistent")
    print("four vector PDFs + four 600-dpi PNGs: PASS")
    print(".tex invariant: PASS; manuscript absent; no release action performed")


if __name__ == "__main__":
    main()
