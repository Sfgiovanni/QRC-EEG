#!/usr/bin/env python3
"""Compute SHA256 of every generated CSV/figure and write
results/eeg/PROVENANCE.md + provenance/eeg_checksums.txt.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_GLOBS = [
    "results/eeg/*.csv",
    "results/eeg/raw/*.csv",
    "results/eeg/bugfix_diff_report.md",
    "results/eeg/fase1_diff_report.md",
    "results/eeg/gate_report.md",
    "results/eeg/gate_report_frozen.sha256",
    "results/eeg/effective_kernel_protocol_frozen.sha256",
    "results/eeg/effective_kernel_symbolic.txt",
    "results/eeg/theory_vs_sim_metadata.json",
    "results/eeg/theory_vs_sim_responses.npz",
    "results/eeg/gate3_report.md",
    "results/eeg/gate3_metadata.json",
    "results/eeg/shot_baseline_reproduction_status.json",
    "results/eeg/_invalid_config_r09_snapshot/*",
    "results/synth/*.csv",
    "results/synth/*.json",
    "results/synth/*.md",
    "results/synth/*.sha256",
    "results/resources/*",
    "results/final/*",
    "results/ARTIFACT_INDEX.csv",
    "results/eeg/run_gate.log",
    "results/eeg/_prefix_snapshot_gate/*",
    "results/eeg/_prefix_snapshot_gate/raw/*.csv",
    "results/eeg/_prefix_snapshot_hpselect/*",
    "results/eeg/_prefix_snapshot_hpselect/raw/*.csv",
    "figures/eeg/*.pdf",
    "figures/eeg/*.png",
    "figures/synth/*.pdf",
    "figures/synth/*.png",
    "figures/final/*.pdf",
    "figures/final/*.png",
    "figures/final/README.md",
    "paper/*.tex",
    "data/eeg/CHECKSUMS.txt",
    "data/eeg/splits/*.json",
    "docs/eeg_gate_preregistration.md",
    "docs/rotaA_plan.md",
    "docs/PROVENANCE.md",
    "docs/effective_kernel_check_protocol.md",
    "docs/effective_kernel_theory.md",
    "docs/synthetic_stage2_protocol.md",
    "docs/gate2_postgate_addendum.md",
    "docs/gate3_protocol.md",
    "docs/physical_resources.md",
    "docs/gates/*.md",
    "docs/README.md",
    "docs/claims_registry.md",
    "docs/final_results_summary.md",
    "docs/figure_inventory.md",
    "docs/table_inventory.md",
    "docs/paper_blueprint.md",
    "docs/evidence_map.md",
    "docs/limitations.md",
    "docs/reviewer_risk_register.md",
    "docs/data_availability.md",
    "docs/code_availability.md",
    "docs/reproducibility.md",
    "docs/repository_structure.md",
    "config/effective_kernel_gate1_frozen.json",
    "config/rotaA_stage2_frozen.json",
    "config/rotaA_gate3_frozen.json",
    "src/qrc_eeg/classical_baselines.py",
    "src/qrc_eeg/physical_resources.py",
    "scripts/run_gate_baselines.py",
    "scripts/make_gate_report.py",
    "scripts/update_results_gate.py",
    "scripts/verify_gate.py",
    "scripts/make_useful_horizon_v2.py",
    "scripts/run_rotaA_stage0.sh",
    "scripts/verify_rotaA_gate0.py",
    "scripts/run_effective_kernel_theory_check.py",
    "scripts/run_effective_kernel_check.py",
    "scripts/run_rotaA_stage1.sh",
    "scripts/verify_rotaA_gate1.py",
    "scripts/run_synthetic_stage2.py",
    "scripts/run_rotaA_stage2.sh",
    "scripts/verify_rotaA_gate2.py",
    "scripts/make_gate2_postgate_addendum.py",
    "scripts/make_physical_resource_table.py",
    "scripts/run_shot_sensitivity.py",
    "scripts/run_rotaA_stage3.sh",
    "scripts/verify_gate3.py",
    "scripts/build_repository_release.py",
    "scripts/make_final_figures.py",
    "scripts/run_repository_release.sh",
    "scripts/verify_repository_quick.sh",
    "scripts/verify_repository_release.py",
    "tests/test_gate_controls.py",
    "tests/test_effective_kernel_check.py",
    "tests/test_synthetic_stage2.py",
    "tests/test_gate3_resources_shots.py",
    "tests/test_repository_release.py",
    "README.md",
    "RELEASE_CHECKLIST.md",
    "CHANGELOG.md",
    "requirements-lock.txt",
    "results/README.md",
    "results/eeg/README.md",
    "results/synth/README.md",
    "results/resources/README.md",
    "figures/README.md",
    "scripts/README.md",
    "config/README.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries = []
    for pattern in OUTPUT_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                entries.append((path.relative_to(ROOT), sha256_file(path)))

    provenance_dir = ROOT / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    checksums_path = provenance_dir / "eeg_checksums.txt"
    with checksums_path.open("w") as f:
        for rel_path, digest in entries:
            f.write(f"{digest}  {rel_path}\n")

    md_lines = [
        "# EEG Study Provenance",
        "",
        "Every number in the EEG tables/figures traces to a script + source file below.",
        "",
        "## Pipeline (run in order by scripts/run_eeg.sh)",
        "",
        "1. `scripts/fetch_eeg.py` -> `data/eeg/sets/{Z,F,S}/*.txt`, `data/eeg/CHECKSUMS.txt`",
        "2. `scripts/run_sanity_checks.py` -> pytest mechanism suite (must pass to proceed)",
        "3. `scripts/run_hp_search.py`, `run_esn66_hp_search.py` -> segment-blocked HP selections",
        "4. `scripts/run_holdout_eval.py`, `run_esn66_holdout.py` -> eight-horizon held-out rows",
        "5. `scripts/run_gate_baselines.py` -> blocked-validation persistence/AR/NVAR2/tapped controls",
        "6. `scripts/run_quadratic_capacity.py`, `run_statistics.py` -> capacity and legacy contrasts",
        "7. `scripts/make_gate_report.py` -> frozen eeg_gate interactions, curves, useful horizons and verdict",
        "8. `scripts/update_results_gate.py` -> CSV-derived narrative",
        "9. `scripts/verify_gate.py` -> final fail-high gate and these checksums",
        "10. Rota A Stage 0: `make_useful_horizon_v2.py`, `update_results_gate.py`, `verify_rotaA_gate0.py`; stops before theory",
        "11. Rota A Stage 1: `run_effective_kernel_check.py`, `verify_rotaA_gate1.py`; committed r=0.7 HP, corrected tangent/separable check, post-gate amplitude sweep, then stop before synthetic validation",
        "12. Rota A Stage 2: freeze `H_actual` predictions, run the nonlinear synthetic battery, write the mechanical validation verdict, and stop before resources/shots/manuscript",
        "13. Rota A Stage 3: frozen Gate 2 analytical addendum, physical-resource audit, exact-baseline reproduction, finite-shot readout sensitivity, mechanical verifier, then stop before manuscript",
        "",
        "## File checksums (SHA256)",
        "",
        "```",
    ]
    for rel_path, digest in entries:
        md_lines.append(f"{digest}  {rel_path}")
    md_lines.append("```")

    (ROOT / "results" / "eeg" / "PROVENANCE.md").write_text("\n".join(md_lines) + "\n")
    print(f"wrote {checksums_path} ({len(entries)} files)")
    print("wrote results/eeg/PROVENANCE.md")


if __name__ == "__main__":
    main()
