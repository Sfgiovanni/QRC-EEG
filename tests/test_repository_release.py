"""Repository-only Stage 4 guardrails."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tex_files_match_stage4_preflight() -> None:
    preflight = json.loads((ROOT / "results/final/repository_preflight.json").read_text())
    before = preflight["tex_sha256_before_stage4"]
    current = sorted(str(path.relative_to(ROOT)) for path in ROOT.rglob("*.tex") if ".git" not in path.parts)
    assert sorted(before) == current
    assert all(digest(ROOT / relative) == expected for relative, expected in before.items())


def test_final_tables_match_canonical_sources() -> None:
    pairs = (
        ("table_eeg_interactions.csv", "results/eeg/gate_interactions.csv"),
        ("table_useful_horizon.csv", "results/eeg/useful_horizon_v2.csv"),
        ("table_synthetic_validation.csv", "results/synth/theory_predictions_vs_measured.csv"),
        ("table_physical_resources.csv", "results/resources/qrc_resource_table.csv"),
        ("table_shot_sensitivity.csv", "results/eeg/shot_sensitivity_by_stratum.csv"),
    )
    for final, source in pairs:
        pd.testing.assert_frame_equal(pd.read_csv(ROOT / "results/final" / final), pd.read_csv(ROOT / source),
                                      check_dtype=False)


def test_frozen_verdicts_and_claim_ids() -> None:
    assert json.loads((ROOT / "results/eeg/theory_vs_sim_metadata.json").read_text())["automatic_verdict"] == "FAIL_SEPARABLE_FACTORIZATION"
    assert json.loads((ROOT / "results/synth/stage2_metadata.json").read_text())["statistics"]["verdict"] == "SUPPORTED"
    gate3 = json.loads((ROOT / "results/eeg/gate3_metadata.json").read_text())
    assert gate3["technical_verdict"] == "COMPLETE"
    assert gate3["scientific_classification"] == "MIXED_SHOT_SENSITIVITY"
    assert set(pd.read_csv(ROOT / "results/final/claims_registry.csv").id) == {f"C{i}" for i in range(1, 8)}


def test_r09_is_invalid_and_not_canonical() -> None:
    assert "INVALID_CONFIG" in (ROOT / "results/eeg/_invalid_config_r09_snapshot/README.md").read_text()
    index = pd.read_csv(ROOT / "results/ARTIFACT_INDEX.csv")
    rows = index[index.artifact_path.str.contains("_invalid_config_r09_snapshot")]
    assert len(rows) and set(rows.status) == {"INVALID_CONFIG"} and not rows.canonical.astype(bool).any()


def test_release_scripts_do_not_write_tex() -> None:
    for name in ("build_repository_release.py", "make_final_figures.py", "run_repository_release.sh"):
        text = (ROOT / "scripts" / name).read_text()
        assert "paper/tab_" not in text
        assert "manuscript.tex" not in text
