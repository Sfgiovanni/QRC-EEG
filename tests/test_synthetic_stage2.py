"""Guardrails for the frozen Rota A Stage 2 synthetic battery."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_stage2", ROOT / "scripts/run_synthetic_stage2.py")
STAGE2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STAGE2)


def config() -> dict:
    return json.loads((ROOT / "config/rotaA_stage2_frozen.json").read_text())


def test_frozen_protocol_and_config_hashes() -> None:
    entries = (ROOT / "results/synth/stage2_protocol_frozen.sha256").read_text().splitlines()
    for entry in entries:
        expected, relative = entry.split(maxsplit=1)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_process_generation_is_reproducible_finite_and_distinct() -> None:
    generated = []
    for spec in config()["scenarios"]:
        first = STAGE2.generate_one(spec, 3, 256)
        second = STAGE2.generate_one(spec, 3, 256)
        assert first.shape == (256,)
        assert np.isfinite(first).all()
        assert np.array_equal(first, second)
        generated.append(first)
    assert not np.allclose(generated[0], generated[-1])


def test_phase_surrogate_preserves_fourier_magnitudes() -> None:
    cfg = config()
    source = next(spec for spec in cfg["scenarios"] if spec["name"] == "nonlinear_ar1_phi085")
    surrogate = next(spec for spec in cfg["scenarios"] if spec["family"] == "phase_surrogate")
    x = STAGE2.generate_one(source, 2, cfg["segment_length"])
    y = STAGE2.generate_one(surrogate, 2, cfg["segment_length"])
    assert np.allclose(np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(y)), rtol=1e-12, atol=1e-12)
    assert not np.allclose(x, y)


def test_scaling_is_fit_on_training_segments_only() -> None:
    cfg = config()
    spec = cfg["scenarios"][0]
    train, validation, test = STAGE2.scenario_data(cfg, spec)
    assert train.shape == (cfg["split"]["train"], cfg["segment_length"])
    assert validation.shape[0] == cfg["split"]["validation"]
    assert test.shape[0] == cfg["split"]["test"]
    assert abs(float(train.mean())) < 1e-12
    assert abs(float(train.std()) - 1) < 1e-12


def test_causal_convolution_has_no_pre_response() -> None:
    signal = np.zeros((1, 20)); signal[0, 7] = 1
    impulse = np.column_stack([np.array([1.0, 0.5, 0.25])])
    output = STAGE2.causal_convolution(signal, impulse)
    assert output.shape == (1, 20, 1)
    assert np.allclose(output[0, :7, 0], 0, atol=1e-14)
    assert np.allclose(output[0, 7:10, 0], impulse[:, 0], atol=1e-14)


def test_committed_models_and_official_gate1_hp_are_loaded() -> None:
    cfg, hp, commit = STAGE2.load_configuration()
    assert commit == cfg["gate1_commit"]
    assert list(hp) == cfg["models"]
    assert hp["single_kernel"] == {"K": 15, "r": 0.7, "past_mass": 0.3}


def test_completed_outputs_and_mechanical_verdict_are_consistent() -> None:
    metadata_path = ROOT / "results/synth/stage2_metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text())
    stats = metadata["statistics"]
    rule_verdict = "SUPPORTED" if (
        stats["aggregate_spearman_ci_low"] > 0
        and stats["median_within_scenario_spearman"] >= 0.5
        and stats["best_model_match_fraction"] >= 0.6
    ) else ("PARTIAL" if stats["aggregate_spearman"] > 0 and stats["median_within_scenario_spearman"] > 0
            else "NOT_SUPPORTED")
    assert stats["verdict"] == rule_verdict
    combined = __import__("pandas").read_csv(ROOT / "results/synth/theory_predictions_vs_measured.csv")
    assert len(combined) == len(config()["scenarios"]) * len(config()["models"])
    assert np.isfinite(combined.select_dtypes(include=["number"])).all().all()
    assert stats["scenario_spearman"]["ar1_phi030"] < 0
    assert stats["scenario_spearman"]["nonlinear_ar1_phi085"] < 0
