"""Gate 3 resource, shot-noise, split and provenance guardrails."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qrc_eeg.physical_resources import (
    buffer_resource_counts, conservative_measurement_counts, operation_counts,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("shot_sensitivity", ROOT / "scripts/run_shot_sensitivity.py")
SHOTS = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(SHOTS)


def test_buffer_state_parameter_and_byte_formulas() -> None:
    k0 = buffer_resource_counts(4, 0)
    k15 = buffer_resource_counts(4, 15)
    assert k0["buffer_states"] == 1 and k15["buffer_states"] == 16
    assert k15["independent_real_parameters"] == 16 * (16**2 - 1)
    assert k15["conservative_real_scalars"] == 16 * 16**2
    assert k0["dense_buffer_bytes"] == np.empty((1, 16, 16), dtype=np.complex128).nbytes
    assert k15["dense_buffer_bytes"] == np.empty((16, 16, 16), dtype=np.complex128).nbytes
    assert k15["dense_buffer_bytes"] == 64 * 1024


def test_operation_and_measurement_counts() -> None:
    ops = operation_counts(4, 15, 66, 4097)
    assert ops["mix_complex_scalar_multiplies_per_step"] == 16 * 16**2
    measure = conservative_measurement_counts(66, 1000, 4097)
    assert measure["preparations_per_step"] == 66000
    assert measure["preparations_per_trajectory"] == 4097 * 66000


def test_binomial_pauli_mean_variance_and_exact_limit() -> None:
    mu, nshots, repetitions = 0.37, 300, 200000
    values = SHOTS.pauli_shot_sample(np.full(repetitions, mu), nshots, np.random.default_rng(71))
    assert abs(values.mean() - mu) < 0.003
    expected = (1 - mu**2) / nshots
    assert abs(values.var() - expected) / expected < 0.03
    coarse = SHOTS.pauli_shot_sample(np.full(50000, mu), 100, np.random.default_rng(2))
    fine = SHOTS.pauli_shot_sample(np.full(50000, mu), 100000, np.random.default_rng(2))
    assert np.mean((fine - mu) ** 2) < np.mean((coarse - mu) ** 2)


def test_noise_is_deterministic_and_does_not_create_intercept() -> None:
    features = np.array([[[-0.5, 0.2], [0.1, 0.9]]])
    first = SHOTS.pauli_shot_sample(features, 1000, SHOTS.stable_rng("test", 1))
    second = SHOTS.pauli_shot_sample(features, 1000, SHOTS.stable_rng("test", 1))
    assert np.array_equal(first, second)
    assert first.shape == features.shape  # intercept is added later by fit_readout, never noised


def test_frozen_splits_are_disjoint_and_all_test_segments_retained() -> None:
    cfg, _, _, _, splits, _ = SHOTS.preflight()
    for set_name in cfg["sets"]:
        train, val, test = map(set, (splits[set_name][part] for part in ("train", "val", "test")))
        assert not train & val and not train & test and not val & test
        assert len(test) == 20


def test_classification_is_mechanical() -> None:
    levels = pd.DataFrame({"shots": [100, 1000], "global_pass": [False, True]})
    assert SHOTS.classify_shot_sensitivity(levels, False) == "ROBUST_AT_1000_SHOTS"
    levels.global_pass = False
    assert SHOTS.classify_shot_sensitivity(levels, True) == "MIXED_SHOT_SENSITIVITY"
    assert SHOTS.classify_shot_sensitivity(levels, False) == "SHOT_SENSITIVE_UP_TO_10000"


def test_gate3_freeze_hashes_and_baseline_when_available() -> None:
    for line in (ROOT / "results/resources/gate3_protocol_frozen.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    status = ROOT / "results/eeg/shot_baseline_reproduction_status.json"
    if status.exists():
        data = json.loads(status.read_text())
        assert data["status"] == "PASS"
        assert data["max_abs_nrmse_difference"] <= data["tolerance"]


def test_generated_shot_outputs_are_complete_when_available() -> None:
    raw_path = ROOT / "results/eeg/shot_sensitivity_raw.csv"
    if not raw_path.exists():
        return
    cfg = json.loads((ROOT / "config/rotaA_gate3_frozen.json").read_text())
    raw = pd.read_csv(raw_path, low_memory=False)
    exact = len(cfg["sets"]) * len(cfg["main_models"]) * len(cfg["channel_seeds"]) * len(cfg["horizons"]) * 20
    assert len(raw) == exact * (1 + len(cfg["shots"]) * cfg["noise_replicates"])
    assert np.isfinite(raw[["nrmse", "rmse", "r2", "mae", "nrmse_exact",
                            "nrmse_difference", "relative_nrmse_inflation"]]).all().all()
    assert set(raw[raw.shots > 0].shots) == set(cfg["shots"])


def test_generated_classification_and_contrasts_when_available() -> None:
    strata_path = ROOT / "results/eeg/shot_sensitivity_strata_classification.csv"
    if not strata_path.exists():
        return
    cfg = json.loads((ROOT / "config/rotaA_gate3_frozen.json").read_text())
    levels = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_classification.csv")
    strata = pd.read_csv(strata_path)
    metadata = json.loads((ROOT / "results/eeg/gate3_metadata.json").read_text())
    assert not levels.global_pass.any()
    assert strata.stratum_pass.any()
    assert int(strata.stratum_pass.sum()) == metadata["passing_strata"] == 66
    assert metadata["scientific_classification"] == "MIXED_SHOT_SENSITIVITY"
    contrasts = pd.read_csv(ROOT / "results/eeg/shot_sensitivity_contrasts.csv")
    principal = contrasts[(contrasts.shots > 0) & contrasts["set"].isin(["F", "Z"])]
    assert principal.sign_preserved.all()
    s_null = contrasts[(contrasts.shots == 0) & (contrasts["set"] == "S")
                       & (contrasts.comparator == "QRC_K0")].iloc[0]
    assert s_null.ci95_low <= 0 <= s_null.ci95_high
