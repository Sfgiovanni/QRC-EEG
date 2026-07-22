"""Mandatory tests for the classical distributed-memory ESN control
(docs/classical_distributed_memory_protocol.md, Section 7).

Self-contained: does not read results/eeg/hp_selected.json and does not
depend on any followup holdout run having happened yet, except for the
completeness check, which is skipped if the raw CSV is not present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from qrc_eeg.esn import ESNConfig, EchoStateNetwork
from qrc_eeg.esn_distributed_memory import (
    DistributedMemoryESNConfig,
    base_reservoir_draw,
    build_esn_reservoir_weights,
    construction_features,
    kernel_for,
    run_batched_distributed_memory_esn,
    run_sequential_distributed_memory_esn,
)
from qrc_eeg.pipeline import batched_esn_features
from qrc_eeg.state_kernels import no_memory_weights, single_exponential_weights

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP_RAW = ROOT / "results/eeg/followup/raw/esn_distributed_memory_holdout_by_segment_seed.csv"
CONFIG_PATH = ROOT / "config/esn_distributed_memory_frozen.yaml"
FROZEN_EEG_CONFIG = ROOT / "config/eeg_frozen.yaml"


def cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def eeg_cfg() -> dict:
    return yaml.safe_load(FROZEN_EEG_CONFIG.read_text())


# --- 1. ESN66_K0 reproduces the existing ESN-66 implementation --------------------------------


def test_k0_reproduces_existing_esn_implementation():
    n = 66
    hp = {"n_reservoir": n, "spectral_radius": 0.5, "input_scale": 1.0, "leak_rate": 0.7}
    seed = 3
    rng = np.random.default_rng(11)
    b, t = 4, 150
    inputs = rng.uniform(-1.0, 1.0, size=(b, t))

    existing = batched_esn_features(ESNConfig(seed=seed, **hp), inputs)

    kernel = no_memory_weights()
    config = DistributedMemoryESNConfig(seed=seed, **hp)
    new = run_batched_distributed_memory_esn(kernel, config, inputs)

    max_err = np.max(np.abs(existing - new))
    assert max_err < 1e-10, f"ESN66_K0 diverges from existing ESN-66: {max_err}"


def test_k0_matches_sequential_echostate_network_reference():
    """K0 must also match the original non-batched EchoStateNetwork class."""

    n = 20
    hp = dict(n_reservoir=n, spectral_radius=0.6, input_scale=0.8, leak_rate=0.4)
    seed = 7
    rng = np.random.default_rng(5)
    inputs = rng.uniform(-1.0, 1.0, size=80)

    esn = EchoStateNetwork(ESNConfig(seed=seed, **hp))
    ref = esn.drive(inputs)

    kernel = no_memory_weights()
    config = DistributedMemoryESNConfig(seed=seed, **hp)
    out = run_sequential_distributed_memory_esn(kernel, config, inputs)

    assert np.max(np.abs(ref - out)) < 1e-10


# --- 2. batched matches non-batched reference for K0/AB/kernel --------------------------------


@pytest.mark.parametrize("name,kernel_hp", [
    ("ESN66_K0", {}),
    ("ESN66_AB", {"tau": 5, "delayed_mass": 0.3}),
    ("ESN66_kernel", {"K": 15, "r": 0.9, "past_mass": 0.3}),
])
def test_batched_matches_sequential_reference(name, kernel_hp):
    n = 30
    esn_hp = dict(n_reservoir=n, spectral_radius=0.7, input_scale=1.0, leak_rate=0.3)
    seed = 9
    rng = np.random.default_rng(21)
    b, t = 6, 120
    inputs = rng.uniform(-1.0, 1.0, size=(b, t))

    kernel = kernel_for(name, kernel_hp)
    config = DistributedMemoryESNConfig(seed=seed, **esn_hp)
    batched = run_batched_distributed_memory_esn(kernel, config, inputs)

    ref = np.empty_like(batched)
    for i in range(b):
        ref[i] = run_sequential_distributed_memory_esn(kernel, config, inputs[i])

    assert np.max(np.abs(batched - ref)) < 1e-10


# --- 3. kernel weights are non-negative and sum to 1 -------------------------------------------


@pytest.mark.parametrize("name,kernel_hp", [
    ("ESN66_K0", {}),
    ("ESN66_AB", {"tau": 5, "delayed_mass": 0.3}),
    ("ESN66_kernel", {"K": 15, "r": 0.9, "past_mass": 0.3}),
])
def test_kernel_weights_nonnegative_and_sum_to_one(name, kernel_hp):
    kernel = kernel_for(name, kernel_hp)
    assert kernel.present >= 0.0
    assert np.all(kernel.delayed >= 0.0)
    assert abs(kernel.present + float(np.sum(kernel.delayed)) - 1.0) < 1e-12


# --- 4. causality: no future state is used ------------------------------------------------------


def test_no_future_state_is_used():
    """Perturbing the input strictly after time t must not change features at or before t."""

    n = 15
    esn_hp = dict(n_reservoir=n, spectral_radius=0.6, input_scale=1.0, leak_rate=0.5)
    seed = 4
    kernel = kernel_for("ESN66_kernel", {"K": 15, "r": 0.9, "past_mass": 0.3})
    config = DistributedMemoryESNConfig(seed=seed, **esn_hp)

    rng = np.random.default_rng(2)
    t = 60
    base = rng.uniform(-1.0, 1.0, size=(1, t))
    perturbed = base.copy()
    cut = 30
    perturbed[0, cut:] += 5.0  # large perturbation strictly after `cut`

    out_base = run_batched_distributed_memory_esn(kernel, config, base)
    out_pert = run_batched_distributed_memory_esn(kernel, config, perturbed)

    assert np.array_equal(out_base[0, :cut], out_pert[0, :cut])


# --- 5. buffers are independent between segments ------------------------------------------------


def test_buffers_independent_across_segments():
    """Perturbing one segment's early history must not change another segment's features."""

    n = 12
    esn_hp = dict(n_reservoir=n, spectral_radius=0.5, input_scale=1.0, leak_rate=0.5)
    seed = 6
    kernel = kernel_for("ESN66_kernel", {"K": 15, "r": 0.9, "past_mass": 0.3})
    config = DistributedMemoryESNConfig(seed=seed, **esn_hp)

    rng = np.random.default_rng(1)
    t = 50
    two_segments = rng.uniform(-1.0, 1.0, size=(2, t))
    perturbed = two_segments.copy()
    perturbed[0, :10] += 3.0  # perturb only segment 0's early history

    out_base = run_batched_distributed_memory_esn(kernel, config, two_segments)
    out_pert = run_batched_distributed_memory_esn(kernel, config, perturbed)

    assert np.array_equal(out_base[1], out_pert[1]), "segment 1 features changed when only segment 0 was perturbed"
    assert not np.array_equal(out_base[0], out_pert[0]), "segment 0 features should have changed"

    # Also check against fully-independent single-segment runs (no shared buffer at all).
    # Batch-size-dependent BLAS matmul kernel selection can introduce tiny
    # (<1e-9) floating-point associativity differences here, same caveat as
    # documented for the QRC batched reservoir (docs/eeg_protocol.md); the
    # within-batch comparisons above use exact equality since batch size is
    # unchanged there.
    single0 = run_batched_distributed_memory_esn(kernel, config, two_segments[0:1])
    single1 = run_batched_distributed_memory_esn(kernel, config, two_segments[1:2])
    assert np.allclose(out_base[0], single0[0], atol=1e-9)
    assert np.allclose(out_base[1], single1[0], atol=1e-9)


# --- 6. shared base draw across arms; final matrices shared only in fixed_core ------------------


def test_base_draw_shared_across_arms_and_modes():
    seed = 13
    w_raw_a, w_in_raw_a = base_reservoir_draw(66, seed)
    w_raw_b, w_in_raw_b = base_reservoir_draw(66, seed)
    assert np.array_equal(w_raw_a, w_raw_b)
    assert np.array_equal(w_in_raw_a, w_in_raw_b)


def test_final_weights_identical_across_arms_in_fixed_core():
    seed = 13
    sr, in_scale = 0.5, 1.0
    w_res1, w_in1 = build_esn_reservoir_weights(66, sr, in_scale, seed)
    w_res2, w_in2 = build_esn_reservoir_weights(66, sr, in_scale, seed)
    assert np.array_equal(w_res1, w_res2)
    assert np.array_equal(w_in1, w_in2)


def test_final_weights_differ_across_rescales_in_retuned_core():
    seed = 13
    w_res_a, _ = build_esn_reservoir_weights(66, 0.5, 1.0, seed)
    w_res_b, _ = build_esn_reservoir_weights(66, 0.9, 1.0, seed)
    assert not np.array_equal(w_res_a, w_res_b)


# --- 7. different seeds produce different reservoirs ---------------------------------------------


def test_different_seeds_produce_different_reservoirs():
    w_res_a, w_in_a = build_esn_reservoir_weights(66, 0.5, 1.0, seed=1)
    w_res_b, w_in_b = build_esn_reservoir_weights(66, 0.5, 1.0, seed=2)
    assert not np.array_equal(w_res_a, w_res_b)
    assert not np.array_equal(w_in_a, w_in_b)


# --- 8. features have dimension 66 ----------------------------------------------------------------


def test_features_have_dimension_66():
    kernel = kernel_for("ESN66_kernel", {"K": 15, "r": 0.9, "past_mass": 0.3})
    config = DistributedMemoryESNConfig(n_reservoir=66, spectral_radius=0.5, input_scale=1.0, leak_rate=0.7, seed=1)
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-1.0, 1.0, size=(2, 40))
    out = run_batched_distributed_memory_esn(kernel, config, inputs)
    assert out.shape == (2, 40, 66)


# --- 9. no NaNs or infinities ------------------------------------------------------------------


@pytest.mark.parametrize("name,kernel_hp", [
    ("ESN66_K0", {}),
    ("ESN66_AB", {"tau": 5, "delayed_mass": 0.3}),
    ("ESN66_kernel", {"K": 15, "r": 0.9, "past_mass": 0.3}),
])
def test_no_nan_or_inf(name, kernel_hp):
    esn_hp = dict(n_reservoir=66, spectral_radius=0.9, input_scale=1.0, leak_rate=0.7)
    features = construction_features(name, kernel_hp, esn_hp, seed=1, segments=np.random.default_rng(0).uniform(-2, 2, size=(3, 300)))
    assert np.all(np.isfinite(features))


# --- 10. train / val / test never overlap (frozen splits) --------------------------------------


def test_frozen_splits_disjoint():
    ecfg = eeg_cfg()
    for set_name in ecfg["data"]["sets"]:
        split = json.loads((ROOT / f"data/eeg/splits/{set_name}_split.json").read_text())
        train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)


# --- 11. HP search never reads the test partition -----------------------------------------------


@pytest.mark.skipif(
    not (ROOT / "scripts/run_esn_distributed_memory_hp_search.py").exists(),
    reason="hp search script not written yet",
)
def test_hp_search_script_never_touches_test_partition():
    script = (ROOT / "scripts/run_esn_distributed_memory_hp_search.py").read_text()
    assert "test_ids" not in script
    assert '["test"]' not in script


# --- 12. every expected holdout cell present exactly once ----------------------------------------


@pytest.mark.skipif(not FOLLOWUP_RAW.exists(), reason="followup holdout not generated yet")
def test_holdout_grid_complete_and_unique():
    import pandas as pd

    df = pd.read_csv(FOLLOWUP_RAW)
    ecfg = eeg_cfg()
    fcfg = cfg()
    sets = ecfg["data"]["sets"]
    horizons = ecfg["readout"]["horizons"]
    seeds = ecfg["channel"]["confirmatory_seeds"]
    constructions = list(fcfg["constructions"].keys())
    modes = ["fixed_core", "retuned_core"]

    key_cols = ["construction", "analysis_mode", "set", "horizon", "seed", "segment_id"]
    assert not df.duplicated(subset=key_cols).any()

    for construction in constructions:
        for mode in modes:
            for set_name in sets:
                split = json.loads((ROOT / f"data/eeg/splits/{set_name}_split.json").read_text())
                test_ids = set(split["test"])
                for horizon in horizons:
                    for seed in seeds:
                        got = set(df[
                            (df.construction == construction)
                            & (df.analysis_mode == mode)
                            & (df.set == set_name)
                            & (df.horizon == horizon)
                            & (df.seed == seed)
                        ]["segment_id"])
                        assert got == test_ids, f"{construction}/{mode}/{set_name}/h={horizon}/seed={seed} incomplete"
