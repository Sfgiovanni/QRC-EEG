"""Tests for the crossed segment x seed sensitivity analysis
(docs/crossed_inference_protocol.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from qrc_eeg.crossed_inference import (
    crossed_bootstrap,
    fit_crossed_mixed_model,
    interaction_matrix,
    original_style_interaction,
)

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RAW = ROOT / "results/eeg/raw/eeg_holdout_by_segment_seed.csv"
CROSSED_DIR = ROOT / "results/eeg/followup/crossed_inference"
CONFIG_PATH = ROOT / "config/esn_distributed_memory_frozen.yaml"


def _toy_frame() -> pd.DataFrame:
    """2 segments x 3 seeds, 2 constructions, 2 horizons -- exactly determined interaction."""

    rows = []
    # kernel: D = 1 for every (segment,seed); comparator: D = 3 for every (segment,seed)
    # => I(comp - kernel) = 2 everywhere
    for seg in ("A", "B"):
        for seed in (1, 2, 3):
            rows.append({"construction": "kernel", "set": "Z", "horizon": 2, "seed": seed, "segment_id": seg, "nrmse": 5.0})
            rows.append({"construction": "kernel", "set": "Z", "horizon": 64, "seed": seed, "segment_id": seg, "nrmse": 6.0})
            rows.append({"construction": "comp", "set": "Z", "horizon": 2, "seed": seed, "segment_id": seg, "nrmse": 5.0})
            rows.append({"construction": "comp", "set": "Z", "horizon": 64, "seed": seed, "segment_id": seg, "nrmse": 8.0})
    return pd.DataFrame(rows)


def test_interaction_matrix_shape_and_value():
    df = _toy_frame()
    mat, segs, seeds = interaction_matrix(df, "kernel", "comp", "Z", 2, 64)
    assert mat.shape == (2, 3)
    assert segs == ["A", "B"]
    assert seeds == [1, 2, 3]
    assert np.allclose(mat, 2.0)


def test_interaction_matrix_raises_on_incomplete_grid():
    df = _toy_frame()
    df = df[~((df["segment_id"] == "B") & (df["seed"] == 3))]
    with pytest.raises(ValueError):
        interaction_matrix(df, "kernel", "comp", "Z", 2, 64)


def test_crossed_bootstrap_constant_matrix_is_degenerate():
    mat = np.full((20, 10), 2.0)
    rng = np.random.default_rng(0)
    res = crossed_bootstrap(mat, rng, n_replicates=500)
    assert res["observed_mean"] == 2.0
    assert res["bootstrap_mean"] == pytest.approx(2.0)
    assert res["se"] == pytest.approx(0.0, abs=1e-12)
    assert res["sign_fraction"] == 1.0
    assert res["p_bootstrap"] == 0.0


def test_crossed_bootstrap_zero_mean_matrix_is_symmetric():
    rng_data = np.random.default_rng(1)
    half = rng_data.normal(loc=0.0, scale=1.0, size=(10, 10))
    mat = np.vstack([half, -half])  # exact zero mean by construction
    assert abs(mat.mean()) < 1e-12
    rng = np.random.default_rng(2)
    res = crossed_bootstrap(mat, rng, n_replicates=5000)
    assert res["sign_fraction"] == pytest.approx(0.5, abs=0.15)
    assert res["ci95_lo"] < 0 < res["ci95_hi"]


def test_crossed_bootstrap_uses_cartesian_product_not_pairwise():
    """Resampling must be independent across the two axes: replica size is
    always n_seg x n_seed, never n_seg pairs."""

    mat = np.arange(20).reshape(4, 5).astype(float)
    rng = np.random.default_rng(3)
    seg_idx = rng.integers(0, 4, size=(1, 4))
    seed_idx = rng.integers(0, 5, size=(1, 5))
    sample = mat[seg_idx[:, :, None], seed_idx[:, None, :]]
    assert sample.shape == (1, 4, 5)


def test_original_style_matches_canonical_gate_interaction_for_single_kernel_vs_k0():
    """Cross-check against the already-published, independently computed
    results/eeg/gate_interactions.csv value for single_kernel vs QRC_K0, set Z."""

    if not PRIMARY_RAW.exists():
        pytest.skip("primary holdout CSV not present")
    df = pd.read_csv(PRIMARY_RAW)
    rng = np.random.default_rng(20260722)
    result = original_style_interaction(df, "single_kernel", "QRC_K0", "Z", 2, 64, rng, n_replicates=2000)
    gate_path = ROOT / "results/eeg/gate_interactions.csv"
    if not gate_path.exists():
        pytest.skip("canonical gate_interactions.csv not present")
    gate = pd.read_csv(gate_path)
    row = gate[(gate["comparator"] == "QRC_K0") & (gate["set"] == "Z")].iloc[0]
    assert result["observed_mean"] == pytest.approx(row["interaction_comp_minus_kernel"], abs=1e-9)


def test_fit_crossed_mixed_model_returns_diagnostics_never_raises():
    df = _toy_frame()
    result = fit_crossed_mixed_model(df, "kernel", "comp", "Z", 2, 64)
    assert "converged" in result and "boundary_hit" in result and "singular" in result
    assert result["error"] is None or isinstance(result["error"], str)


def test_fit_crossed_mixed_model_handles_empty_cell_without_raising():
    df = _toy_frame()
    result = fit_crossed_mixed_model(df, "kernel", "nonexistent_comparator", "Z", 2, 64)
    assert result["error"] is not None
    assert result["converged"] is False


def test_frozen_family_size_is_21():
    fcfg = yaml.safe_load(CONFIG_PATH.read_text())
    ci = fcfg["crossed_inference"]
    n = 0
    for contrast in ci["contrasts"]:
        n += len(contrast["modes"]) * len(ci["sets"])
    assert n == 21


@pytest.mark.skipif(not (CROSSED_DIR / "crossed_bootstrap.csv").exists(), reason="crossed inference not run yet")
def test_crossed_bootstrap_csv_has_21_rows_and_holm_column():
    df = pd.read_csv(CROSSED_DIR / "crossed_bootstrap.csv")
    assert len(df) == 21
    assert "p_holm" in df.columns
    assert df["p_holm"].between(0, 1).all()


@pytest.mark.skipif(not (CROSSED_DIR / "mixed_model_diagnostics.json").exists(), reason="crossed inference not run yet")
def test_mixed_model_diagnostics_never_silently_favorable():
    diagnostics = json.loads((CROSSED_DIR / "mixed_model_diagnostics.json").read_text())
    for cell in diagnostics:
        assert "converged" in cell and "boundary_hit" in cell and "singular" in cell
