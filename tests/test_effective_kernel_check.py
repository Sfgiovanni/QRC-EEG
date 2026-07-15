"""Reproducibility and fail-high tests for corrected effective-kernel Gate 1."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from qrc_eeg.state_kernels import single_exponential_weights

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "effective_kernel_check", ROOT / "scripts/run_effective_kernel_check.py"
)
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def frozen() -> dict:
    return json.loads((ROOT / "config/effective_kernel_gate1_frozen.json").read_text())


def test_hp_are_loaded_from_committed_blob_without_hardcode() -> None:
    configuration, provenance = GATE.load_official_configuration()
    assert provenance["construction"] == "single_kernel"
    assert provenance["hp"] == configuration["hp"]
    source = (ROOT / "scripts/run_effective_kernel_check.py").read_text()
    assert "git_output(\"show\"" in source
    assert '"r": 0.7' not in source
    assert '"K": 15' not in source


def test_configuration_divergence_aborts() -> None:
    configuration = frozen()
    bad = {"single_kernel": {"hp": {**configuration["hp"], "r": 0.123}}}
    with pytest.raises(GATE.GateInvalid, match="committed HP changed") as error:
        GATE.validate_configuration(bad, configuration)
    assert error.value.verdict == "INVALID_CONFIG"


def test_geometric_weights_formula_sum_and_teff() -> None:
    hp = frozen()["hp"]
    kernel = single_exponential_weights(**hp)
    lags = np.arange(1, hp["K"] + 1)
    direct = hp["r"] ** lags
    expected = hp["past_mass"] * direct / direct.sum()
    assert np.allclose(kernel.delayed, expected, rtol=0, atol=1e-15)
    weights = np.r_[kernel.present, kernel.delayed]
    assert np.isclose(weights.sum(), 1.0, atol=1e-14)
    conditional = float(np.dot(lags, kernel.delayed) / kernel.past_mass)
    full = float(np.dot(lags, kernel.delayed))
    assert np.isclose(full, kernel.past_mass * conditional, atol=1e-14)
    metadata = json.loads((ROOT / "results/eeg/theory_vs_sim_metadata.json").read_text())
    assert np.isclose(conditional, metadata["kernel"]["conditional_delayed_T_eff"], atol=1e-13)
    assert np.isclose(full, metadata["kernel"]["full_mean_lag_including_w0"], atol=1e-13)


def test_finite_Wk_closed_form_matches_direct_fir() -> None:
    hp = frozen()["hp"]
    kernel = single_exponential_weights(**hp)
    z = 1.17 + 0.23j
    direct = kernel.present + sum(
        kernel.delayed[lag - 1] * z ** (-lag) for lag in range(1, hp["K"] + 1)
    )
    r, k, mass = hp["r"], hp["K"], hp["past_mass"]
    normalization = r * (1 - r**k) / (1 - r)
    delayed_closed = (r / z) * (1 - (r / z) ** k) / (1 - r / z)
    closed = (1 - mass) + mass * delayed_closed / normalization
    assert np.isclose(direct, closed, rtol=1e-14, atol=1e-14)


def test_actual_transfer_is_not_symbolically_separable() -> None:
    z, a, w = sp.symbols("z a w", nonzero=True)
    actual = 1 / (z - a * w)
    separable = w / (z - a)
    assert sp.simplify(actual - separable) != 0


def test_response_shapes_finiteness_and_stored_metrics() -> None:
    arrays = np.load(ROOT / "results/eeg/theory_vs_sim_responses.npz")
    for key in (
        "measured_impulse", "tangent_impulse", "separable_impulse",
        "measured_step", "tangent_step", "separable_step", "k0_impulse",
    ):
        assert arrays[key].shape == (256, 66)
        assert np.isfinite(arrays[key]).all()
    assert np.isfinite(arrays["kernel_weights"]).all()
    assert np.isfinite(arrays["companion_eigenvalues"]).all()
    table = pd.read_csv(ROOT / "results/eeg/theory_vs_sim_check.csv")
    for theory, prefix in (("tangent_recurrence", "tangent"), ("separable_W_times_R", "separable")):
        metrics = GATE.theory_metrics(
            arrays[f"{prefix}_impulse"], arrays[f"{prefix}_step"],
            arrays["measured_impulse"], arrays["measured_step"],
        )
        for metric, value in metrics.items():
            stored = table[(table.theory == theory) & (table.metric == metric)].value.iloc[0]
            assert np.isclose(stored, value, rtol=1e-12, atol=1e-14)


def verdict_rows(tangent: list[bool], separable: list[bool]) -> list[dict]:
    rows = []
    for theory, flags in (("tangent_recurrence", tangent), ("separable_W_times_R", separable)):
        for metric, passed in zip(GATE.TOLERANCES, flags, strict=True):
            rows.append({"theory": theory, "metric": metric, "pass": passed})
    return rows


def test_automatic_verdict_classification() -> None:
    assert GATE.classify_verdict(verdict_rows([True] * 4, [True] * 4)) == "PASS"
    assert GATE.classify_verdict(verdict_rows([True] * 4, [False] * 4)) == "FAIL_SEPARABLE_FACTORIZATION"
    assert GATE.classify_verdict(verdict_rows([False, True, True, True], [True] * 4)) == "FAIL_LINEARIZATION"
    assert GATE.classify_verdict([]) == "INVALID_PROVENANCE"
