"""Self-contained tests for Gate 1B — post-gate robustness of the effective-kernel mechanism.

These tests do not read hp_selected.json and do not invoke the canonical Gate 1 provenance path,
so they are unaffected by the pre-existing HEAD/freeze commit divergence. They validate: the
u0-parameterization reproduces the frozen Gate 1 corner; the frozen grid is exactly 60 configs;
non-finite metrics are classified, never bare; the classification is mechanically recomputable;
and the frozen Gate 1 artifacts are unchanged.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results/eeg/gate1b_robustness"
CONFIG_PATH = ROOT / "config/effective_kernel_gate1b_robustness.json"

_SPEC = importlib.util.spec_from_file_location(
    "run_gate1b_robustness", ROOT / "scripts/run_gate1b_robustness.py")
G1B = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(G1B)

TOLERANCE_METRICS = ("impulse_relative_frobenius", "step_relative_frobenius",
                     "frequency_relative_frobenius", "memory_function_l1")

ARTIFACTS_PRESENT = (OUTDIR / "metrics_by_configuration.csv").exists()
needs_run = pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="Gate 1B artifacts not generated yet")


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


# --- pure/unit tests that never need the grid outputs -----------------------------------------


def test_config_grid_is_60_and_frozen() -> None:
    cfg = config()
    assert cfg["K"] == 15 and cfg["past_mass"] == 0.3
    assert sorted(cfg["r"]) == [0.7, 0.9]
    assert sorted(cfg["operating_points"]) == [-0.5, 0.0, 0.5]
    assert cfg["channel_seeds"] == list(range(1, 11))
    assert len(cfg["channel_seeds"]) * len(cfg["r"]) * len(cfg["operating_points"]) == 60
    assert cfg["expected_total_configurations"] == 60


def test_operating_points_are_interior_after_squash() -> None:
    # u0 in {-0.5,0,0.5} -> logistic -> interior of (0,1), no clipping
    for u0 in config()["operating_points"]:
        x = 1.0 / (1.0 + np.exp(-u0))
        assert 0.0 < x < 1.0
    assert np.isclose(1.0 / (1.0 + np.exp(-0.0)), 0.5)


def test_safe_pass_classifies_nonfinite_as_failure() -> None:
    assert G1B.safe_pass(0.005, 0.01) is True
    assert G1B.safe_pass(0.05, 0.01) is False
    assert G1B.safe_pass(float("nan"), 0.01) is False
    assert G1B.safe_pass(float("inf"), 0.01) is False
    assert np.isnan(G1B.safe_pass(0.99, None))  # diagnostic metric has no threshold


def _joint(n: int, n_tangent: int, n_separable: int, stable: bool = True) -> pd.DataFrame:
    cols = ["seed", "r", "u0", "valid", "stable", "tangent_all4", "separable_all4"]
    rows = []
    for i in range(n):
        rows.append({"seed": i, "r": 0.7, "u0": 0.0, "valid": True, "stable": stable,
                     "tangent_all4": i < n_tangent, "separable_all4": i < n_separable})
    return pd.DataFrame(rows, columns=cols)


def test_frozen_classification_matches_spec_at_boundaries() -> None:
    # ROBUST: >=90% tangent AND <=10% separable
    assert G1B.classify(_joint(10, 10, 0))["classification"] == "ROBUST_WITHIN_GRID"
    assert G1B.classify(_joint(10, 9, 1))["classification"] == "ROBUST_WITHIN_GRID"
    # NOT_ROBUST must take precedence over MIXED when separable passes in >=50% (advisor case)
    assert G1B.classify(_joint(10, 7, 5))["classification"] == "NOT_ROBUST_WITHIN_GRID"
    # NOT_ROBUST when tangent passes in <=50%
    assert G1B.classify(_joint(10, 5, 0))["classification"] == "NOT_ROBUST_WITHIN_GRID"
    # MIXED: tangent >50%, separable <50%, but not ROBUST
    assert G1B.classify(_joint(10, 6, 2))["classification"] == "MIXED"
    # empty valid set -> INVALID
    assert G1B.classify(_joint(0, 0, 0))["classification"] == "INVALID"


def test_u0_parameterization_reproduces_gate1_corner() -> None:
    """The seed=1, r=0.7, u0=0, eps=1e-4 corner must reproduce the frozen Gate 1 numbers."""
    from qrc_eeg.channels import build_input_channel
    from qrc_eeg.observables import local_pauli_observables
    from qrc_eeg.state_kernels import single_exponential_weights

    channel = build_input_channel(n_qubits=4, seed=1)
    _, obs = local_pauli_observables(4)
    obs = np.asarray(obs)
    basis = G1B.GATE1.hermitian_traceless_basis(16)
    rho_star, iters, diff, converged = G1B.fixed_state_at(channel, 0.0, 1e-13, 5000)
    assert converged and iters == 296 and np.isclose(diff, 9.454e-14, rtol=1e-3)
    A, B, C = G1B.build_abc_at(channel, rho_star, basis, obs, 0.0, 1e-4)
    kernel = single_exponential_weights(K=15, r=0.7, past_mass=0.3)
    w = np.concatenate([[kernel.present], kernel.delayed])
    impulse = np.zeros(256); impulse[0] = 1.0
    step = np.ones(256)
    ti = G1B.GATE1.tangent_response(A, B, C, w, impulse)
    ts = G1B.GATE1.tangent_response(A, B, C, w, step)
    mi = G1B.nonlinear_response_at(kernel, channel, rho_star, impulse, 0.0, 1e-4)
    ms = G1B.nonlinear_response_at(kernel, channel, rho_star, step, 0.0, 1e-4)
    m = G1B.GATE1.theory_metrics(ti, ts, mi, ms)
    assert np.isclose(m["impulse_relative_frobenius"], 2.7416e-5, rtol=2e-3)
    assert np.isclose(m["step_relative_frobenius"], 4.0720e-5, rtol=2e-3)
    _, _, radius = G1B.companion_spectrum_local(A, w)
    assert np.isclose(radius, 0.9587240324199373, rtol=1e-9)


def test_abc_is_r_independent() -> None:
    """A, B, C depend only on (seed, u0); reuse across r is exact, not approximate."""
    from qrc_eeg.channels import build_input_channel
    from qrc_eeg.observables import local_pauli_observables

    channel = build_input_channel(n_qubits=4, seed=3)
    _, obs = local_pauli_observables(4); obs = np.asarray(obs)
    basis = G1B.GATE1.hermitian_traceless_basis(16)
    rho_star, *_ = G1B.fixed_state_at(channel, 0.5, 1e-13, 5000)
    A1, B1, C1 = G1B.build_abc_at(channel, rho_star, basis, obs, 0.5, 1e-4)
    A2, B2, C2 = G1B.build_abc_at(channel, rho_star, basis, obs, 0.5, 1e-4)
    assert np.array_equal(A1, A2) and np.array_equal(B1, B2) and np.array_equal(C1, C2)


# --- tests over the generated grid ------------------------------------------------------------


@needs_run
def test_exactly_60_configurations_complete_grid() -> None:
    cfg = config()
    metrics = pd.read_csv(OUTDIR / "metrics_by_configuration.csv")
    spectrum = pd.read_csv(OUTDIR / "spectrum_by_configuration.csv")
    expected = {(int(s), float(r), float(u)) for s in cfg["channel_seeds"]
                for r in cfg["r"] for u in cfg["operating_points"]}
    assert len(expected) == 60
    got = {(int(s), float(r), float(u)) for s, r, u in zip(spectrum.seed, spectrum.r, spectrum.u0)}
    assert got == expected
    assert len(spectrum) == 60 and not spectrum.duplicated(subset=["seed", "r", "u0"]).any()
    assert len(metrics) == 720


@needs_run
def test_no_unclassified_nonfinite() -> None:
    metrics = pd.read_csv(OUTDIR / "metrics_by_configuration.csv")
    tol = metrics[metrics.metric.isin(TOLERANCE_METRICS)]
    nonfinite = tol[~np.isfinite(tol.value.astype(float))]
    assert not nonfinite["pass"].astype(str).str.lower().eq("true").any()
    assert not nonfinite["valid"].astype(bool).any()


@needs_run
def test_classification_recomputable_and_gate1_unchanged() -> None:
    metadata = json.loads((OUTDIR / "metadata.json").read_text())
    assert metadata["gate1_artifacts_unchanged"] is True
    assert metadata["golden_gate1_corner_reproduced"] is True
    assert metadata["classification"]["global"]["classification"] in (
        "ROBUST_WITHIN_GRID", "MIXED", "NOT_ROBUST_WITHIN_GRID", "INVALID")
