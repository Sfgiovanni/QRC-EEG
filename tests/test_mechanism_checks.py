"""Mechanism sanity checks (spec section 7), run on synthetic input.

These validate the reservoir+channel machinery itself, not any scientific
hypothesis about EEG -- they must pass before any pre-registered experiment
is run, and again before real EEG data is substituted in.
"""

from __future__ import annotations

import numpy as np
import pytest

from qrc_eeg import (
    DiscreteDelayReservoir,
    DualExponentialStateKernelReservoir,
    SingleExponentialStateKernelReservoir,
    TriangularStateKernelReservoir,
    UniformStateKernelReservoir,
    build_input_channel,
    pure_zero_state,
    quantum_features,
)
from qrc_eeg.memory_capacity import memory_target
from qrc_eeg.metrics import capacity_score
from qrc_eeg.readout import fit_readout, predict_readout

N_QUBITS = 4
CHANNEL_SEED = 20260712


def _synthetic_input(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=n)


def _channel():
    return build_input_channel(n_qubits=N_QUBITS, seed=CHANNEL_SEED)


def _all_constructions(channel):
    init = pure_zero_state(2**N_QUBITS)
    return {
        "AB-noaux": DiscreteDelayReservoir(init, tau=5, delayed_mass=0.4, channel=channel),
        "single-kernel": SingleExponentialStateKernelReservoir(init, K=10, r=0.85, past_mass=0.4, channel=channel),
        "dual-kernel": DualExponentialStateKernelReservoir(
            init, K=10, r_fast=0.5, r_slow=0.9, fast_mass=0.2, slow_mass=0.2, channel=channel
        ),
        "triangular": TriangularStateKernelReservoir(init, K=10, past_mass=0.4, channel=channel),
        "uniform": UniformStateKernelReservoir(init, K=10, past_mass=0.4, channel=channel),
    }


def test_trace_hermiticity_positivity_over_trajectory():
    channel = _channel()
    reservoirs = _all_constructions(channel)
    u = _synthetic_input(120, seed=1)
    for name, res in reservoirs.items():
        for rho in res.drive(u):
            diag_trace = abs(np.trace(rho) - 1.0)
            diag_herm = np.linalg.norm(rho - rho.conj().T)
            min_eig = float(np.linalg.eigvalsh(rho).min())
            assert diag_trace < 1e-8, f"{name}: trace error {diag_trace}"
            assert diag_herm < 1e-8, f"{name}: hermiticity error {diag_herm}"
            assert min_eig > -1e-10, f"{name}: negative eigenvalue {min_eig}"


def test_fading_memory_echo_state_property():
    """Two different initial states converge under the same driving input."""

    channel = _channel()
    u = _synthetic_input(300, seed=2)
    rho_zero = pure_zero_state(2**N_QUBITS)
    rho_other = np.eye(2**N_QUBITS, dtype=np.complex128) / (2**N_QUBITS)

    res_a = SingleExponentialStateKernelReservoir(rho_zero, K=10, r=0.85, past_mass=0.4, channel=channel)
    res_b = SingleExponentialStateKernelReservoir(rho_other, K=10, r=0.85, past_mass=0.4, channel=channel)
    traj_a = res_a.drive(u)
    traj_b = res_b.drive(u)

    early_diff = np.linalg.norm(traj_a[5] - traj_b[5])
    late_diff = np.linalg.norm(traj_a[-1] - traj_b[-1])
    assert late_diff < early_diff, "trajectories should converge (fading memory), not diverge"
    assert late_diff < 1e-3, f"residual divergence too large: {late_diff}"


def test_shuffled_target_leakage_r2_near_zero():
    channel = _channel()
    init = pure_zero_state(2**N_QUBITS)
    res = SingleExponentialStateKernelReservoir(init, K=10, r=0.85, past_mass=0.4, channel=channel)
    u = _synthetic_input(800, seed=3)
    rhos = res.drive(u)
    feats = quantum_features(rhos, N_QUBITS)

    washout = 50
    rng = np.random.default_rng(4)
    shuffled_target = rng.permutation(u)

    train_feats, train_y = feats[washout:-1], shuffled_target[washout + 1 :]
    weights = fit_readout(train_feats, train_y, alpha=1e-3)
    pred = predict_readout(train_feats, weights)
    r2 = capacity_score(train_y, pred)
    assert r2 < 0.05, f"shuffled-target leakage detected: R^2={r2}"


def test_nonzero_quadratic_capacity():
    channel = _channel()
    init = pure_zero_state(2**N_QUBITS)
    res = SingleExponentialStateKernelReservoir(init, K=10, r=0.85, past_mass=0.4, channel=channel)
    u = _synthetic_input(1500, seed=5)
    rhos = res.drive(u)
    feats = quantum_features(rhos, N_QUBITS)

    tau = 2
    target = memory_target(u, tau=tau, kind="quadratic")
    washout = 50
    valid = ~np.isnan(target)
    idx = np.where(valid)[0]
    idx = idx[idx >= washout]
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]

    weights = fit_readout(feats[train_idx], target[train_idx], alpha=1e-3)
    pred = predict_readout(feats[test_idx], weights)
    capacity = capacity_score(target[test_idx], pred)
    assert capacity > 0.05, f"quadratic capacity too low: {capacity}"


def test_complex64_matches_complex128():
    """Manual complex64 replica of the mix-then-channel update vs. the complex128 class."""

    channel = _channel()
    init = pure_zero_state(2**N_QUBITS)
    res = SingleExponentialStateKernelReservoir(init, K=6, r=0.85, past_mass=0.4, channel=channel)
    u = _synthetic_input(60, seed=6)
    traj128 = res.drive(u)

    unitary64 = channel.unitary.astype(np.complex64)
    kernel = res.kernel
    dim = 2**N_QUBITS
    rho64 = init.astype(np.complex64)
    buffer = [rho64.copy() for _ in range(kernel.K + 1)]

    def channel64(x: float, rho: np.ndarray) -> np.ndarray:
        from qrc_eeg.channels import _input_qubit_state, _partial_trace_qubit0, squash_to_unit_interval

        xs = squash_to_unit_interval(x)
        rest = _partial_trace_qubit0(rho.astype(np.complex128), N_QUBITS).astype(np.complex64)
        injected = np.kron(_input_qubit_state(xs).astype(np.complex64), rest)
        return unitary64 @ injected @ unitary64.conj().T

    def normalize64(rho: np.ndarray) -> np.ndarray:
        out = 0.5 * (rho + rho.conj().T)
        return (out / np.trace(out)).astype(np.complex64)

    traj64 = []
    for x in u:
        mix = kernel.present * rho64
        for i, w in enumerate(kernel.delayed, start=1):
            if w:
                mix = mix + np.complex64(w) * buffer[-1 - i]
        mix = normalize64(mix)
        rho64 = normalize64(channel64(float(x), mix))
        buffer.append(rho64.copy())
        buffer.pop(0)
        traj64.append(rho64.copy())

    final_err = np.max(np.abs(traj128[-1] - traj64[-1].astype(np.complex128)))
    assert final_err < 1e-4, f"complex64 vs complex128 mismatch: {final_err}"


def test_constructions_produce_distinct_trajectories():
    """Guardrail: distinct KernelWeights must yield distinct feature trajectories.

    Regression guard against the known failure mode where 'noaux' construction
    variants silently ran identical code and produced spuriously tied results.
    """

    channel = _channel()
    reservoirs = _all_constructions(channel)
    u = _synthetic_input(400, seed=7)

    feature_trajs = {}
    for name, res in reservoirs.items():
        rhos = res.drive(u)
        feature_trajs[name] = quantum_features(rhos, N_QUBITS)

    names = list(feature_trajs.keys())
    min_diff = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = float(np.mean(np.abs(feature_trajs[names[i]] - feature_trajs[names[j]])))
            if min_diff is None or diff < min_diff:
                min_diff = diff
            assert diff > 1e-6, f"{names[i]} and {names[j]} produce near-identical trajectories (diff={diff})"
    assert min_diff is not None and min_diff > 1e-6
