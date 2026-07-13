"""Guardrail against the known 'noaux' failure mode: constructions that
should differ (different KernelWeights / mixing history) must actually
produce numerically distinct feature trajectories on identical input, or the
whole cross-construction comparison is vacuous.
"""

from __future__ import annotations

import numpy as np

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

N_QUBITS = 4


def test_constructions_differ_on_real_shaped_input():
    channel = build_input_channel(n_qubits=N_QUBITS, seed=20260712)
    init = pure_zero_state(2**N_QUBITS)
    rng = np.random.default_rng(11)
    t = np.arange(500)
    u = np.sin(0.05 * t) + 0.3 * np.sin(0.13 * t) + 0.1 * rng.standard_normal(500)

    constructions = {
        "AB-noaux": DiscreteDelayReservoir(init, tau=5, delayed_mass=0.4, channel=channel),
        "single-kernel": SingleExponentialStateKernelReservoir(init, K=10, r=0.85, past_mass=0.4, channel=channel),
        "dual-kernel": DualExponentialStateKernelReservoir(
            init, K=10, r_fast=0.5, r_slow=0.9, fast_mass=0.2, slow_mass=0.2, channel=channel
        ),
        "triangular": TriangularStateKernelReservoir(init, K=10, past_mass=0.4, channel=channel),
        "uniform": UniformStateKernelReservoir(init, K=10, past_mass=0.4, channel=channel),
    }

    trajs = {name: quantum_features(res.drive(u), N_QUBITS) for name, res in constructions.items()}
    names = list(trajs)
    n_pairs = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = float(np.mean(np.abs(trajs[names[i]] - trajs[names[j]])))
            assert diff > 1e-6, (
                f"{names[i]} vs {names[j]} trajectories are near-identical "
                f"(diff={diff}); comparison would be vacuous"
            )
            n_pairs += 1
    assert n_pairs == len(names) * (len(names) - 1) // 2
