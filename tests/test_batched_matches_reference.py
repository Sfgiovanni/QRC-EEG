"""The batched evolution path must reproduce the per-segment reference exactly."""

from __future__ import annotations

import numpy as np

from qrc_eeg import (
    SingleExponentialStateKernelReservoir,
    build_input_channel,
    pure_zero_state,
    quantum_features,
)
from qrc_eeg.batched import run_batched_reservoir

N_QUBITS = 4


def test_batched_matches_sequential_reference():
    channel = build_input_channel(n_qubits=N_QUBITS, seed=20260712)
    init = pure_zero_state(2**N_QUBITS)
    kernel = single_exponential_kernel = SingleExponentialStateKernelReservoir(
        init, K=10, r=0.85, past_mass=0.4, channel=channel
    ).kernel

    rng = np.random.default_rng(42)
    b, t = 5, 200
    inputs = rng.uniform(-1.0, 1.0, size=(b, t))

    ref_features = np.empty((b, t, 66), dtype=np.float64)
    for i in range(b):
        res = SingleExponentialStateKernelReservoir(init, K=10, r=0.85, past_mass=0.4, channel=channel)
        rhos = res.drive(inputs[i])
        ref_features[i] = quantum_features(rhos, N_QUBITS)

    result = run_batched_reservoir(kernel, channel, init, inputs, check_every=10)

    max_err = np.max(np.abs(result.features - ref_features))
    assert max_err < 1e-8, f"batched vs sequential mismatch: {max_err}"
    assert result.trace_error_max < 1e-8
    assert result.hermiticity_error_max < 1e-8
    assert result.min_eigenvalue > -1e-9
