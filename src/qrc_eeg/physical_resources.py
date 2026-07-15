"""Auditable resource formulas for density-matrix state-history reservoirs."""

from __future__ import annotations

import numpy as np


def buffer_resource_counts(n_qubits: int, K: int, dtype=np.complex128) -> dict[str, int | str]:
    if n_qubits < 1 or K < 0:
        raise ValueError("n_qubits must be positive and K non-negative")
    d = 2**n_qubits
    states = K + 1
    array = np.empty((states, d, d), dtype=dtype)
    return {
        "n_qubits": n_qubits,
        "dimension": d,
        "K": K,
        "buffer_states": states,
        "independent_real_parameters": states * (d * d - 1),
        "conservative_real_scalars": states * d * d,
        "implementation_dtype": array.dtype.name,
        "dtype_itemsize_bytes": array.dtype.itemsize,
        "dense_buffer_bytes": array.nbytes,
    }


def operation_counts(n_qubits: int, K: int, observables: int, trajectory_length: int) -> dict[str, int | str]:
    counts = buffer_resource_counts(n_qubits, K)
    d, states = int(counts["dimension"]), int(counts["buffer_states"])
    mix_mult = states * d * d
    mix_add = K * d * d
    channel = 2 * d**3
    readout = observables * d * d
    per_step = mix_mult + mix_add + channel + readout
    return {
        "mix_complex_scalar_multiplies_per_step": mix_mult,
        "mix_complex_additions_per_step": mix_add,
        "dense_channel_complex_mac_upper_per_step": channel,
        "observable_trace_complex_mac_upper_per_step": readout,
        "classical_complex_ops_proxy_per_step": per_step,
        "classical_complex_ops_proxy_trajectory": trajectory_length * per_step,
        "mix_cost_expression": "O((K+1)d^2)",
        "dense_channel_cost_expression": "O(d^3)",
        "observable_cost_expression": "O(Md^2)",
    }


def conservative_measurement_counts(observables: int, shots: int, trajectory_length: int) -> dict[str, int]:
    if observables < 1 or shots < 1 or trajectory_length < 1:
        raise ValueError("measurement counts must be positive")
    return {
        "measurement_groups_conservative": observables,
        "shots_per_group": shots,
        "preparations_per_step": observables * shots,
        "preparations_per_trajectory": trajectory_length * observables * shots,
    }
