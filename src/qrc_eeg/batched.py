"""Batched (vectorized) reservoir evolution.

`qrc_eeg.models.StateMemoryReservoir` (vendored) drives one segment at a
time with a Python loop; a per-step cost of ~0.28 ms means the full protocol
(hundreds of segments x seeds x HP grid) would take on the order of 10 hours.
Neither source repository contains a batched/vectorized evolution path, so
this module was built new: it evolves many segments (and/or seeds) of the
SAME construction and SAME frozen channel simultaneously by stacking density
matrices into a leading batch axis and using numpy's native broadcasting over
that axis (matmul, einsum, and eigvalsh all batch for free). This changes
nothing about the model -- it is numerically equivalent, checked in
tests/test_batched_matches_reference.py -- only the execution strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels import InputEncodingChannel
from .observables import local_pauli_observables
from .state_kernels import KernelWeights


def _normalize_batch(rho: np.ndarray) -> np.ndarray:
    """Hermitize and trace-normalize a batch of density matrices, shape (B, d, d)."""

    out = 0.5 * (rho + np.conj(np.swapaxes(rho, -1, -2)))
    tr = np.einsum("bii->b", out).real
    return out / tr[:, None, None]


def _batched_partial_trace_qubit0(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    b = rho.shape[0]
    dim_rest = 2 ** (n_qubits - 1)
    rho5 = rho.reshape(b, 2, dim_rest, 2, dim_rest)
    return np.einsum("nkakc->nac", rho5)


def _batched_input_qubit_state(u: np.ndarray) -> np.ndarray:
    x = np.clip(u, 0.0, 1.0)
    psi = np.stack([np.sqrt(1.0 - x), np.sqrt(x)], axis=-1).astype(np.complex128)
    return np.einsum("bi,bj->bij", psi, np.conj(psi))


def batched_channel_step(channel: InputEncodingChannel, u: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Apply the frozen input-encoding channel to a batch of mixed states."""

    x = 1.0 / (1.0 + np.exp(-u)) if channel.squash else u
    rest = _batched_partial_trace_qubit0(rho, channel.n_qubits)
    injected_state = _batched_input_qubit_state(x)
    b = rho.shape[0]
    dim_rest = 2 ** (channel.n_qubits - 1)
    injected = np.einsum("bij,bac->biajc", injected_state, rest).reshape(b, 2 * dim_rest, 2 * dim_rest)
    return channel.unitary @ injected @ channel.unitary.conj().T


@dataclass
class BatchedFeatureResult:
    features: np.ndarray  # shape (B, T, F)
    trace_error_max: float
    hermiticity_error_max: float
    min_eigenvalue: float


def run_batched_reservoir(
    kernel: KernelWeights,
    channel: InputEncodingChannel,
    initial_state: np.ndarray,
    inputs: np.ndarray,
    check_every: int = 50,
) -> BatchedFeatureResult:
    """Evolve a batch of independent trajectories under one fixed construction.

    Parameters
    ----------
    inputs: array shape (B, T) -- B independent driving sequences (segments
        and/or seeds), T time steps each.
    check_every: run the (comparatively expensive) positivity/hermiticity
        diagnostic every `check_every` steps rather than every step; trace
        and hermiticity are still enforced by construction every step via
        `_normalize_batch`.
    """

    b, t = inputs.shape
    dim = initial_state.shape[0]
    n_qubits = channel.n_qubits
    labels, mats = local_pauli_observables(n_qubits)
    n_feat = len(labels)

    rho = np.tile(initial_state[None, :, :], (b, 1, 1)).astype(np.complex128)
    buffer = np.tile(rho[None, :, :, :], (kernel.K + 1, 1, 1, 1))

    features = np.empty((b, t, n_feat), dtype=np.float64)
    trace_err_max = 0.0
    herm_err_max = 0.0
    min_eig = 1.0

    for step in range(t):
        mix = kernel.present * rho
        for i, w in enumerate(kernel.delayed, start=1):
            if w:
                mix = mix + w * buffer[-1 - i]
        mix = _normalize_batch(mix)
        rho = _normalize_batch(batched_channel_step(channel, inputs[:, step], mix))
        buffer = np.concatenate([buffer[1:], rho[None]], axis=0)

        features[:, step, :] = np.real(np.einsum("bij,kji->bk", rho, mats))

        if step % check_every == 0:
            trace_err_max = max(trace_err_max, float(np.max(np.abs(np.einsum("bii->b", rho) - 1.0))))
            herm_err = np.max(np.abs(rho - np.conj(np.swapaxes(rho, -1, -2))))
            herm_err_max = max(herm_err_max, float(herm_err))
            eigs = np.linalg.eigvalsh(rho)
            min_eig = min(min_eig, float(eigs.min()))

    return BatchedFeatureResult(
        features=features,
        trace_error_max=trace_err_max,
        hermiticity_error_max=herm_err_max,
        min_eigenvalue=min_eig,
    )
