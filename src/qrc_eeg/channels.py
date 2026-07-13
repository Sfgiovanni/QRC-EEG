"""Input-encoding quantum channel for the QRC-EEG study.

Neither `QRC-Glicose` nor `QRC-Kernel` ships an input-dependent channel usable
on real signals: the only channels in QRC-Glicose (`identity_channel`,
`depolarizing_channel`) ignore the scalar input entirely and exist solely to
exercise `StateMemoryReservoir` in smoke tests. This module builds the piece
that was missing: a standard Fujii-Nakajima-style QRC input channel.

Construction (fixed for every reservoir arm -- AB-noaux, single/dual
exponential, triangular, uniform kernels):

1. One qubit (qubit 0) is the *input qubit*. Its reduced state is discarded
   (partial trace) and replaced by an amplitude encoding of the scalar input
   ``u in [0, 1]``: ``|psi(u)> = sqrt(1-u)|0> + sqrt(u)|1>``.
2. The resulting product state is evolved by a fixed, seeded entangling
   unitary ``U = expm(-i H t)`` generated once from a random transverse-field
   Ising Hamiltonian ``H = sum_i h_i Z_i + sum_{i<j} J_ij X_i X_j``.

Steps (1)-(2) are held IDENTICAL in every arm of the comparison. Only the
`KernelWeights` history-mixing mechanism upstream of this channel differs
between constructions, so this channel cannot confound the memory-mechanism
contrast that is the object of the study (see docs/eeg_protocol.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import expm

DensityMatrix = np.ndarray

PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def _kron_all(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def fixed_entangling_unitary(n_qubits: int, seed: int, evolution_time: float = 1.0) -> np.ndarray:
    """Build a fixed random transverse-field Ising unitary ``expm(-i H t)``.

    Deterministic given (n_qubits, seed, evolution_time); frozen once chosen.
    """

    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    h = np.zeros((dim, dim), dtype=np.complex128)
    fields = rng.uniform(0.5, 1.5, size=n_qubits)
    for i in range(n_qubits):
        ops = [I2] * n_qubits
        ops[i] = PAULI_Z
        h += fields[i] * _kron_all(ops)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coupling = rng.uniform(0.5, 1.5)
            ops = [I2] * n_qubits
            ops[i] = PAULI_X
            ops[j] = PAULI_X
            h += coupling * _kron_all(ops)
    return expm(-1j * h * evolution_time)


def _partial_trace_qubit0(rho: DensityMatrix, n_qubits: int) -> DensityMatrix:
    """Trace out qubit 0, returning the reduced state of the remaining qubits."""

    dim_rest = 2 ** (n_qubits - 1)
    rho4 = rho.reshape(2, dim_rest, 2, dim_rest)
    return np.einsum("kakb->ab", rho4)


def _input_qubit_state(u: float) -> DensityMatrix:
    """Pure-state amplitude encoding ``sqrt(1-u)|0> + sqrt(u)|1>``, ``u in [0, 1]``."""

    u = float(np.clip(u, 0.0, 1.0))
    psi = np.array([np.sqrt(1.0 - u), np.sqrt(u)], dtype=np.complex128)
    return np.outer(psi, psi.conj())


def squash_to_unit_interval(z: np.ndarray | float) -> np.ndarray | float:
    """Logistic squashing of a (already z-scored) signal into ``(0, 1)``."""

    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


@dataclass(frozen=True)
class InputEncodingChannel:
    """Callable ``channel(u, rho) -> rho'`` matching qrc_eeg.models.Channel."""

    n_qubits: int
    unitary: np.ndarray
    squash: bool = True

    def __call__(self, u: float, rho: DensityMatrix) -> DensityMatrix:
        x = squash_to_unit_interval(u) if self.squash else u
        rest = _partial_trace_qubit0(rho, self.n_qubits)
        injected = np.kron(_input_qubit_state(x), rest)
        return self.unitary @ injected @ self.unitary.conj().T


def build_input_channel(n_qubits: int = 4, seed: int = 20260712, evolution_time: float = 1.0, squash: bool = True) -> InputEncodingChannel:
    """Construct the frozen input-encoding channel shared by every arm."""

    unitary = fixed_entangling_unitary(n_qubits, seed=seed, evolution_time=evolution_time)
    return InputEncodingChannel(n_qubits=n_qubits, unitary=unitary, squash=squash)
