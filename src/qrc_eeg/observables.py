# Vendored from QRC-Glicose (github.com/Sfgiovanni/QRC-Glicose), MIT License, same author.
# Adapted here for the QRC-EEG study; see docs/eeg_protocol.md for provenance.
"""Pauli-observable helpers for small reservoirs."""

from __future__ import annotations

import itertools

import numpy as np

PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def local_pauli_observables(n_qubits: int) -> tuple[list[str], np.ndarray]:
    """Return all non-identity Pauli strings up to weight two."""

    labels = []
    mats = []
    for ops in itertools.product("IXYZ", repeat=n_qubits):
        weight = sum(op != "I" for op in ops)
        if weight == 0 or weight > 2:
            continue
        mat = PAULI[ops[0]]
        for op in ops[1:]:
            mat = np.kron(mat, PAULI[op])
        labels.append("".join(ops))
        mats.append(mat)
    return labels, np.asarray(mats)
