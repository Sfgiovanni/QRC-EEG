"""Feature extraction from reservoir trajectories.

Quantum arms: local Pauli-observable expectation values (weight <= 2), reused
from the vendored `observables.py`. Held identical across all quantum arms so
only the upstream `KernelWeights` history-mixing mechanism differs.
"""

from __future__ import annotations

import numpy as np

from .observables import local_pauli_observables


def quantum_features(rhos: list[np.ndarray], n_qubits: int) -> np.ndarray:
    """Return real Pauli-expectation feature vectors, one row per time step."""

    _, mats = local_pauli_observables(n_qubits)
    out = np.empty((len(rhos), len(mats)), dtype=np.float64)
    for t, rho in enumerate(rhos):
        out[t] = np.real(np.einsum("ij,kji->k", rho, mats))
    return out
