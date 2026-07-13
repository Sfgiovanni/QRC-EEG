# Vendored from QRC-Glicose (github.com/Sfgiovanni/QRC-Glicose), MIT License, same author.
# Adapted here for the QRC-EEG study; see docs/eeg_protocol.md for provenance.
"""Echo-state and trace-distance diagnostics."""

from __future__ import annotations

import numpy as np


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Return one-half trace norm of two density matrices."""

    diff = 0.5 * (rho_a - rho_b + (rho_a - rho_b).conj().T)
    eig = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eig)))


def contraction_rate(distances: np.ndarray) -> float:
    """Estimate an exponential contraction rate from positive distances."""

    d = np.asarray(distances, dtype=float)
    idx = np.where(d > 1e-12)[0]
    if len(idx) < 2:
        return float("nan")
    x = idx.astype(float)
    y = np.log(d[idx])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)
