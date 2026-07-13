# Vendored from QRC-Glicose (github.com/Sfgiovanni/QRC-Glicose), MIT License, same author.
# Adapted here for the QRC-EEG study; see docs/eeg_protocol.md for provenance.
"""Canonical memory-capacity targets."""

from __future__ import annotations

import numpy as np


def memory_target(inputs: np.ndarray, tau: int, kind: str, tau2: int | None = None) -> np.ndarray:
    """Build linear, quadratic, parity, or cross-delay memory targets."""

    u = np.asarray(inputs, dtype=np.float64)
    y = np.full_like(u, np.nan)
    if kind == "linear":
        y[tau:] = u[:-tau]
    elif kind == "quadratic":
        y[tau:] = u[:-tau] ** 2
    elif kind == "parity":
        y[tau:] = np.sign(u[:-tau])
    elif kind == "cross":
        if tau2 is None:
            raise ValueError("tau2 is required for cross memory")
        tmax = max(tau, tau2)
        y[tmax:] = u[tmax - tau : len(u) - tau] * u[tmax - tau2 : len(u) - tau2]
    else:
        raise ValueError(f"unknown memory target kind: {kind}")
    return y
