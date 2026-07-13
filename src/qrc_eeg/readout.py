# Vendored from QRC-Glicose (github.com/Sfgiovanni/QRC-Glicose), MIT License, same author.
# Adapted here for the QRC-EEG study; see docs/eeg_protocol.md for provenance.
"""Linear ridge readout utilities."""

from __future__ import annotations

import numpy as np


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), np.asarray(x)])


def fit_readout(features: np.ndarray, target: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    """Fit a linear ridge readout with an unregularized intercept."""

    x = add_bias(features)
    y = np.asarray(target)
    if y.ndim == 1:
        y = y[:, None]
    reg = np.eye(x.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + float(alpha) * reg, x.T @ y)


def predict_readout(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pred = add_bias(features) @ weights
    return pred[:, 0] if pred.shape[1] == 1 else pred
