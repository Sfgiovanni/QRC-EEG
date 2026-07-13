# Vendored from QRC-Glicose (github.com/Sfgiovanni/QRC-Glicose), MIT License, same author.
# Adapted here for the QRC-EEG study; see docs/eeg_protocol.md for provenance.
"""Forecast and memory metrics."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def skill_score_mse(model_mse: float, persistence_mse: float) -> float:
    return float(1.0 - model_mse / persistence_mse) if persistence_mse > 0.0 else float("nan")


def capacity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    vy = float(np.var(y))
    vp = float(np.var(p))
    if vy <= 1e-14 or vp <= 1e-14:
        return 0.0
    cov = float(np.mean((y - y.mean()) * (p - p.mean())))
    return float(max(0.0, min(1.0, cov * cov / (vy * vp))))
