"""Causal classical forecasting controls for the EEG horizon gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import mae, rmse
from .readout import add_bias, predict_readout
from .tasks import nrmse, r2_score


@dataclass(frozen=True)
class SelectedRidge:
    alpha: float
    weights: np.ndarray
    validation_nrmse: float


def lag_features(segments: np.ndarray, p: int) -> np.ndarray:
    """Current input plus ``p-1`` causal lags; no future sample is used."""

    if p < 1:
        raise ValueError("p must be positive")
    x = np.asarray(segments, dtype=np.float64)
    out = np.zeros((x.shape[0], x.shape[1], p), dtype=np.float64)
    out[:, :, 0] = x
    for lag in range(1, p):
        out[:, lag:, lag] = x[:, :-lag]
    return out


def diagonal_nvar2(features: np.ndarray) -> np.ndarray:
    """Degree-2 NVAR: linear lag coordinates and their quadratic powers."""

    x = np.asarray(features, dtype=np.float64)
    return np.concatenate([x, x * x], axis=-1)


def tapped_delay_features(segments: np.ndarray, present: float, delayed: np.ndarray) -> np.ndarray:
    """Classical tapped input history with exactly the quantum kernel weights."""

    lags = lag_features(segments, len(delayed) + 1)
    weights = np.concatenate([[present], np.asarray(delayed, dtype=np.float64)])
    return lags * weights[None, None, :]


def _rows(features: np.ndarray, segments: np.ndarray, horizon: int, washout: int) -> tuple[np.ndarray, np.ndarray]:
    end = segments.shape[1] - horizon
    return (
        features[:, washout:end, :].reshape(-1, features.shape[-1]),
        segments[:, washout + horizon :,].reshape(-1),
    )


def _ridge_from_sufficient_stats(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xb = add_bias(x)
    gram = xb.T @ xb
    rhs = xb.T @ y[:, None]
    reg = np.eye(gram.shape[0])
    reg[0, 0] = 0.0
    return np.linalg.solve(gram + float(alpha) * reg, rhs)


def select_ridge_blocked(
    train_features: np.ndarray,
    train_segments: np.ndarray,
    validation_features: np.ndarray,
    validation_segments: np.ndarray,
    horizon: int,
    washout: int,
    alpha_grid: list[float],
) -> SelectedRidge:
    """Select alpha on complete validation segments and refit on train+validation."""

    x_train, y_train = _rows(train_features, train_segments, horizon, washout)
    x_val, y_val = _rows(validation_features, validation_segments, horizon, washout)
    xb = add_bias(x_train)
    gram = xb.T @ xb
    rhs = xb.T @ y_train[:, None]
    reg = np.eye(gram.shape[0])
    reg[0, 0] = 0.0
    best_alpha, best_score = float(alpha_grid[0]), float("inf")
    for alpha in alpha_grid:
        weights = np.linalg.solve(gram + float(alpha) * reg, rhs)
        score = nrmse(y_val, predict_readout(x_val, weights))
        if score < best_score:
            best_alpha, best_score = float(alpha), float(score)
    x_final = np.vstack([x_train, x_val])
    y_final = np.concatenate([y_train, y_val])
    weights = _ridge_from_sufficient_stats(x_final, y_final, best_alpha)
    return SelectedRidge(best_alpha, weights, best_score)


def evaluate_feature_model(
    features: np.ndarray,
    segments: np.ndarray,
    horizon: int,
    washout: int,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-segment held-out metrics for a fitted feature model."""

    count = len(segments)
    out = {name: np.full(count, np.nan) for name in ("nrmse", "rmse", "r2", "mae")}
    end = segments.shape[1] - horizon
    for i in range(count):
        y = segments[i, washout + horizon :]
        pred = predict_readout(features[i, washout:end, :], weights)
        out["nrmse"][i] = nrmse(y, pred)
        out["rmse"][i] = rmse(y, pred)
        out["r2"][i] = r2_score(y, pred)
        out["mae"][i] = mae(y, pred)
    return out


def evaluate_persistence(segments: np.ndarray, horizon: int, washout: int) -> dict[str, np.ndarray]:
    """Per-segment persistence metrics, yhat(t+h)=u(t)."""

    count = len(segments)
    out = {name: np.full(count, np.nan) for name in ("nrmse", "rmse", "r2", "mae")}
    end = segments.shape[1] - horizon
    for i in range(count):
        y = segments[i, washout + horizon :]
        pred = segments[i, washout:end]
        out["nrmse"][i] = nrmse(y, pred)
        out["rmse"][i] = rmse(y, pred)
        out["r2"][i] = r2_score(y, pred)
        out["mae"][i] = mae(y, pred)
    return out
