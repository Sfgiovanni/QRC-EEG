"""EEG-specific tasks: forecasting targets, nonlinear-demand score, seizure task.

New module -- neither source repository targets EEG. Implements exactly the
formulas frozen in docs/eeg_preregistration.md, computed before any
reservoir construction is compared.
"""

from __future__ import annotations

import numpy as np

from .readout import fit_readout, predict_readout

FORECAST_HORIZONS = (1, 2, 4, 8)
DEMAND_EMBEDDING_ORDER = 10
DEMAND_ALPHA_GRID = np.logspace(-6, 2, 9)
DEMAND_N_FOLDS = 5
DEMAND_EPSILON = 1e-6


def zscore(signal: np.ndarray, mean: float | None = None, std: float | None = None) -> tuple[np.ndarray, float, float]:
    x = np.asarray(signal, dtype=np.float64)
    m = float(np.mean(x)) if mean is None else mean
    s = float(np.std(x)) if std is None else std
    s = s if s > 1e-12 else 1.0
    return (x - m) / s, m, s


def forecast_target(signal: np.ndarray, horizon: int) -> np.ndarray:
    """Causal h-step-ahead target: target[t] = signal[t + horizon], NaN-padded at the tail."""

    x = np.asarray(signal, dtype=np.float64)
    y = np.full_like(x, np.nan)
    if horizon < len(x):
        y[: len(x) - horizon] = x[horizon:]
    return y


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-14 else 0.0


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    denom = float(np.std(y))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return float(rmse / denom) if denom > 1e-14 else float("nan")


def _delay_embedding(signal: np.ndarray, order: int) -> np.ndarray:
    """Column i (0-indexed) holds signal[t - (i+1)]; NaN where unavailable."""

    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    emb = np.full((n, order), np.nan)
    for lag in range(1, order + 1):
        emb[lag:, lag - 1] = x[: n - lag]
    return emb


def _quadratic_expand(embedding: np.ndarray) -> np.ndarray:
    n, p = embedding.shape
    extra = [embedding[:, i] * embedding[:, j] for i in range(p) for j in range(i, p)]
    return np.column_stack([embedding] + extra)


def _select_alpha_and_fit(x_train, y_train, x_val, y_val, alpha_grid) -> np.ndarray:
    best_alpha, best_r2 = alpha_grid[0], -np.inf
    for alpha in alpha_grid:
        w = fit_readout(x_train, y_train, alpha=alpha)
        pred = predict_readout(x_val, w)
        r2 = r2_score(y_val, pred)
        if r2 > best_r2:
            best_alpha, best_r2 = alpha, r2
    return fit_readout(np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]), alpha=best_alpha)


def nonlinear_demand_score(
    segments: list[np.ndarray],
    order: int = DEMAND_EMBEDDING_ORDER,
    alpha_grid: np.ndarray = DEMAND_ALPHA_GRID,
    n_folds: int = DEMAND_N_FOLDS,
    seed: int = 0,
) -> dict:
    """Frozen nonlinear-demand score (docs/eeg_preregistration.md).

    D = max(0, R2_quadratic - R2_linear) / max(R2_quadratic, epsilon),
    computed by segment-level cross-validation on the raw signal alone.
    """

    rng = np.random.default_rng(seed)
    order_idx = rng.permutation(len(segments))
    folds = np.array_split(order_idx, n_folds)

    def build(idxs, quadratic: bool):
        xs, ys = [], []
        for i in idxs:
            z, _, _ = zscore(segments[i])
            emb = _delay_embedding(z, order)
            valid = ~np.isnan(emb).any(axis=1)
            feats = _quadratic_expand(emb[valid]) if quadratic else emb[valid]
            xs.append(feats)
            ys.append(z[valid])
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    fold_lin_r2, fold_quad_r2 = [], []
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        val_cut = max(1, int(0.8 * len(train_idx)))
        inner_train, inner_val = train_idx[:val_cut], train_idx[val_cut:]
        if len(inner_val) == 0:
            inner_val = inner_train

        for quadratic, sink in ((False, fold_lin_r2), (True, fold_quad_r2)):
            x_tr, y_tr = build(inner_train, quadratic)
            x_val, y_val = build(inner_val, quadratic)
            weights = _select_alpha_and_fit(x_tr, y_tr, x_val, y_val, alpha_grid)
            x_te, y_te = build(test_idx, quadratic)
            pred = predict_readout(x_te, weights)
            sink.append(r2_score(y_te, pred))

    r2_linear = float(np.mean(fold_lin_r2))
    r2_quadratic = float(np.mean(fold_quad_r2))
    demand = max(0.0, r2_quadratic - r2_linear) / max(r2_quadratic, DEMAND_EPSILON)
    return {
        "r2_linear": r2_linear,
        "r2_quadratic": r2_quadratic,
        "nonlinear_demand": float(min(1.0, max(0.0, demand))),
        "fold_r2_linear": fold_lin_r2,
        "fold_r2_quadratic": fold_quad_r2,
    }
