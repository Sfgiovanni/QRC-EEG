"""Logistic readout and ranking metrics for the ictal-classification secondary
task (pre-registered in docs/eeg_preregistration.md, "Secondary task").

New module: neither source repository nor the rest of qrc_eeg has a
classification readout (the vendored readout.py is ridge regression only) or
AUROC/AUPRC (metrics.py is forecast-only). Mirrors the conventions of
readout.py (unregularized intercept, ridge-penalized weights) and
statistics.py (default_rng(seed), 10000-resample bootstrap, percentile CIs)
rather than introducing new ones.
"""

from __future__ import annotations

import numpy as np


def mean_pool_features(features: np.ndarray, washout: int) -> np.ndarray:
    """Reduce (B, T, F) reservoir feature trajectories to (B, F) by averaging
    over time after washout -- the only pooling used, identically, across all
    constructions, so the classification readout budget stays equalized at
    the raw feature count (66 for every quantum arm, ESN-66)."""

    return features[:, washout:, :].mean(axis=1)


def _add_bias(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), np.asarray(x)])


def fit_logistic_readout(features: np.ndarray, labels: np.ndarray, alpha: float = 1.0, max_iter: int = 100, tol: float = 1e-9) -> np.ndarray:
    """L2-penalized logistic regression via Newton-Raphson (IRLS), unregularized
    intercept -- same regularization convention as readout.fit_readout."""

    x = _add_bias(features)
    y = np.asarray(labels, dtype=float)
    n, d = x.shape
    w = np.zeros(d)
    reg = np.eye(d) * float(alpha)
    reg[0, 0] = 0.0
    for _ in range(max_iter):
        z = np.clip(x @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = x.T @ (p - y) + reg @ w
        weight = np.clip(p * (1.0 - p), 1e-6, None)
        hessian = x.T @ (x * weight[:, None]) + reg
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        w_new = w - step
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def predict_logistic_proba(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = _add_bias(features)
    z = np.clip(x @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney-U form of AUROC (rank-based, average-rank tie handling)."""

    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    # average-rank tie handling
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (step-function area under precision-recall curve)."""

    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 10000, seed: int = 1234) -> dict:
    """Bootstrap over segments (the exchangeable unit here, mirroring
    statistics.paired_patient_summary's segment-level bootstrap)."""

    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    aucs, ap = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(auroc(yb, s[idx]))
        ap.append(auprc(yb, s[idx]))
    return {
        "auroc": auroc(y, s),
        "auroc_ci_lo": float(np.percentile(aucs, 2.5)),
        "auroc_ci_hi": float(np.percentile(aucs, 97.5)),
        "auprc": auprc(y, s),
        "auprc_ci_lo": float(np.percentile(ap, 2.5)),
        "auprc_ci_hi": float(np.percentile(ap, 97.5)),
        "n_boot_used": len(aucs),
    }


def paired_bootstrap_delta_auc(
    y_true: np.ndarray, scores_state: np.ndarray, scores_comparator: np.ndarray, state_name: str, comparator_name: str, n_boot: int = 10000, seed: int = 1234
) -> dict:
    """Paired bootstrap: same resampled segment indices used for both arms'
    scores each replicate, so the pairing (and any shared segment-difficulty
    noise) is preserved -- the AUROC analogue of paired_patient_summary."""

    y = np.asarray(y_true)
    sa = np.asarray(scores_state, dtype=float)
    sb = np.asarray(scores_comparator, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    d_auc, d_ap = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        d_auc.append(auroc(yb, sa[idx]) - auroc(yb, sb[idx]))
        d_ap.append(auprc(yb, sa[idx]) - auprc(yb, sb[idx]))
    d_auc = np.asarray(d_auc)
    d_ap = np.asarray(d_ap)
    p_boot_auc = float(min(1.0, 2.0 * min(np.mean(d_auc <= 0.0), np.mean(d_auc >= 0.0))))
    return {
        "comparison": f"{state_name} vs {comparator_name}",
        "n_segments": n,
        "delta_auroc": float(auroc(y, sa) - auroc(y, sb)),
        "delta_auroc_ci_lo": float(np.percentile(d_auc, 2.5)),
        "delta_auroc_ci_hi": float(np.percentile(d_auc, 97.5)),
        "p_boot_auroc": p_boot_auc,
        "delta_auprc": float(auprc(y, sa) - auprc(y, sb)),
        "delta_auprc_ci_lo": float(np.percentile(d_ap, 2.5)),
        "delta_auprc_ci_hi": float(np.percentile(d_ap, 97.5)),
        "n_boot_used": len(d_auc),
    }
