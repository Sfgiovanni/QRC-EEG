# Vendored from QRC-Glicose (github.com/Sfgiovanni/QRC-Glicose), MIT License, same author.
# Adapted here for the QRC-EEG study; see docs/eeg_protocol.md for provenance.
"""Patient-level paired statistical summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as st


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""

    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p[idx]
        running = max(running, adjusted)
        out[idx] = min(1.0, running)
    return out


def paired_patient_summary(
    state_kernel: pd.Series,
    comparator: pd.Series,
    state_name: str,
    comparator_name: str,
    seed: int = 1234,
) -> dict[str, float | int | str]:
    """Summarize paired patient differences.

    Positive differences mean ``RMSE_comparator - RMSE_state_kernel``.
    """

    common = sorted(set(state_kernel.index) & set(comparator.index))
    a = state_kernel.loc[common].to_numpy(dtype=float)
    b = comparator.loc[common].to_numpy(dtype=float)
    diff = b - a
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(10000)] if len(diff) else [float("nan")]
    p_w = float(st.wilcoxon(diff).pvalue) if len(diff) and np.any(np.abs(diff) > 1e-12) else 1.0
    p_t = float(st.ttest_1samp(diff, 0.0).pvalue) if len(diff) > 1 else float("nan")
    dz = float(np.mean(diff) / np.std(diff, ddof=1)) if len(diff) > 1 and np.std(diff, ddof=1) > 0 else float("nan")
    return {
        "comparison": f"{state_name} vs {comparator_name}",
        "n_patients": len(common),
        "mean_diff_rmse_comparator_minus_state": float(np.mean(diff)) if len(diff) else float("nan"),
        "median_diff": float(np.median(diff)) if len(diff) else float("nan"),
        "ci95_lo": float(np.percentile(boots, 2.5)),
        "ci95_hi": float(np.percentile(boots, 97.5)),
        "p_wilcoxon": p_w,
        "p_ttest": p_t,
        "cohen_dz": dz,
        "wins_state": int(np.sum(diff > 0)),
        "win_fraction_state": float(np.mean(diff > 0)) if len(diff) else float("nan"),
    }
