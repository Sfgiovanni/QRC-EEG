# Vendored from QRC-Glicose (github.com/Sfgiovanni/QRC-Glicose), MIT License, same author.
# Adapted here for the QRC-EEG study; see docs/eeg_protocol.md for provenance.
"""State-memory kernel construction.

All kernels return non-negative delayed-state weights for delays
``tau = 1, ..., K`` and a present-state weight ``w0`` such that

    w0 + sum_tau w_tau = 1.

The reservoir update mixes states first and applies the input-dependent
channel once after the mixture.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class KernelWeights:
    """Normalized kernel weights for a causal state-memory buffer."""

    delayed: np.ndarray
    present: float
    kind: str
    metadata: dict[str, Any]

    @property
    def K(self) -> int:
        return int(len(self.delayed))

    @property
    def past_mass(self) -> float:
        return float(np.sum(self.delayed))

    @property
    def total_mass(self) -> float:
        return float(self.present + self.past_mass)

    def hash(self) -> str:
        payload = {
            "kind": self.kind,
            "present": self.present,
            "delayed": [float(x) for x in self.delayed],
            "metadata": self.metadata,
        }
        return config_hash(payload)


def config_hash(obj: Mapping[str, Any]) -> str:
    """Return a deterministic short hash for a configuration mapping."""

    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _normalize(shape: np.ndarray, past_mass: float, kind: str, metadata: dict[str, Any]) -> KernelWeights:
    if past_mass < 0.0 or past_mass > 1.0:
        raise ValueError("past_mass must be in [0, 1]")
    raw = np.asarray(shape, dtype=np.float64)
    if raw.ndim != 1 or len(raw) == 0:
        raise ValueError("kernel shape must be a non-empty one-dimensional array")
    if np.any(raw < 0.0):
        raise ValueError("kernel shape must be non-negative")
    total = float(np.sum(raw))
    delayed = np.zeros_like(raw, dtype=np.float64) if total == 0.0 else past_mass * raw / total
    present = max(0.0, 1.0 - float(np.sum(delayed)))
    return KernelWeights(delayed=delayed, present=present, kind=kind, metadata=dict(metadata))


def single_exponential_weights(K: int, r: float, past_mass: float) -> KernelWeights:
    """Create weights proportional to ``r**tau`` for ``tau=1..K``."""

    if K < 1:
        raise ValueError("K must be positive")
    if not 0.0 < r < 1.0:
        raise ValueError("r must be in (0, 1)")
    tau = np.arange(1, K + 1, dtype=np.float64)
    return _normalize(r**tau, past_mass, "single_exponential", {"K": K, "r": float(r)})


def dual_exponential_weights(
    K: int,
    r_fast: float,
    r_slow: float,
    fast_mass: float,
    slow_mass: float,
) -> KernelWeights:
    """Create a two-scale kernel with separately normalized fast and slow masses."""

    if K < 1:
        raise ValueError("K must be positive")
    if not 0.0 < r_fast < r_slow < 1.0:
        raise ValueError("required 0 < r_fast < r_slow < 1")
    if fast_mass < 0.0 or slow_mass < 0.0 or fast_mass + slow_mass > 1.0:
        raise ValueError("fast_mass and slow_mass must be non-negative and sum to at most 1")
    tau = np.arange(1, K + 1, dtype=np.float64)
    fast_shape = r_fast**tau
    slow_shape = r_slow**tau
    delayed = np.zeros(K, dtype=np.float64)
    delayed += fast_mass * fast_shape / float(np.sum(fast_shape))
    delayed += slow_mass * slow_shape / float(np.sum(slow_shape))
    present = max(0.0, 1.0 - float(np.sum(delayed)))
    return KernelWeights(
        delayed=delayed,
        present=present,
        kind="dual_exponential",
        metadata={
            "K": K,
            "r_fast": float(r_fast),
            "r_slow": float(r_slow),
            "fast_mass": float(fast_mass),
            "slow_mass": float(slow_mass),
        },
    )


def uniform_weights(K: int, past_mass: float) -> KernelWeights:
    """Create a uniform delayed-state kernel."""

    return _normalize(np.ones(K, dtype=np.float64), past_mass, "uniform", {"K": int(K)})


def triangular_weights(K: int, past_mass: float) -> KernelWeights:
    """Create linearly decreasing weights favoring recent states."""

    tau = np.arange(1, K + 1, dtype=np.float64)
    return _normalize(K - tau + 1.0, past_mass, "triangular_recent", {"K": int(K)})


def permuted_weights(base: KernelWeights, seed: int) -> KernelWeights:
    """Permute a kernel's weights with a fixed seed while preserving values and mass."""

    rng = np.random.default_rng(seed)
    permuted = base.delayed[rng.permutation(base.K)]
    return KernelWeights(
        delayed=permuted,
        present=base.present,
        kind="permuted",
        metadata={"base_kind": base.kind, "permutation_seed": int(seed), **base.metadata},
    )


def matched_delay_weights(K: int, tau_star: int, past_mass: float) -> KernelWeights:
    """Place all delayed mass at a single causal delay."""

    if not 1 <= tau_star <= K:
        raise ValueError("tau_star must be in [1, K]")
    shape = np.zeros(K, dtype=np.float64)
    shape[tau_star - 1] = 1.0
    return _normalize(shape, past_mass, "single_delay_matched", {"K": int(K), "tau_star": int(tau_star)})


def kernel_mean_lag(weights: KernelWeights | np.ndarray) -> float:
    """Return the mean delay among delayed weights."""

    delayed = weights.delayed if isinstance(weights, KernelWeights) else np.asarray(weights, dtype=np.float64)
    mass = float(np.sum(delayed))
    if mass <= 0.0:
        return 0.0
    tau = np.arange(1, len(delayed) + 1, dtype=np.float64)
    return float(np.sum(tau * delayed) / mass)


def kernel_characteristics(weights: KernelWeights, step_minutes: float = 15.0) -> dict[str, float | str]:
    """Summarize lag distribution and characteristic times."""

    delayed = weights.delayed
    tau = np.arange(1, len(delayed) + 1, dtype=np.float64)
    mass = max(float(np.sum(delayed)), 1e-12)
    cum = np.cumsum(delayed) / mass
    mean = kernel_mean_lag(weights)
    var = float(np.sum(((tau - mean) ** 2) * delayed) / mass)

    def lag_at(p: float) -> int:
        return int(tau[min(int(np.searchsorted(cum, p)), len(tau) - 1)])

    r = weights.metadata.get("r")
    characteristic_steps = float("nan")
    characteristic_minutes = float("nan")
    if isinstance(r, (float, int)) and 0.0 < float(r) < 1.0:
        characteristic_steps = float(-1.0 / math.log(float(r)))
        characteristic_minutes = float(step_minutes * characteristic_steps)
    return {
        "kernel": weights.kind,
        "K": weights.K,
        "past_mass": weights.past_mass,
        "present_mass": weights.present,
        "mean_lag_steps": mean,
        "variance_lag_steps2": var,
        "lag_50": lag_at(0.50),
        "lag_80": lag_at(0.80),
        "lag_90": lag_at(0.90),
        "lag_95": lag_at(0.95),
        "characteristic_steps": characteristic_steps,
        "characteristic_minutes": characteristic_minutes,
    }


def causal_weighted_history(values: np.ndarray, weights: KernelWeights | np.ndarray) -> np.ndarray:
    """Return a causal weighted history using only samples before each time."""

    x = np.asarray(values, dtype=np.float64)
    delayed = weights.delayed if isinstance(weights, KernelWeights) else np.asarray(weights, dtype=np.float64)
    out = np.empty(len(x), dtype=np.float64)
    for t in range(len(x)):
        max_lag = min(t, len(delayed))
        if max_lag == 0:
            out[t] = x[t]
            continue
        w = delayed[:max_lag]
        denom = float(np.sum(w))
        lagged = x[t - np.arange(1, max_lag + 1)]
        out[t] = float(np.dot(w, lagged) / denom) if denom > 0.0 else x[t]
    return out
