"""End-to-end nested-CV / HP-search / held-out evaluation pipeline.

New module tying together the vendored mixing reservoir, the new channel/ESN,
and the new EEG tasks. No analogue in either source repository (both drive
one segment through one fixed config at a time; there is no HP-search
harness or batched multi-segment evaluation loop in either).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .batched import run_batched_reservoir
from .channels import build_input_channel
from .esn import ESNConfig
from .models import pure_zero_state
from .state_kernels import (
    KernelWeights,
    dual_exponential_weights,
    matched_delay_weights,
    single_exponential_weights,
    triangular_weights,
    uniform_weights,
)
from .metrics import mae, rmse
from .readout import fit_readout, predict_readout
from .tasks import forecast_target, nrmse, r2_score, zscore

N_QUBITS = 4
WASHOUT = 50


def kernel_for(name: str, hp: dict) -> KernelWeights | None:
    if name == "AB_noaux":
        return matched_delay_weights(K=hp["tau"], tau_star=hp["tau"], past_mass=hp["delayed_mass"])
    if name == "single_kernel":
        return single_exponential_weights(K=hp["K"], r=hp["r"], past_mass=hp["past_mass"])
    if name == "dual_kernel":
        return dual_exponential_weights(
            K=hp["K"], r_fast=hp["r_fast"], r_slow=hp["r_slow"], fast_mass=hp["fast_mass"], slow_mass=hp["slow_mass"]
        )
    if name == "triangular":
        return triangular_weights(K=hp["K"], past_mass=hp["past_mass"])
    if name == "uniform":
        return uniform_weights(K=hp["K"], past_mass=hp["past_mass"])
    if name == "ESN":
        return None
    raise ValueError(f"unknown construction: {name}")


def hp_grid_combinations(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def batched_esn_features(esn_cfg: ESNConfig, inputs: np.ndarray) -> np.ndarray:
    """Vectorized leaky-integrator ESN evolution over a batch of segments."""

    b, t = inputs.shape
    rng = np.random.default_rng(esn_cfg.seed)
    w = rng.normal(size=(esn_cfg.n_reservoir, esn_cfg.n_reservoir))
    radius = np.max(np.abs(np.linalg.eigvals(w)))
    w_res = w * (esn_cfg.spectral_radius / radius)
    w_in = rng.normal(size=esn_cfg.n_reservoir) * esn_cfg.input_scale

    state = np.zeros((b, esn_cfg.n_reservoir))
    out = np.empty((b, t, esn_cfg.n_reservoir), dtype=np.float64)
    a = esn_cfg.leak_rate
    for step in range(t):
        pre = state @ w_res.T + inputs[:, step, None] * w_in[None, :]
        state = (1.0 - a) * state + a * np.tanh(pre)
        out[:, step, :] = state
    return out


def construction_features(
    name: str,
    hp: dict,
    seed: int,
    segments: np.ndarray,  # (B, T) raw z-scored signal per segment
) -> np.ndarray:
    """Return (B, T, F) feature trajectories for the given construction/HP/seed."""

    if name == "ESN":
        cfg = ESNConfig(
            n_reservoir=hp["n_reservoir"],
            spectral_radius=hp["spectral_radius"],
            input_scale=hp["input_scale"],
            leak_rate=hp["leak_rate"],
            seed=seed,
        )
        return batched_esn_features(cfg, segments)

    channel = build_input_channel(n_qubits=N_QUBITS, seed=seed)
    kernel = kernel_for(name, hp)
    init = pure_zero_state(2**N_QUBITS)
    result = run_batched_reservoir(kernel, channel, init, segments, check_every=500)
    return result.features


@dataclass
class HorizonFit:
    horizon: int
    alpha: float
    weights: np.ndarray


def _pool_rows(features: np.ndarray, target: np.ndarray, washout: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    b, t, f = features.shape
    end = t - horizon
    feats = features[:, washout:end, :].reshape(-1, f)
    targs = target[:, washout:end].reshape(-1)
    valid = ~np.isnan(targs)
    return feats[valid], targs[valid]


def fit_readouts_per_horizon(
    features: np.ndarray, segments: np.ndarray, horizons: list[int], alpha_grid: list[float], washout: int = WASHOUT
) -> dict[int, HorizonFit]:
    """Fit one pooled ridge readout per horizon, selecting alpha by in-sample-free grid search.

    `segments` here is the z-scored raw signal (B, T) that the targets are
    derived from; `features` is aligned reservoir output for the same B, T.
    Alpha selection uses the same rows passed in (caller is responsible for
    passing only train-side data when fitting, or train+val split rows when
    selecting alpha against a held validation slice).
    """

    fits = {}
    for h in horizons:
        target = np.stack([forecast_target(seg, h) for seg in segments])
        x, y = _pool_rows(features, target, washout, h)
        best_alpha, best_err = alpha_grid[0], np.inf
        # simple 80/20 row split for alpha selection (rows already pooled/shuffled across segments)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(x))
        cut = max(1, int(0.8 * len(idx)))
        tr_idx, val_idx = idx[:cut], idx[cut:]
        for alpha in alpha_grid:
            w = fit_readout(x[tr_idx], y[tr_idx], alpha=alpha)
            pred = predict_readout(x[val_idx], w)
            err = float(np.sqrt(np.mean((pred - y[val_idx]) ** 2)))
            if err < best_err:
                best_alpha, best_err = alpha, err
        final_w = fit_readout(x, y, alpha=best_alpha)
        fits[h] = HorizonFit(horizon=h, alpha=best_alpha, weights=final_w)
    return fits


def evaluate_segments(
    features: np.ndarray, segments: np.ndarray, fits: dict[int, HorizonFit], washout: int = WASHOUT
) -> dict[int, np.ndarray]:
    """Return per-segment NRMSE array for each horizon (used by HP search)."""

    return {h: metrics["nrmse"] for h, metrics in evaluate_segments_full(features, segments, fits, washout).items()}


def evaluate_segments_full(
    features: np.ndarray, segments: np.ndarray, fits: dict[int, HorizonFit], washout: int = WASHOUT
) -> dict[int, dict[str, np.ndarray]]:
    """Return per-segment NRMSE/RMSE/R2/MAE arrays for each horizon."""

    out = {}
    for h, fit in fits.items():
        b, t, f = features.shape
        end = t - h
        target = np.stack([forecast_target(seg, h) for seg in segments])
        per_metric = {name: np.full(b, np.nan) for name in ("nrmse", "rmse", "r2", "mae")}
        for i in range(b):
            feats_i = features[i, washout:end, :]
            targ_i = target[i, washout:end]
            valid = ~np.isnan(targ_i)
            pred_i = predict_readout(feats_i[valid], fit.weights)
            per_metric["nrmse"][i] = nrmse(targ_i[valid], pred_i)
            per_metric["rmse"][i] = rmse(targ_i[valid], pred_i)
            per_metric["r2"][i] = r2_score(targ_i[valid], pred_i)
            per_metric["mae"][i] = mae(targ_i[valid], pred_i)
        out[h] = per_metric
    return out
