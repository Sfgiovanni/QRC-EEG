"""Classical distributed-memory ESN control (follow-up, additive module).

Grafts the same causal state-mixing structure used by the QRC arms
(:mod:`qrc_eeg.state_kernels`, :func:`qrc_eeg.batched.run_batched_reservoir`'s
buffer convention) onto the classical leaky-integrator ESN
(:mod:`qrc_eeg.esn`), so the distributed-vs-concentrated-vs-no memory
contrast can be run on a dimension-matched classical substrate. See
``docs/classical_distributed_memory_protocol.md`` for the frozen protocol.

Does not modify ``qrc_eeg.esn``, ``qrc_eeg.batched``, or ``qrc_eeg.pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_kernels import (
    KernelWeights,
    matched_delay_weights,
    no_memory_weights,
    single_exponential_weights,
)

CONSTRUCTIONS = ("ESN66_K0", "ESN66_AB", "ESN66_kernel")


@dataclass(frozen=True)
class DistributedMemoryESNConfig:
    n_reservoir: int
    spectral_radius: float
    input_scale: float
    leak_rate: float
    seed: int


def base_reservoir_draw(n_reservoir: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seed-only base draw, shared across every construction and analysis mode.

    Same single rng stream and draw order as
    ``qrc_eeg.pipeline.batched_esn_features``: ``W`` (n,n) normal, then
    ``W_in`` (n,) normal, drawn from one ``numpy.random.default_rng(seed)``.
    """

    rng = np.random.default_rng(seed)
    w_raw = rng.normal(size=(n_reservoir, n_reservoir))
    w_in_raw = rng.normal(size=n_reservoir)
    return w_raw, w_in_raw


def rescale_reservoir_weights(
    w_raw: np.ndarray, w_in_raw: np.ndarray, spectral_radius: float, input_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale a shared base draw to a given (spectral_radius, input_scale)."""

    radius = np.max(np.abs(np.linalg.eigvals(w_raw)))
    w_res = w_raw * (spectral_radius / radius)
    w_in = w_in_raw * input_scale
    return w_res, w_in


def build_esn_reservoir_weights(
    n_reservoir: int, spectral_radius: float, input_scale: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Base draw + rescale in one call.

    Numerically identical to the ``W_res``/``W_in`` that
    ``qrc_eeg.pipeline.batched_esn_features`` would build for the same
    ``(n_reservoir, spectral_radius, input_scale, seed)``.
    """

    w_raw, w_in_raw = base_reservoir_draw(n_reservoir, seed)
    return rescale_reservoir_weights(w_raw, w_in_raw, spectral_radius, input_scale)


def kernel_for(name: str, kernel_hp: dict) -> KernelWeights:
    """Map a distributed-memory-ESN construction name to its frozen kernel."""

    if name == "ESN66_K0":
        return no_memory_weights()
    if name == "ESN66_AB":
        return matched_delay_weights(K=kernel_hp["tau"], tau_star=kernel_hp["tau"], past_mass=kernel_hp["delayed_mass"])
    if name == "ESN66_kernel":
        return single_exponential_weights(K=kernel_hp["K"], r=kernel_hp["r"], past_mass=kernel_hp["past_mass"])
    raise ValueError(f"unknown distributed-memory ESN construction: {name}")


def run_batched_distributed_memory_esn(
    kernel: KernelWeights,
    config: DistributedMemoryESNConfig,
    inputs: np.ndarray,  # (B, T)
) -> np.ndarray:  # (B, T, n_reservoir)
    """Vectorized evolution across a batch of independent segments.

    ``m_t = w0*x_t + sum_{tau=1..K} w_tau*x_{t-tau}``;
    ``x_{t+1} = (1-a)*m_t + a*tanh(W_res @ m_t + W_in*u_t)``. The mixture
    enters before the leak and before the nonlinearity, exactly as
    :func:`qrc_eeg.batched.run_batched_reservoir` mixes density matrices
    before its channel step; the buffer indexing convention
    (``buffer[-1-i]`` for delay ``i``, rolled with
    ``concat(buffer[1:], [new])`` every step) is identical to that function.
    """

    b, t = inputs.shape
    n = config.n_reservoir
    w_res, w_in = build_esn_reservoir_weights(n, config.spectral_radius, config.input_scale, config.seed)
    a = config.leak_rate

    state = np.zeros((b, n), dtype=np.float64)
    buffer = np.tile(state[None, :, :], (kernel.K + 1, 1, 1))  # (K+1, B, n)

    out = np.empty((b, t, n), dtype=np.float64)
    for step in range(t):
        mix = kernel.present * state
        for i, w in enumerate(kernel.delayed, start=1):
            if w:
                mix = mix + w * buffer[-1 - i]
        pre = mix @ w_res.T + inputs[:, step, None] * w_in[None, :]
        state = (1.0 - a) * mix + a * np.tanh(pre)
        buffer = np.concatenate([buffer[1:], state[None]], axis=0)
        out[:, step, :] = state
    return out


def run_sequential_distributed_memory_esn(
    kernel: KernelWeights,
    config: DistributedMemoryESNConfig,
    inputs: np.ndarray,  # (T,) single segment
) -> np.ndarray:  # (T, n_reservoir)
    """Non-batched reference implementation for one segment.

    Used only to verify the batched implementation is numerically
    equivalent (``tests/test_esn_distributed_memory.py``); never used on the
    full held-out evaluation.
    """

    n = config.n_reservoir
    w_res, w_in = build_esn_reservoir_weights(n, config.spectral_radius, config.input_scale, config.seed)
    a = config.leak_rate

    state = np.zeros(n, dtype=np.float64)
    buffer = [state.copy() for _ in range(kernel.K + 1)]

    out = np.empty((len(inputs), n), dtype=np.float64)
    for step, u in enumerate(inputs):
        mix = kernel.present * state
        for i, w in enumerate(kernel.delayed, start=1):
            if w:
                mix = mix + w * buffer[-1 - i]
        pre = w_res @ mix + w_in * float(u)
        state = (1.0 - a) * mix + a * np.tanh(pre)
        buffer = buffer[1:] + [state.copy()]
        out[step] = state
    return out


def construction_features(
    name: str,
    kernel_hp: dict,
    esn_hp: dict,
    seed: int,
    segments: np.ndarray,  # (B, T)
) -> np.ndarray:  # (B, T, n_reservoir)
    """Feature trajectories for a distributed-memory-ESN construction.

    Mirrors the signature/semantics of
    ``qrc_eeg.pipeline.construction_features`` for the new constructions
    without modifying that module.
    """

    kernel = kernel_for(name, kernel_hp)
    config = DistributedMemoryESNConfig(
        n_reservoir=esn_hp["n_reservoir"],
        spectral_radius=esn_hp["spectral_radius"],
        input_scale=esn_hp["input_scale"],
        leak_rate=esn_hp["leak_rate"],
        seed=seed,
    )
    return run_batched_distributed_memory_esn(kernel, config, segments)


def resource_accounting(name: str, kernel_hp: dict, n_reservoir: int) -> dict:
    """Trainable/fixed parameter and per-step operation accounting (Section 6)."""

    kernel = kernel_for(name, kernel_hp)
    k = kernel.K
    buffer_floats = (k + 1) * n_reservoir
    fixed_params = n_reservoir * n_reservoir + n_reservoir  # W_res + W_in
    kernel_hp_count = k + 1  # w0 + w_tau, frozen HP, not fit by ridge/gradient
    approx_ops_per_step = n_reservoir * n_reservoir + k * n_reservoir + 3 * n_reservoir
    return {
        "construction": name,
        "n_reservoir": n_reservoir,
        "K": k,
        "trainable_params_per_horizon": n_reservoir + 1,  # ridge weight + intercept-equivalent
        "fixed_untrained_params": fixed_params,
        "kernel_hp_count": kernel_hp_count,
        "buffer_floats_per_active_segment": buffer_floats,
        "approx_multiply_adds_per_step": approx_ops_per_step,
    }
