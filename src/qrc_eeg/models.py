# Vendored from QRC-Glicose (github.com/Sfgiovanni/QRC-Glicose), MIT License, same author.
# Adapted here for the QRC-EEG study; see docs/eeg_protocol.md for provenance.
"""Reservoir models for state-space memory experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .state_kernels import (
    KernelWeights,
    dual_exponential_weights,
    matched_delay_weights,
    permuted_weights,
    single_exponential_weights,
    triangular_weights,
    uniform_weights,
)

DensityMatrix = np.ndarray
Channel = Callable[[float, DensityMatrix], DensityMatrix]


def normalize_density(rho: DensityMatrix, eig_floor: float = -1e-12) -> DensityMatrix:
    """Hermitize and trace-normalize a density matrix.

    The function does not clip eigenvalues by default; it raises if numerical
    negativity is beyond a small tolerance.
    """

    out = 0.5 * (np.asarray(rho, dtype=np.complex128) + np.asarray(rho, dtype=np.complex128).conj().T)
    tr = np.trace(out)
    if abs(tr) <= 1e-14:
        raise ValueError("density matrix has zero trace")
    out = out / tr
    min_eig = float(np.linalg.eigvalsh(out).min())
    if min_eig < eig_floor:
        raise ValueError(f"density matrix is not positive semidefinite within tolerance: {min_eig}")
    return out


def identity_channel(_: float, rho: DensityMatrix) -> DensityMatrix:
    """A trace-preserving identity channel used for tests and smoke runs."""

    return normalize_density(rho)


def depolarizing_channel(strength: float = 0.02) -> Channel:
    """Return a simple input-independent depolarizing channel for smoke tests."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    def channel(_: float, rho: DensityMatrix) -> DensityMatrix:
        d = rho.shape[0]
        return normalize_density((1.0 - strength) * rho + strength * np.eye(d, dtype=np.complex128) / d)

    return channel


@dataclass
class StateDiagnostics:
    trace_error: float
    hermiticity_error: float
    min_eigenvalue: float


class StateMemoryReservoir:
    """Generic mix-first-then-channel state-memory reservoir.

    The update rule is

    ``rho_mix = w0*rho_t + sum_tau w_tau*rho_{t-tau}``

    followed by ``rho_{t+1} = E_{u_t}(rho_mix)``. The delay buffer is
    initialized with the initial density matrix, so early steps are causal and
    do not require future padding.
    """

    def __init__(
        self,
        initial_state: DensityMatrix,
        kernel: KernelWeights,
        channel: Channel | None = None,
        name: str = "state-memory-reservoir",
    ) -> None:
        self.name = name
        self.kernel = kernel
        self.channel = channel or identity_channel
        self.rho = normalize_density(initial_state)
        self._buffer: deque[DensityMatrix] = deque([self.rho.copy() for _ in range(kernel.K + 1)], maxlen=kernel.K + 1)
        self.steps = 0

    @property
    def dimension(self) -> int:
        return int(self.rho.shape[0])

    def delayed_state(self, tau: int) -> DensityMatrix:
        if tau < 0 or tau > self.kernel.K:
            raise ValueError("tau outside buffer")
        return list(self._buffer)[-1 - tau]

    def mixed_state(self) -> DensityMatrix:
        mix = self.kernel.present * self.rho
        for i, weight in enumerate(self.kernel.delayed, start=1):
            if weight:
                mix = mix + float(weight) * self.delayed_state(i)
        return normalize_density(mix)

    def step(self, input_value: float) -> DensityMatrix:
        mixed = self.mixed_state()
        self.rho = normalize_density(self.channel(float(input_value), mixed))
        self._buffer.append(self.rho.copy())
        self.steps += 1
        return self.rho

    def drive(self, inputs: np.ndarray) -> list[DensityMatrix]:
        return [self.step(float(x)).copy() for x in inputs]

    def diagnostics(self) -> StateDiagnostics:
        herm = np.linalg.norm(self.rho - self.rho.conj().T)
        return StateDiagnostics(
            trace_error=float(abs(np.trace(self.rho) - 1.0)),
            hermiticity_error=float(herm),
            min_eigenvalue=float(np.linalg.eigvalsh(self.rho).min()),
        )


class DiscreteDelayReservoir(StateMemoryReservoir):
    """AB-style reservoir with all delayed mass at one discrete lag."""

    def __init__(self, initial_state: DensityMatrix, tau: int, delayed_mass: float, channel: Channel | None = None) -> None:
        super().__init__(initial_state, matched_delay_weights(tau, tau, delayed_mass), channel, "AB-noaux-residual")


class SingleExponentialStateKernelReservoir(StateMemoryReservoir):
    """Primary parsimonious single-exponential state-memory reservoir."""

    def __init__(self, initial_state: DensityMatrix, K: int, r: float, past_mass: float, channel: Channel | None = None) -> None:
        super().__init__(initial_state, single_exponential_weights(K, r, past_mass), channel, "single-exponential-state-kernel")


class DualExponentialStateKernelReservoir(StateMemoryReservoir):
    """Dual-timescale extension of the state-memory reservoir."""

    def __init__(
        self,
        initial_state: DensityMatrix,
        K: int,
        r_fast: float,
        r_slow: float,
        fast_mass: float,
        slow_mass: float,
        channel: Channel | None = None,
    ) -> None:
        super().__init__(
            initial_state,
            dual_exponential_weights(K, r_fast, r_slow, fast_mass, slow_mass),
            channel,
            "dual-exponential-state-kernel",
        )


class UniformStateKernelReservoir(StateMemoryReservoir):
    def __init__(self, initial_state: DensityMatrix, K: int, past_mass: float, channel: Channel | None = None) -> None:
        super().__init__(initial_state, uniform_weights(K, past_mass), channel, "uniform-state-kernel")


class TriangularStateKernelReservoir(StateMemoryReservoir):
    def __init__(self, initial_state: DensityMatrix, K: int, past_mass: float, channel: Channel | None = None) -> None:
        super().__init__(initial_state, triangular_weights(K, past_mass), channel, "triangular-state-kernel")


class PermutedStateKernelReservoir(StateMemoryReservoir):
    def __init__(self, initial_state: DensityMatrix, base: KernelWeights, seed: int, channel: Channel | None = None) -> None:
        super().__init__(initial_state, permuted_weights(base, seed), channel, "permuted-state-kernel")


class MatchedDelayReservoir(StateMemoryReservoir):
    def __init__(self, initial_state: DensityMatrix, K: int, tau_star: int, past_mass: float, channel: Channel | None = None) -> None:
        super().__init__(initial_state, matched_delay_weights(K, tau_star, past_mass), channel, "matched-delay-state-kernel")


def pure_zero_state(dimension: int) -> DensityMatrix:
    """Return |0><0| in the given Hilbert-space dimension."""

    rho = np.zeros((dimension, dimension), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho
