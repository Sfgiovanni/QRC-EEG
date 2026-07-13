"""Classical echo state network control.

Not present in either source repository (`QRC-Glicose`'s `baselines.py` only
provides EMA/lagged-window feature maps, no recurrent reservoir). Standard
leaky-integrator ESN, used as the classical-substrate control -- not as the
model the kernel construction is trying to beat (see docs/eeg_protocol.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ESNConfig:
    n_reservoir: int
    spectral_radius: float
    input_scale: float
    leak_rate: float
    seed: int


class EchoStateNetwork:
    """Leaky-integrator ESN: ``x_t = (1-a) x_{t-1} + a*tanh(W_res x_{t-1} + W_in u_t)``."""

    def __init__(self, config: ESNConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        w = rng.normal(size=(config.n_reservoir, config.n_reservoir))
        radius = np.max(np.abs(np.linalg.eigvals(w)))
        self.w_res = w * (config.spectral_radius / radius)
        self.w_in = rng.normal(size=config.n_reservoir) * config.input_scale
        self.state = np.zeros(config.n_reservoir)

    def reset(self) -> None:
        self.state = np.zeros(self.config.n_reservoir)

    def step(self, u: float) -> np.ndarray:
        a = self.config.leak_rate
        pre = self.w_res @ self.state + self.w_in * float(u)
        self.state = (1.0 - a) * self.state + a * np.tanh(pre)
        return self.state.copy()

    def drive(self, inputs: np.ndarray) -> np.ndarray:
        return np.stack([self.step(float(x)) for x in inputs])
