"""QRC-EEG: quantum reservoir memory-kernel comparison on Bonn EEG data.

Vendored numerical core (state_kernels, models, metrics, statistics, readout,
memory_capacity, provenance, stability, observables) comes from
`QRC-Glicose` (MIT license, same author); see docs/eeg_protocol.md for the
full provenance and for the modeling choices (input-encoding channel, ESN)
that had no existing implementation to reuse.
"""

from .state_kernels import (
    KernelWeights,
    dual_exponential_weights,
    kernel_characteristics,
    kernel_mean_lag,
    matched_delay_weights,
    single_exponential_weights,
    triangular_weights,
    uniform_weights,
)
from .models import (
    DiscreteDelayReservoir,
    DualExponentialStateKernelReservoir,
    SingleExponentialStateKernelReservoir,
    StateMemoryReservoir,
    TriangularStateKernelReservoir,
    UniformStateKernelReservoir,
    pure_zero_state,
)
from .channels import build_input_channel, InputEncodingChannel
from .esn import EchoStateNetwork, ESNConfig
from .features import quantum_features

__all__ = [
    "KernelWeights",
    "single_exponential_weights",
    "dual_exponential_weights",
    "uniform_weights",
    "triangular_weights",
    "matched_delay_weights",
    "kernel_mean_lag",
    "kernel_characteristics",
    "StateMemoryReservoir",
    "DiscreteDelayReservoir",
    "SingleExponentialStateKernelReservoir",
    "DualExponentialStateKernelReservoir",
    "UniformStateKernelReservoir",
    "TriangularStateKernelReservoir",
    "pure_zero_state",
    "build_input_channel",
    "InputEncodingChannel",
    "EchoStateNetwork",
    "ESNConfig",
    "quantum_features",
]

__version__ = "0.1.0"
