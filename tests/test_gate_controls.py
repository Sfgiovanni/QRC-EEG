"""Gate-specific causal-control tests."""

from __future__ import annotations

import numpy as np

from qrc_eeg.pipeline import construction_features


def test_qrc_k0_dynamics_differ_from_state_memory_kernel() -> None:
    rng = np.random.default_rng(20260713)
    segments = rng.normal(size=(2, 180))
    k0 = construction_features("QRC_K0", {}, seed=7, segments=segments)
    kernel = construction_features(
        "single_kernel",
        {"K": 15, "r": 0.7, "past_mass": 0.3},
        seed=7,
        segments=segments,
    )
    assert k0.shape == kernel.shape
    assert not np.allclose(k0, kernel, rtol=1e-8, atol=1e-10)
    assert float(np.max(np.abs(k0 - kernel))) > 1e-6
