from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from qrc_eeg.preprocessing import fit_training_scaler, scale_set_from_training

ROOT = Path(__file__).resolve().parents[1]


def _load_capacity_script():
    path = ROOT / "scripts" / "run_quadratic_capacity.py"
    spec = importlib.util.spec_from_file_location("run_quadratic_capacity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_scaler_is_invariant_to_test_future_samples():
    raw = {
        "train_a": np.array([1.0, 2.0, 3.0, 4.0]),
        "train_b": np.array([2.0, 3.0, 4.0, 5.0]),
        "test": np.array([10.0, 11.0, 12.0, 13.0]),
    }
    scaled_a, scaler_a = scale_set_from_training(raw, ["train_a", "train_b"])
    changed = dict(raw)
    changed["test"] = np.array([10.0, 11.0, 1e9, -1e9])
    scaled_b, scaler_b = scale_set_from_training(changed, ["train_a", "train_b"])

    assert scaler_a == scaler_b
    np.testing.assert_array_equal(scaled_a["test"][:2], scaled_b["test"][:2])
    np.testing.assert_allclose(
        fit_training_scaler(np.stack([raw["train_a"], raw["train_b"]])).transform(raw["test"]),
        scaled_a["test"],
    )


def test_capacity_alpha_selection_cannot_see_reserved_test_target():
    module = _load_capacity_script()
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(120, 5))
    target = rng.normal(size=120)
    train_idx = np.arange(0, 60)
    val_idx = np.arange(60, 90)
    alpha_grid = [1e-6, 1e-3, 1.0, 100.0]

    alpha_a = module.select_capacity_alpha(feats, target, train_idx, val_idx, alpha_grid)
    target_with_changed_test = target.copy()
    target_with_changed_test[90:] = 1e12
    alpha_b = module.select_capacity_alpha(feats, target_with_changed_test, train_idx, val_idx, alpha_grid)
    assert alpha_a == alpha_b
