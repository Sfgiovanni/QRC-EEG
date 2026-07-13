from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qrc_eeg.pipeline import assert_disjoint_segment_ids, fit_readouts_per_horizon

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_train_validation_test_splits_are_segment_disjoint():
    for set_name in ("Z", "F", "S"):
        split = json.loads((ROOT / "data" / "eeg" / "splits" / f"{set_name}_split.json").read_text())
        train, validation, test = map(set, (split["train"], split["val"], split["test"]))
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert len(train | validation | test) == 100


def test_ridge_selection_rejects_segment_overlap():
    rng = np.random.default_rng(4)
    train_segments = rng.normal(size=(2, 80))
    validation_segments = rng.normal(size=(2, 80))
    train_features = rng.normal(size=(2, 80, 3))
    validation_features = rng.normal(size=(2, 80, 3))
    with pytest.raises(ValueError, match="segment leakage"):
        fit_readouts_per_horizon(
            train_features,
            train_segments,
            [1],
            [1e-3, 1.0],
            washout=5,
            validation_features=validation_features,
            validation_segments=validation_segments,
            train_segment_ids=["seg_a", "seg_b"],
            validation_segment_ids=["seg_b", "seg_c"],
        )


def test_group_guard_rejects_duplicates_inside_partition():
    with pytest.raises(ValueError, match="duplicate segment_id"):
        assert_disjoint_segment_ids(["seg_a", "seg_a"], ["seg_b"])
