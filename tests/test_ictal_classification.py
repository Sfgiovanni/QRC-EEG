"""Sanity checks for the ictal-classification secondary task (Item 3 of the
overnight run): split leakage and AUROC/AUPRC correctness on synthetic data.
Fast -- no reservoir simulation, so these run in every `pytest` invocation
without the ~2h feature-extraction cost of scripts/run_ictal_classification.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.classification import (  # noqa: E402
    auprc,
    auroc,
    bootstrap_auc_ci,
    fit_logistic_readout,
    mean_pool_features,
    paired_bootstrap_delta_auc,
    predict_logistic_proba,
)
from qrc_eeg.splits import assert_disjoint, load_splits  # noqa: E402

SPLITS_DIR = ROOT / "data" / "eeg" / "splits"


def test_frozen_splits_are_disjoint_every_set() -> None:
    splits = load_splits(SPLITS_DIR, ["Z", "F", "S"])
    for name, split in splits.items():
        assert_disjoint(split)  # raises on overlap
        assert len(split["train"]) > 0 and len(split["val"]) > 0 and len(split["test"]) > 0, name


def test_frozen_splits_no_cross_set_segment_id_collision() -> None:
    # A segment id from one set's test fold must never appear in another
    # set's train/val fold -- classification pools segments across Z/F/S,
    # so cross-set collisions would be leakage even though within-set splits
    # are individually disjoint.
    splits = load_splits(SPLITS_DIR, ["Z", "F", "S"])
    test_ids = set()
    trainval_ids = set()
    for split in splits.values():
        test_ids |= set(split["test"])
        trainval_ids |= set(split["train"]) | set(split["val"])
    assert test_ids.isdisjoint(trainval_ids)


def test_auroc_perfect_separation_is_one() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(y, scores) == pytest.approx(1.0)
    assert auprc(y, scores) == pytest.approx(1.0)


def test_auroc_perfect_anti_separation_is_zero() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert auroc(y, scores) == pytest.approx(0.0)


def test_auroc_matches_mann_whitney_u_on_random_data() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    scores = rng.normal(size=200) + 0.5 * y  # weak signal
    from scipy.stats import mannwhitneyu

    u = mannwhitneyu(scores[y == 1], scores[y == 0], alternative="two-sided").statistic
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    expected = u / (n_pos * n_neg)
    assert auroc(y, scores) == pytest.approx(expected, abs=1e-9)


def test_shuffled_labels_collapse_auroc_to_chance() -> None:
    # This is the leakage sanity check required by Item 3, run here on cheap
    # synthetic features so it participates in the normal pytest suite;
    # scripts/run_ictal_classification.py repeats the equivalent check on
    # real reservoir features and aborts loudly if it fails there. Must be
    # evaluated on a held-out split -- in-sample AUROC on a shuffled label
    # with 66 features and weak L2 will overfit the noise and look
    # deceptively high, which is overfitting, not leakage.
    rng = np.random.default_rng(42)
    n = 300
    n_features = 66
    y_true = (rng.uniform(size=n) < 0.33).astype(int)
    # features genuinely informative about y_true
    features = rng.normal(size=(n, n_features))
    features[:, 0] += 2.0 * y_true

    n_train = 200
    aucs = []
    for trial in range(20):
        y_shuffled = rng.permutation(y_true)
        w = fit_logistic_readout(features[:n_train], y_shuffled[:n_train], alpha=1.0)
        p_test = predict_logistic_proba(features[n_train:], w)
        aucs.append(auroc(y_shuffled[n_train:], p_test))
    mean_auc = float(np.mean(aucs))
    assert 0.35 < mean_auc < 0.65, f"held-out shuffled-label AUROC {mean_auc:.3f} did not collapse to chance -- leakage?"


def test_real_labels_recover_signal_after_shuffle_check() -> None:
    # Companion to the shuffle test: with true labels the same informative
    # features should NOT collapse to chance, confirming the shuffle test
    # actually distinguishes signal from noise rather than always failing.
    rng = np.random.default_rng(42)
    n = 300
    n_features = 66
    y_true = (rng.uniform(size=n) < 0.33).astype(int)
    features = rng.normal(size=(n, n_features))
    features[:, 0] += 2.0 * y_true

    n_train = 200
    w = fit_logistic_readout(features[:n_train], y_true[:n_train], alpha=1.0)
    p_test = predict_logistic_proba(features[n_train:], w)
    assert auroc(y_true[n_train:], p_test) > 0.8


def test_mean_pool_features_shape_and_washout() -> None:
    b, t, f = 5, 100, 66
    rng = np.random.default_rng(1)
    features = rng.normal(size=(b, t, f))
    pooled = mean_pool_features(features, washout=50)
    assert pooled.shape == (b, f)
    expected = features[:, 50:, :].mean(axis=1)
    assert np.allclose(pooled, expected)


def test_bootstrap_auc_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(7)
    n = 60
    y = (rng.uniform(size=n) < 0.33).astype(int)
    scores = rng.normal(size=n) + 1.5 * y
    result = bootstrap_auc_ci(y, scores, n_boot=500, seed=1)
    assert result["auroc_ci_lo"] <= result["auroc"] <= result["auroc_ci_hi"]
    assert result["auprc_ci_lo"] <= result["auprc"] <= result["auprc_ci_hi"]


def test_paired_bootstrap_delta_auc_zero_for_identical_scores() -> None:
    rng = np.random.default_rng(3)
    n = 60
    y = (rng.uniform(size=n) < 0.33).astype(int)
    scores = rng.normal(size=n) + 1.5 * y
    result = paired_bootstrap_delta_auc(y, scores, scores, "state", "comparator", n_boot=500, seed=1)
    assert result["delta_auroc"] == pytest.approx(0.0)
    assert result["delta_auroc_ci_lo"] <= 0.0 <= result["delta_auroc_ci_hi"]
