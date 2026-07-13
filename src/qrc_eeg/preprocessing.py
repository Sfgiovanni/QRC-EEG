"""Leak-free EEG scaling fitted only on frozen training segments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingScaler:
    """Global scalar location/scale estimated from training samples only."""

    mean: float
    std: float
    n_training_samples: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std


def fit_training_scaler(training_segments: np.ndarray) -> TrainingScaler:
    """Fit one scaler without consulting validation or test segments."""

    train = np.asarray(training_segments, dtype=np.float64)
    if train.ndim != 2 or train.size == 0:
        raise ValueError("training_segments must be a non-empty (segments, samples) array")
    if not np.isfinite(train).all():
        raise ValueError("training_segments contain non-finite values")
    mean = float(np.mean(train))
    std = float(np.std(train))
    if std <= 1e-12:
        raise ValueError("training-only scaling failed: near-zero standard deviation")
    return TrainingScaler(mean=mean, std=std, n_training_samples=int(train.size))


def scale_set_from_training(
    raw_segments: dict[str, list[float] | np.ndarray], training_ids: list[str]
) -> tuple[dict[str, np.ndarray], TrainingScaler]:
    """Scale every segment using statistics from ``training_ids`` only."""

    missing = sorted(set(training_ids) - set(raw_segments))
    if missing:
        raise KeyError(f"training segment ids missing from raw set: {missing}")
    train = np.stack([np.asarray(raw_segments[sid], dtype=np.float64) for sid in training_ids])
    scaler = fit_training_scaler(train)
    scaled = {sid: scaler.transform(values) for sid, values in raw_segments.items()}
    return scaled, scaler
