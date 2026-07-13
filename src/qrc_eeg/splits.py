"""Deterministic segment-level train/validation/test splits, stratified by set.

New module; no analogue in either source repository (both split by patient,
not by segment). Written once per run and then frozen -- never regenerated
against the same data with a different seed mid-study.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_split(segment_ids: list[str], train_frac: float, val_frac: float, seed: int) -> dict[str, list[str]]:
    ids = sorted(segment_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_train = int(round(train_frac * len(ids)))
    n_val = int(round(val_frac * len(ids)))
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return {
        "train": sorted(ids[i] for i in train_idx),
        "val": sorted(ids[i] for i in val_idx),
        "test": sorted(ids[i] for i in test_idx),
    }


def assert_disjoint(split: dict[str, list[str]]) -> None:
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    overlap = (train & val) | (train & test) | (val & test)
    if overlap:
        raise ValueError(f"split leakage detected: {sorted(overlap)[:5]}")


def build_and_save_splits(
    sets: dict[str, list[str]], train_frac: float, val_frac: float, seed: int, out_dir: Path
) -> dict[str, dict[str, list[str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for set_name, ids in sets.items():
        split = make_split(ids, train_frac, val_frac, seed=seed)
        assert_disjoint(split)
        result[set_name] = split
        (out_dir / f"{set_name}_split.json").write_text(json.dumps(split, indent=2, sort_keys=True))
    return result


def load_splits(out_dir: Path, set_names: list[str]) -> dict[str, dict[str, list[str]]]:
    return {name: json.loads((out_dir / f"{name}_split.json").read_text()) for name in set_names}
