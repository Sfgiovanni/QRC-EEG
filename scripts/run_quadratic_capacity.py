#!/usr/bin/env python3
"""Frozen synthetic quadratic-capacity protocol, 5 common seeds, using each
construction's EEG-selected HP (capacity is NOT used for model selection).

Writes results/eeg/quadratic_capacity.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.channels import build_input_channel  # noqa: E402
from qrc_eeg.esn import ESNConfig  # noqa: E402
from qrc_eeg.memory_capacity import memory_target  # noqa: E402
from qrc_eeg.metrics import capacity_score  # noqa: E402
from qrc_eeg.pipeline import batched_esn_features, construction_features, kernel_for  # noqa: E402
from qrc_eeg.readout import fit_readout, predict_readout  # noqa: E402

CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
RESULTS_DIR = ROOT / "results" / "eeg"
N_QUBITS = 4
N_QUANTUM_FEATURES = 66


def n_features_for(construction: str, hp: dict) -> int:
    return hp["n_reservoir"] if construction == "ESN" else N_QUANTUM_FEATURES


def n_dof_for(construction: str, hp: dict) -> int:
    """Register size: qubits for quantum arms, reservoir units for ESN."""

    return hp["n_reservoir"] if construction == "ESN" else N_QUBITS


def select_capacity_alpha(
    feats: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    alpha_grid: list[float],
) -> float:
    """Select ridge alpha on validation rows; reserved test rows are absent."""

    best_alpha, best_score = alpha_grid[0], -np.inf
    for alpha in alpha_grid:
        weights = fit_readout(feats[train_idx], target[train_idx], alpha=alpha)
        pred = predict_readout(feats[val_idx], weights)
        score = capacity_score(target[val_idx], pred)
        if score > best_score:
            best_alpha, best_score = alpha, score
    return float(best_alpha)


def capacity_for_kind(feats: np.ndarray, u: np.ndarray, tau_values: list[int], kind: str, washout: int, alpha_grid: list[float]) -> float:
    total_capacity = 0.0
    for tau in tau_values:
        target = memory_target(u, tau=tau, kind=kind)
        valid = ~np.isnan(target)
        idx = np.where(valid)[0]
        idx = idx[idx >= washout]
        if len(idx) < 20:
            continue
        train_end = int(len(idx) * 0.6)
        val_end = int(len(idx) * 0.8)
        train_idx, val_idx, test_idx = idx[:train_end], idx[train_end:val_end], idx[val_end:]
        if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
            raise RuntimeError(f"capacity split empty for tau={tau}")
        best_alpha = select_capacity_alpha(feats, target, train_idx, val_idx, alpha_grid)
        weights = fit_readout(
            feats[np.concatenate([train_idx, val_idx])],
            target[np.concatenate([train_idx, val_idx])],
            alpha=best_alpha,
        )
        pred = predict_readout(feats[test_idx], weights)
        total_capacity += capacity_score(target[test_idx], pred)
    return total_capacity


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    selected = json.loads((RESULTS_DIR / "hp_selected.json").read_text())
    seeds = cfg["quadratic_capacity"]["seeds"]
    seq_len = cfg["quadratic_capacity"]["sequence_length"]
    tau_values = cfg["quadratic_capacity"]["tau_values"]
    washout = cfg["readout"]["washout"]
    alpha_grid = cfg["readout"]["alpha_grid"]

    quad_rows, linear_rows = [], []
    for construction, choice in selected.items():
        hp = choice["hp"]
        quad_capacities, linear_capacities = [], []
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            u = rng.uniform(-1.0, 1.0, size=(1, seq_len))
            feats = construction_features(construction, hp, seed=seed, segments=u)[0]  # (T, F), reused for both kinds

            quad_capacities.append(capacity_for_kind(feats, u[0], tau_values, "quadratic", washout, alpha_grid))
            linear_capacities.append(capacity_for_kind(feats, u[0], tau_values, "linear", washout, alpha_grid))

        for kind, values, sink in (("quadratic", quad_capacities, quad_rows), ("linear", linear_capacities, linear_rows)):
            mean_capacity = float(np.mean(values))
            sink.append(
                {
                    "construction": construction,
                    "hp": json.dumps(hp),
                    f"{kind}_capacity_mean": mean_capacity,
                    f"{kind}_capacity_std": float(np.std(values)),
                    "n_qubits_or_units": n_dof_for(construction, hp),
                    "n_features": n_features_for(construction, hp),
                    "n_seeds": len(seeds),
                }
            )
        print(
            f"{construction}: quadratic = {np.mean(quad_capacities):.4f} +/- {np.std(quad_capacities):.4f}, "
            f"linear = {np.mean(linear_capacities):.4f} +/- {np.std(linear_capacities):.4f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for kind, rows in (("quadratic", quad_rows), ("linear", linear_rows)):
        out_path = RESULTS_DIR / f"{kind}_capacity.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "construction",
                    "hp",
                    f"{kind}_capacity_mean",
                    f"{kind}_capacity_std",
                    "n_qubits_or_units",
                    "n_features",
                    "n_seeds",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
