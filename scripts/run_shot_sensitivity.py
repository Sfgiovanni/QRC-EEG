#!/usr/bin/env python3
"""Gate 3 exact-baseline reproduction and finite-binomial-shot EEG analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.eeg_data import load_set  # noqa: E402
from qrc_eeg.metrics import mae, rmse  # noqa: E402
from qrc_eeg.pipeline import construction_features, evaluate_segments_full, fit_readouts_per_horizon  # noqa: E402
from qrc_eeg.preprocessing import scale_set_from_training  # noqa: E402
from qrc_eeg.readout import fit_readout, predict_readout  # noqa: E402
from qrc_eeg.splits import load_splits  # noqa: E402
from qrc_eeg.tasks import nrmse, r2_score  # noqa: E402

CONFIG = ROOT / "config/rotaA_gate3_frozen.json"
PROTOCOL = ROOT / "docs/gate3_protocol.md"
EEG_CONFIG = ROOT / "config/eeg_frozen.yaml"
RESULTS = ROOT / "results/eeg"
CACHE = Path(os.environ.get("QRC_GATE3_CACHE", "/tmp/qrc_eeg_gate3_cache"))
RAW_PATH = RESULTS / "shot_sensitivity_raw.csv"
EXACT_FULL_PATH = RESULTS / "shot_sensitivity_exact_full.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_rng(*parts) -> np.random.Generator:
    payload = ":".join(map(str, parts)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))


def preflight() -> tuple[dict, dict, str, dict, dict, dict]:
    cfg = json.loads(CONFIG.read_text())
    for line in (ROOT / "results/resources/gate3_protocol_frozen.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        if sha(ROOT / relative) != expected:
            raise SystemExit(f"INVALID_PROVENANCE: Gate 3 freeze changed: {relative}")
    for relative, expected in cfg["reference_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise SystemExit(f"INVALID_PROVENANCE: frozen reference changed: {relative}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != cfg["git_commit"]:
        raise SystemExit(f"INVALID_CONFIG: commit differs: {commit}")
    selected = json.loads(subprocess.check_output(
        ["git", "show", f"{commit}:results/eeg/hp_selected.json"], cwd=ROOT, text=True
    ))
    hp = {"QRC_K0": {}, **{m: selected[m]["hp"] for m in cfg["resource_models"] if m != "QRC_K0"}}
    if canonical_hash(hp) != cfg["official_hp_sha256"]:
        raise SystemExit("INVALID_CONFIG: committed HP mapping differs")
    eeg_cfg = yaml.safe_load(EEG_CONFIG.read_text())
    if cfg["horizons"] != eeg_cfg["readout"]["horizons"] or cfg["alpha_grid"] != eeg_cfg["readout"]["alpha_grid"]:
        raise SystemExit("INVALID_CONFIG: Gate 3 readout grid differs from frozen EEG configuration")
    splits = load_splits(ROOT / "data/eeg/splits", cfg["sets"])
    raw = {name: load_set(ROOT / "data/eeg/sets" / name) for name in cfg["sets"]}
    scaled = {name: scale_set_from_training(raw[name], splits[name]["train"])[0] for name in cfg["sets"]}
    return cfg, hp, commit, eeg_cfg, splits, scaled


def cache_paths(set_name: str, model: str, seed: int) -> tuple[Path, Path, Path]:
    stem = f"{set_name}__{model}__seed{seed}"
    return tuple(CACHE / f"{stem}__{part}.npy" for part in ("train", "val", "test"))


def segment_arrays(set_name, splits, scaled):
    ids = tuple(splits[set_name][part] for part in ("train", "val", "test"))
    arrays = tuple(np.stack([scaled[set_name][segment] for segment in part_ids]) for part_ids in ids)
    return ids, arrays


def strided_indices(length: int, washout: int, stride: int) -> np.ndarray:
    return np.arange(washout, length - 1, stride, dtype=int)


def fit_strided(train_f, train, val_f, val, horizon, indices, alpha_grid):
    valid = indices < train.shape[1] - horizon
    idx = indices[valid]
    xtr = train_f[:, valid, :].reshape(-1, train_f.shape[-1]); ytr = train[:, idx + horizon].reshape(-1)
    xv = val_f[:, valid, :].reshape(-1, val_f.shape[-1]); yv = val[:, idx + horizon].reshape(-1)
    best_alpha, best_score = alpha_grid[0], np.inf
    for alpha in alpha_grid:
        weights = fit_readout(xtr, ytr, alpha)
        score = nrmse(yv, predict_readout(xv, weights))
        if score < best_score:
            best_alpha, best_score = alpha, score
    weights = fit_readout(np.vstack([xtr, xv]), np.concatenate([ytr, yv]), best_alpha)
    return float(best_alpha), weights


def evaluate_strided(test_f, test, horizon, indices, weights):
    valid = indices < test.shape[1] - horizon
    idx = indices[valid]
    output = {name: np.empty(len(test)) for name in ("nrmse", "rmse", "r2", "mae")}
    for segment in range(len(test)):
        target = test[segment, idx + horizon]
        predicted = predict_readout(test_f[segment, valid], weights)
        output["nrmse"][segment] = nrmse(target, predicted)
        output["rmse"][segment] = rmse(target, predicted)
        output["r2"][segment] = r2_score(target, predicted)
        output["mae"][segment] = mae(target, predicted)
    return output


def pauli_shot_sample(features: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    mu = np.clip(np.asarray(features, dtype=np.float64), -1.0, 1.0)
    counts = rng.binomial(shots, (1.0 + mu) / 2.0)
    return 2.0 * counts / shots - 1.0


def raw_rows(metrics, set_name, model, horizon, seed, replicate, shots, test_ids, alpha):
    return [{
        "set": set_name, "model": model, "horizon": horizon, "channel_seed": seed,
        "noise_replicate": replicate, "shots": shots, "shot_label": "exact" if shots == 0 else str(shots),
        "segment_id": segment_id, "selected_alpha": alpha,
        **{metric: float(values[index]) for metric, values in metrics.items()},
    } for index, segment_id in enumerate(test_ids)]


def reproduce_baseline(cfg, hp, eeg_cfg, splits, scaled) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    exact_rows, strided_rows = [], []
    for set_name in cfg["sets"]:
        ids, arrays = segment_arrays(set_name, splits, scaled)
        train_ids, val_ids, test_ids = ids; train, val, test = arrays
        indices = strided_indices(train.shape[1], cfg["washout"], cfg["shot_temporal_stride"])
        for model in cfg["main_models"]:
            for seed in cfg["channel_seeds"]:
                paths = cache_paths(set_name, model, seed)
                if all(path.exists() for path in paths):
                    features = tuple(np.load(path, mmap_mode="r") for path in paths)
                else:
                    features = tuple(construction_features(model, hp[model], seed, part) for part in arrays)
                    for path, values in zip(paths, features):
                        np.save(path, values)
                train_f, val_f, test_f = features
                fits = fit_readouts_per_horizon(
                    train_f, train, cfg["horizons"], cfg["alpha_grid"], washout=cfg["washout"],
                    validation_features=val_f, validation_segments=val,
                    train_segment_ids=train_ids, validation_segment_ids=val_ids,
                )
                full_metrics = evaluate_segments_full(test_f, test, fits, washout=cfg["washout"])
                for horizon, metrics in full_metrics.items():
                    exact_rows.extend(raw_rows(metrics, set_name, model, horizon, seed, 0, 0, test_ids, fits[horizon].alpha))
                    alpha, weights = fit_strided(
                        train_f[:, indices], train, val_f[:, indices], val, horizon, indices,
                        cfg["alpha_grid"],
                    )
                    metrics_s = evaluate_strided(test_f[:, indices], test, horizon, indices, weights)
                    strided_rows.extend(raw_rows(metrics_s, set_name, model, horizon, seed, 0, 0, test_ids, alpha))
                print(f"exact baseline/cache complete: {set_name} {model} seed={seed}", flush=True)
    exact = pd.DataFrame(exact_rows); exact.to_csv(EXACT_FULL_PATH, index=False)
    strided = pd.DataFrame(strided_rows); strided.to_csv(RAW_PATH, index=False)
    references = []
    snapshot = pd.read_csv(ROOT / cfg["baseline_references"]["official_r07_h1_to_h8"])
    current = pd.read_csv(ROOT / cfg["baseline_references"]["k0_and_ab_extended"])
    references.append(snapshot[snapshot.construction.isin(["single_kernel", "AB_noaux"]) & snapshot.seed.isin(cfg["channel_seeds"])])
    references.append(current[current.construction.isin(["QRC_K0", "AB_noaux"]) & current.seed.isin(cfg["channel_seeds"])])
    reference = pd.concat(references).drop_duplicates(["construction", "set", "horizon", "seed", "segment_id"])
    merged = exact.merge(reference, left_on=["model", "set", "horizon", "channel_seed", "segment_id"],
                         right_on=["construction", "set", "horizon", "seed", "segment_id"], suffixes=("_new", "_reference"))
    merged["abs_nrmse_difference"] = (merged.nrmse_new - merged.nrmse_reference).abs()
    reproduction = merged[["model", "set", "horizon", "channel_seed", "segment_id", "nrmse_new",
                           "nrmse_reference", "abs_nrmse_difference"]]
    reproduction.to_csv(RESULTS / "shot_baseline_reproduction.csv", index=False)
    expected_rows = len(reference)
    maximum = float(reproduction.abs_nrmse_difference.max())
    print(f"baseline matched rows={len(reproduction)}/{expected_rows}; max_abs_nrmse={maximum:.3e}")
    if len(reproduction) != expected_rows or maximum > cfg["baseline_tolerance_max_abs_nrmse"]:
        raise SystemExit("INVALID_BASELINE_REPRODUCTION")
    (RESULTS / "shot_baseline_reproduction_status.json").write_text(json.dumps({
        "status": "PASS", "matched_rows": len(reproduction), "expected_rows": expected_rows,
        "max_abs_nrmse_difference": maximum, "tolerance": cfg["baseline_tolerance_max_abs_nrmse"],
        "extended_single_kernel_r07_reference": "unavailable_and_explicitly_not_claimed",
    }, indent=2) + "\n")


def run_shots(cfg, hp, splits, scaled) -> None:
    status = json.loads((RESULTS / "shot_baseline_reproduction_status.json").read_text())
    if status["status"] != "PASS":
        raise SystemExit("INVALID_BASELINE_REPRODUCTION")
    fieldnames = ["set", "model", "horizon", "channel_seed", "noise_replicate", "shots", "shot_label",
                  "segment_id", "selected_alpha", "nrmse", "rmse", "r2", "mae"]
    with RAW_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for set_name in cfg["sets"]:
            ids, arrays = segment_arrays(set_name, splits, scaled)
            _, _, test_ids = ids; train, val, test = arrays
            indices = strided_indices(train.shape[1], cfg["washout"], cfg["shot_temporal_stride"])
            for model in cfg["main_models"]:
                for seed in cfg["channel_seeds"]:
                    train_f, val_f, test_f = (np.load(path, mmap_mode="r")[:, indices] for path in cache_paths(set_name, model, seed))
                    for shots in cfg["shots"]:
                        for replicate in range(1, cfg["noise_replicates"] + 1):
                            noisy_train = pauli_shot_sample(train_f, shots, stable_rng("train", set_name, model, seed, shots, replicate))
                            noisy_val = pauli_shot_sample(val_f, shots, stable_rng("val", set_name, model, seed, shots, replicate))
                            noisy_test = pauli_shot_sample(test_f, shots, stable_rng("test", set_name, model, seed, shots, replicate))
                            for horizon in cfg["horizons"]:
                                alpha, weights = fit_strided(noisy_train, train, noisy_val, val, horizon, indices, cfg["alpha_grid"])
                                metrics = evaluate_strided(noisy_test, test, horizon, indices, weights)
                                writer.writerows(raw_rows(metrics, set_name, model, horizon, seed, replicate,
                                                          shots, test_ids, alpha))
                            handle.flush()
                        print(f"shots complete: {set_name} {model} seed={seed} N={shots}", flush=True)


def bootstrap_ci(values: np.ndarray, label: str, resamples: int) -> tuple[float, float]:
    rng = stable_rng("bootstrap", label)
    boot = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def classify_shot_sensitivity(levels: pd.DataFrame, any_set_pass: bool) -> str:
    passing = levels[levels.global_pass]
    if len(passing):
        return f"ROBUST_AT_{int(passing.shots.min())}_SHOTS"
    return "MIXED_SHOT_SENSITIVITY" if any_set_pass else "SHOT_SENSITIVE_UP_TO_10000"


def persistence_strided(cfg, splits, scaled) -> pd.DataFrame:
    rows = []
    for set_name in cfg["sets"]:
        test_ids = splits[set_name]["test"]
        test = np.stack([scaled[set_name][sid] for sid in test_ids])
        indices = strided_indices(test.shape[1], cfg["washout"], cfg["shot_temporal_stride"])
        for horizon in cfg["horizons"]:
            idx = indices[indices < test.shape[1] - horizon]
            for i, segment_id in enumerate(test_ids):
                rows.append({"set": set_name, "horizon": horizon, "segment_id": segment_id,
                             "nrmse": nrmse(test[i, idx + horizon], test[i, idx])})
    return pd.DataFrame(rows)


def summarize(cfg, splits, scaled, commit) -> tuple[str, str]:
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    if "relative_nrmse_inflation" in raw.columns:
        finite = raw[raw.shots > 0].copy()
    else:
        exact = raw[raw.shots == 0].drop(columns=["noise_replicate", "shot_label", "selected_alpha", "rmse", "r2", "mae"])
        finite = raw[raw.shots > 0].merge(
            exact, on=["set", "model", "horizon", "channel_seed", "segment_id"], suffixes=("", "_exact"), validate="many_to_one"
        )
        finite["nrmse_difference"] = finite.nrmse - finite.nrmse_exact
        finite["relative_nrmse_inflation"] = finite.nrmse_difference / finite.nrmse_exact
        # Rewrite enriched raw, retaining exact rows with zero differences.
        exact_out = raw[raw.shots == 0].copy(); exact_out["nrmse_exact"] = exact_out.nrmse
        exact_out["nrmse_difference"] = 0.0; exact_out["relative_nrmse_inflation"] = 0.0
        raw = pd.concat([exact_out, finite[exact_out.columns]], ignore_index=True)
        raw.to_csv(RAW_PATH, index=False)
    summary_rows = []
    for keys, group in finite.groupby(["set", "model", "horizon", "shots"]):
        per_segment = group.groupby("segment_id").agg(nrmse_difference=("nrmse_difference", "mean"),
                                                       relative=("relative_nrmse_inflation", "mean"))
        lo, hi = bootstrap_ci(per_segment.nrmse_difference.to_numpy(), ":".join(map(str, keys)), cfg["bootstrap_resamples"])
        summary_rows.append({
            "set": keys[0], "model": keys[1], "horizon": keys[2], "shots": keys[3],
            "mean_nrmse": group.nrmse.mean(), "median_nrmse_inflation": group.nrmse_difference.median(),
            "mean_nrmse_inflation": group.nrmse_difference.mean(), "paired_segment_ci95_low": lo,
            "paired_segment_ci95_high": hi, "median_relative_nrmse_inflation": group.relative_nrmse_inflation.median(),
            "mean_relative_nrmse_inflation": group.relative_nrmse_inflation.mean(),
            "p90_relative_nrmse_inflation": group.relative_nrmse_inflation.quantile(0.90),
        })
    summary = pd.DataFrame(summary_rows)
    persistence = persistence_strided(cfg, splits, scaled)
    useful_rows = []
    for (shots, set_name, model), group in raw.groupby(["shots", "set", "model"]):
        per_seg = group.groupby(["horizon", "segment_id"], as_index=False).nrmse.mean()
        evidence = []
        for horizon in cfg["horizons"]:
            model_h = per_seg[per_seg.horizon == horizon].set_index("segment_id").nrmse
            pers_h = persistence[(persistence["set"] == set_name) & (persistence.horizon == horizon)].set_index("segment_id").nrmse
            improvement = (pers_h - model_h).dropna().to_numpy()
            lo, _ = bootstrap_ci(improvement, f"useful:{shots}:{set_name}:{model}:{horizon}", cfg["bootstrap_resamples"])
            evidence.append((horizon, model_h.mean(), lo, model_h.mean() < 1 and lo > 0))
        valid = [row for row in evidence if row[-1]]
        useful_rows.append({"shots": shots, "set": set_name, "model": model,
                            "useful_horizon": max([row[0] for row in valid], default=np.nan),
                            "criterion": "mean NRMSE < 1 and paired-bootstrap lower CI(persistence-model) > 0"})
    useful = pd.DataFrame(useful_rows)
    exact_useful = useful[useful.shots == 0][["set", "model", "useful_horizon"]].rename(
        columns={"useful_horizon": "exact_useful_horizon"})
    summary = summary.merge(useful[useful.shots > 0], on=["shots", "set", "model"], how="left")
    summary = summary.merge(exact_useful, on=["set", "model"], how="left", validate="many_to_one")
    summary["useful_horizon_change_from_exact"] = summary.useful_horizon - summary.exact_useful_horizon
    summary.to_csv(RESULTS / "shot_sensitivity_summary.csv", index=False)

    contrast_rows = []
    for shots in [0] + cfg["shots"]:
        shot_data = raw[raw.shots == shots]
        for set_name in cfg["sets"]:
            for comparator in ("QRC_K0", "AB_noaux"):
                cols = ["channel_seed", "noise_replicate", "segment_id"]
                kernel = shot_data[(shot_data["set"] == set_name) & (shot_data.model == "single_kernel") & shot_data.horizon.isin([2, 64])]
                comp = shot_data[(shot_data["set"] == set_name) & (shot_data.model == comparator) & shot_data.horizon.isin([2, 64])]
                kp = kernel.pivot_table(index=cols, columns="horizon", values="nrmse")
                cp = comp.pivot_table(index=cols, columns="horizon", values="nrmse")
                common = kp.index.intersection(cp.index)
                interaction = (cp.loc[common, 64] - cp.loc[common, 2]) - (kp.loc[common, 64] - kp.loc[common, 2])
                segment_values = interaction.groupby(level="segment_id").mean().to_numpy()
                lo, hi = bootstrap_ci(segment_values, f"contrast:{shots}:{set_name}:{comparator}", cfg["bootstrap_resamples"])
                contrast_rows.append({"shots": shots, "set": set_name, "comparator": comparator,
                                      "interaction_comp_minus_kernel": interaction.mean(), "ci95_low": lo,
                                      "ci95_high": hi})
    contrasts = pd.DataFrame(contrast_rows)
    exact_contrast = contrasts[contrasts.shots == 0][["set", "comparator", "interaction_comp_minus_kernel"]].rename(
        columns={"interaction_comp_minus_kernel": "exact_interaction"})
    contrasts = contrasts.merge(exact_contrast, on=["set", "comparator"])
    contrasts["interaction_change_from_exact"] = contrasts.interaction_comp_minus_kernel - contrasts.exact_interaction
    contrasts["sign_preserved"] = np.sign(contrasts.interaction_comp_minus_kernel) == np.sign(contrasts.exact_interaction)
    contrasts.to_csv(RESULTS / "shot_sensitivity_contrasts.csv", index=False)

    thresholds = cfg["robustness_thresholds"]
    level_rows = []
    for shots in cfg["shots"]:
        slab = finite[finite.shots == shots]
        principal = contrasts[(contrasts.shots == shots) & contrasts["set"].isin(["F", "Z"])]
        level_rows.append({"shots": shots, "median": slab.relative_nrmse_inflation.median(),
                           "p90": slab.relative_nrmse_inflation.quantile(.9),
                           "sign_fraction": principal.sign_preserved.mean()})
    levels = pd.DataFrame(level_rows)
    levels["global_pass"] = ((levels["median"] <= thresholds["median_relative_nrmse_inflation_max"])
                              & (levels.p90 <= thresholds["p90_relative_nrmse_inflation_max"])
                              & (levels.sign_fraction >= thresholds["principal_contrast_sign_fraction_required"]))
    passing = levels[levels.global_pass]
    strata_rows = []
    for (shots, set_name, horizon), slab in finite.groupby(["shots", "set", "horizon"]):
        sign_ok = True
        if set_name in ("F", "Z"):
            sign_ok = bool(contrasts[(contrasts.shots == shots) & (contrasts["set"] == set_name)].sign_preserved.all())
        median = slab.relative_nrmse_inflation.median()
        p90 = slab.relative_nrmse_inflation.quantile(.9)
        strata_rows.append({"shots": shots, "set": set_name, "horizon": horizon, "median": median,
                            "p90": p90, "principal_signs_preserved": sign_ok,
                            "stratum_pass": median <= .05 and p90 <= .10 and sign_ok})
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(RESULTS / "shot_sensitivity_strata_classification.csv", index=False)
    scientific = classify_shot_sensitivity(levels, bool(strata.stratum_pass.any()))
    levels.to_csv(RESULTS / "shot_sensitivity_classification.csv", index=False)
    make_figure(summary, contrasts)
    technical = "COMPLETE"
    metadata = {
        "technical_verdict": technical, "scientific_classification": scientific,
        "git_commit": commit, "configuration": cfg, "timestamp": datetime.now(ZoneInfo(cfg["timezone"])).isoformat(),
        "working_tree_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__},
        "baseline_reproduction": json.loads((RESULTS / "shot_baseline_reproduction_status.json").read_text()),
        "minimum_robust_shots": int(passing.shots.min()) if len(passing) else None,
    }
    metadata["passing_strata"] = int(strata.stratum_pass.sum())
    metadata["total_strata"] = int(len(strata))
    report = build_report(cfg, levels, contrasts, useful, technical, scientific, metadata)
    (RESULTS / "gate3_report.md").write_text(report)
    artifacts = [RAW_PATH, RESULTS / "shot_sensitivity_exact_full.csv",
                 RESULTS / "shot_baseline_reproduction.csv",
                 RESULTS / "shot_baseline_reproduction_status.json",
                 RESULTS / "shot_sensitivity_summary.csv", RESULTS / "shot_sensitivity_contrasts.csv",
                 RESULTS / "shot_sensitivity_classification.csv", RESULTS / "shot_sensitivity_strata_classification.csv",
                 RESULTS / "gate3_report.md",
                 ROOT / "figures/eeg/fig_shot_sensitivity.pdf", ROOT / "figures/eeg/fig_shot_sensitivity.png",
                 ROOT / "results/resources/qrc_resource_table.csv", ROOT / "paper/tab_physical_resources.tex",
                 ROOT / "docs/gate2_postgate_addendum.md", ROOT / "docs/gate3_protocol.md",
                 ROOT / "docs/physical_resources.md", ROOT / "config/rotaA_gate3_frozen.json"]
    metadata["artifact_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in artifacts}
    (RESULTS / "gate3_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return technical, scientific


def make_figure(summary, contrasts):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for (set_name, model), group in summary.groupby(["set", "model"]):
        curve = group.groupby("shots").median_relative_nrmse_inflation.median().sort_index()
        axes[0].plot(curve.index, 100 * curve, marker="o", label=f"{set_name}/{model}")
    axes[0].axhline(5, ls="--", color="0.3"); axes[0].set_xscale("log")
    axes[0].set(xlabel="Shots per Pauli observable", ylabel="Median relative NRMSE inflation (%)",
                title="Finite-shot readout sensitivity")
    finite = contrasts[contrasts.shots > 0]
    for (set_name, comparator), group in finite.groupby(["set", "comparator"]):
        axes[1].plot(group.shots, group.interaction_change_from_exact, marker="o", label=f"{set_name}/{comparator}")
    axes[1].axhline(0, color="0.3", lw=.8); axes[1].set_xscale("log")
    axes[1].set(xlabel="Shots", ylabel="Interaction change from exact", title="Kernel contrast stability")
    axes[0].legend(fontsize=6, ncol=2); axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/eeg/fig_shot_sensitivity.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_report(cfg, levels, contrasts, useful, technical, scientific, metadata):
    lines = ["# Rota A Gate 3 — resources and finite-shot sensitivity", "",
             f"**Technical verdict: {technical}.**", f"**Scientific classification: {scientific}.**", "",
             "The exact official-r=0.7 baseline passed every available frozen-row comparison before shots.",
             "Extended single-kernel r=0.7 horizons had no frozen reference and are not presented as a reproduction.",
             "", "## Shot-level classification", "", "| Shots | Median relative inflation | P90 | Principal sign fraction | Pass |",
             "|---:|---:|---:|---:|:---:|"]
    for row in levels.itertuples():
        lines.append(f"| {row.shots} | {row.median:.4%} | {row.p90:.4%} | {row.sign_fraction:.3f} | {bool(row.global_pass)} |")
    lines += ["", f"No finite shot level passes globally, but {metadata['passing_strata']} of "
              f"{metadata['total_strata']} set×horizon strata pass the frozen 5%/10% criteria. "
              "The classification is therefore MIXED rather than globally robust or uniformly sensitive."]
    contrast_table = ["| Shots | Set | Comparator | Interaction | Change | Sign preserved |",
                      "|---:|---|---|---:|---:|:---:|"]
    for row in contrasts.itertuples():
        contrast_table.append(f"| {row.shots} | {row.set} | {row.comparator} | "
                              f"{row.interaction_comp_minus_kernel:.6f} | {row.interaction_change_from_exact:.6f} | "
                              f"{bool(row.sign_preserved)} |")
    lines += ["", "## Contrasts", "", "Kernel-vs-K0 and kernel-vs-AB interactions use h=2 and h=64. S is reported even when null.", "",
              *contrast_table, "", "## Required limitations", "",
              "This experiment adds binomial estimation noise to Pauli observables in train, validation and test.",
              "It does not model gate decoherence, state-preparation error, drift, complete measurement backaction,",
              "or physical storage of past states. Independent observable ensembles do not emulate every hardware",
              "correlation. Finite shots are not a real hardware execution, and robustness is not quantum advantage.",
              "", "Gate 3 stops here; no Stage 4 or manuscript work was performed."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("baseline", "shots", "summarize", "all"))
    args = parser.parse_args()
    cfg, hp, commit, eeg_cfg, splits, scaled = preflight()
    print(f"Gate 3 commit={commit}; HP hash={cfg['official_hp_sha256']}; cache={CACHE}", flush=True)
    if args.phase in ("baseline", "all"):
        reproduce_baseline(cfg, hp, eeg_cfg, splits, scaled)
    if args.phase in ("shots", "all"):
        run_shots(cfg, hp, splits, scaled)
    if args.phase in ("summarize", "all"):
        technical, scientific = summarize(cfg, splits, scaled, commit)
        print(f"technical={technical}; scientific={scientific}")


if __name__ == "__main__":
    main()
