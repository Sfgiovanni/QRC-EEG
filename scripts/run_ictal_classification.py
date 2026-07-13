#!/usr/bin/env python3
"""Ictal (S) vs non-ictal (Z, F) classification, pre-registered secondary task
(docs/eeg_preregistration.md, "Secondary task"). Reuses the already-generated
reservoir features (qrc_eeg.pipeline.construction_features) -- same
constructions, same channel/kernel code, same frozen segment-level splits as
forecasting. Only new pieces: mean-pooling to a per-segment feature vector
and the logistic readout (src/qrc_eeg/classification.py), because the
forecasting pipeline has no classification readout.

Equalized-budget comparison: single_kernel, dual_kernel, AB_noaux, ESN_66
(66 readout features on every arm -- ESN_66's selected HP comes from
results/eeg/hp_selected_esn66.json, Item 2, not the dimension-unmatched
ESN-200).

No leakage: segments are split once (data/eeg/splits/*, frozen, same split
used by forecasting); a segment's label (ictal iff its set is S) is fixed by
which split fold it already sits in, so pooling Z/F/S together introduces no
new leakage surface. Mandatory sanity check: fit on label-shuffled data,
evaluate held out -- must collapse to chance AUROC or this script aborts.

Writes:
  results/eeg/tab_ictal_classification.csv        (+ .tex)  -- per-construction AUROC/AUPRC endpoints
  results/eeg/tab_ictal_classification_contrasts.csv (+ .tex) -- kernel vs AB_noaux / kernel vs ESN_66
  results/eeg/raw/eeg_ictal_by_seed.csv            -- per-seed, pre-ensemble AUROC/AUPRC (diagnostic)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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
from qrc_eeg.eeg_data import load_set  # noqa: E402
from qrc_eeg.pipeline import construction_features  # noqa: E402
from qrc_eeg.splits import load_splits  # noqa: E402
from qrc_eeg.statistics import holm  # noqa: E402
from qrc_eeg.tasks import zscore  # noqa: E402

CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
SPLITS_DIR = ROOT / "data" / "eeg" / "splits"
RESULTS_DIR = ROOT / "results" / "eeg"
RAW_DIR = RESULTS_DIR / "raw"
PAPER_DIR = ROOT / "paper"

CONSTRUCTIONS = ["single_kernel", "dual_kernel", "AB_noaux", "ESN_66"]
CONTRASTS = [("single_kernel", "AB_noaux"), ("single_kernel", "ESN_66")]
SHUFFLE_N_PERMS = 20
SHUFFLE_TOLERANCE = (0.35, 0.65)


def hp_for(construction: str, selected: dict, selected_esn66: dict) -> dict:
    if construction == "ESN_66":
        return selected_esn66["ESN_66"]["hp"]
    return selected[construction]["hp"]


def construction_name_for_features(construction: str) -> str:
    return "ESN" if construction == "ESN_66" else construction


def revtex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = " & ".join(str(c) for c in df.columns)
    lines = [
        r"\begin{table}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{" + "l" * len(df.columns) + "}",
        cols + r" \\",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    sets = cfg["data"]["sets"]
    washout = cfg["readout"]["washout"]
    alpha_grid = cfg["readout"]["alpha_grid"]
    n_boot = cfg["statistics"]["bootstrap_resamples"]
    boot_seed = cfg["split"]["seed"]
    selected = json.loads((RESULTS_DIR / "hp_selected.json").read_text())
    selected_esn66 = json.loads((RESULTS_DIR / "hp_selected_esn66.json").read_text())

    raw = {name: load_set(ROOT / "data" / "eeg" / "sets" / name) for name in sets}
    zscored = {name: {sid: zscore(np.array(v))[0] for sid, v in segs.items()} for name, segs in raw.items()}
    splits = load_splits(SPLITS_DIR, sets)

    # Build pooled train/val/test segment id lists (order fixed once, reused for every construction/seed).
    train_ids, val_ids, test_ids = [], [], []
    train_set, val_set, test_set = [], [], []
    for set_name in sets:
        for sid in splits[set_name]["train"]:
            train_ids.append(sid)
            train_set.append(set_name)
        for sid in splits[set_name]["val"]:
            val_ids.append(sid)
            val_set.append(set_name)
        for sid in splits[set_name]["test"]:
            test_ids.append(sid)
            test_set.append(set_name)

    # No segment id appears in more than one of train/val/test (checked at split-build time,
    # reasserted here since classification pools across sets differently than forecasting does).
    assert set(train_ids).isdisjoint(test_ids) and set(val_ids).isdisjoint(test_ids) and set(train_ids).isdisjoint(val_ids)

    train_arr = {s: np.stack([zscored[s][i] for i in splits[s]["train"]]) for s in sets}
    val_arr = {s: np.stack([zscored[s][i] for i in splits[s]["val"]]) for s in sets}
    test_arr = {s: np.stack([zscored[s][i] for i in splits[s]["test"]]) for s in sets}

    y_train = np.array([1 if s == "S" else 0 for s in train_set])
    y_val = np.array([1 if s == "S" else 0 for s in val_set])
    y_trainval = np.concatenate([y_train, y_val])
    y_test = np.array([1 if s == "S" else 0 for s in test_set])

    per_seed_rows = []
    test_probs = {c: np.zeros(len(test_ids)) for c in CONSTRUCTIONS}
    shuffle_pooled_features = {}  # construction -> (trainval_features, test_features) at first seed, for the mandatory sanity check

    for construction in CONSTRUCTIONS:
        hp = hp_for(construction, selected, selected_esn66)
        feat_name = construction_name_for_features(construction)
        for seed in cfg["channel"]["confirmatory_seeds"]:
            t0 = time.perf_counter()
            feats_train = np.concatenate([construction_features(feat_name, hp, seed=seed, segments=train_arr[s]) for s in sets], axis=0)
            feats_val = np.concatenate([construction_features(feat_name, hp, seed=seed, segments=val_arr[s]) for s in sets], axis=0)
            feats_test = np.concatenate([construction_features(feat_name, hp, seed=seed, segments=test_arr[s]) for s in sets], axis=0)

            x_train = mean_pool_features(feats_train, washout)
            x_val = mean_pool_features(feats_val, washout)
            x_test = mean_pool_features(feats_test, washout)
            x_trainval = np.concatenate([x_train, x_val], axis=0)

            best_alpha, best_ll = alpha_grid[0], np.inf
            for alpha in alpha_grid:
                w = fit_logistic_readout(x_train, y_train, alpha=alpha)
                p_val = np.clip(predict_logistic_proba(x_val, w), 1e-9, 1 - 1e-9)
                ll = float(-np.mean(y_val * np.log(p_val) + (1 - y_val) * np.log(1 - p_val)))
                if ll < best_ll:
                    best_alpha, best_ll = alpha, ll

            w_final = fit_logistic_readout(x_trainval, y_trainval, alpha=best_alpha)
            p_test = predict_logistic_proba(x_test, w_final)
            test_probs[construction] += p_test / len(cfg["channel"]["confirmatory_seeds"])

            elapsed = time.perf_counter() - t0
            seed_auc = auroc(y_test, p_test)
            seed_ap = auprc(y_test, p_test)
            per_seed_rows.append(
                {"construction": construction, "seed": seed, "alpha": best_alpha, "auroc": seed_auc, "auprc": seed_ap, "seconds": elapsed}
            )
            print(f"{construction} seed={seed}: {elapsed:.1f}s alpha={best_alpha:g} auroc={seed_auc:.4f} auprc={seed_ap:.4f}", flush=True)

            if seed == cfg["channel"]["confirmatory_seeds"][0]:
                shuffle_pooled_features[construction] = (x_train, x_val, x_test)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(RAW_DIR / "eeg_ictal_by_seed.csv", index=False)
    print("wrote", RAW_DIR / "eeg_ictal_by_seed.csv")

    # --- MANDATORY leakage sanity check: shuffled labels must collapse AUROC to chance. ---
    rng = np.random.default_rng(boot_seed)
    shuffle_report = []
    for construction in CONSTRUCTIONS:
        x_train, x_val, x_test = shuffle_pooled_features[construction]
        x_trainfit = np.concatenate([x_train, x_val], axis=0)
        aucs = []
        for _ in range(SHUFFLE_N_PERMS):
            y_shuffled_all = rng.permutation(np.concatenate([y_trainval, y_test]))
            y_fit = y_shuffled_all[: len(x_trainfit)]
            y_eval = y_shuffled_all[len(x_trainfit) :]
            if len(np.unique(y_fit)) < 2:
                continue
            w = fit_logistic_readout(x_trainfit, y_fit, alpha=1.0)
            p = predict_logistic_proba(x_test, w)
            aucs.append(auroc(y_eval, p))
        mean_shuffled_auc = float(np.mean(aucs))
        shuffle_report.append({"construction": construction, "mean_shuffled_auroc": mean_shuffled_auc, "n_perms": len(aucs)})
        print(f"[sanity] {construction}: mean shuffled-label held-out AUROC = {mean_shuffled_auc:.4f}", flush=True)
        if not (SHUFFLE_TOLERANCE[0] < mean_shuffled_auc < SHUFFLE_TOLERANCE[1]):
            raise SystemExit(
                f"ABORT: leakage sanity check failed for {construction}: shuffled-label held-out AUROC "
                f"{mean_shuffled_auc:.4f} is outside chance band {SHUFFLE_TOLERANCE} -- classification split "
                "or label wiring is leaking information. Fix before trusting any number in this run."
            )
    pd.DataFrame(shuffle_report).to_csv(RESULTS_DIR / "ictal_shuffle_check.csv", index=False)
    print("PASS: leakage sanity check -- all constructions collapse to chance under label shuffling")

    # --- Endpoint table: AUROC/AUPRC + bootstrap CI per construction. ---
    endpoint_rows = []
    for construction in CONSTRUCTIONS:
        result = bootstrap_auc_ci(y_test, test_probs[construction], n_boot=n_boot, seed=boot_seed)
        endpoint_rows.append(
            {
                "construction": construction,
                "n_test_segments": len(y_test),
                "n_pos_ictal": int(np.sum(y_test)),
                "n_neg_nonictal": int(len(y_test) - np.sum(y_test)),
                **result,
            }
        )
    endpoints = pd.DataFrame(endpoint_rows)
    endpoints.to_csv(RESULTS_DIR / "tab_ictal_classification.csv", index=False)
    print("wrote", RESULTS_DIR / "tab_ictal_classification.csv")
    print(endpoints)

    # --- Contrasts: kernel vs AB_noaux, kernel vs ESN_66 (paired bootstrap, shared resample indices). ---
    contrast_rows = []
    for state_name, comparator_name in CONTRASTS:
        contrast_rows.append(
            paired_bootstrap_delta_auc(
                y_test, test_probs[state_name], test_probs[comparator_name], state_name, comparator_name, n_boot=n_boot, seed=boot_seed
            )
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_holm"] = holm(contrasts["p_boot_auroc"].to_numpy())  # family eeg_ictal_classification (2 tests)
    contrasts.to_csv(RESULTS_DIR / "tab_ictal_classification_contrasts.csv", index=False)
    print("wrote", RESULTS_DIR / "tab_ictal_classification_contrasts.csv")
    print(contrasts)

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    endpoint_cols = ["construction", "n_test_segments", "n_pos_ictal", "n_neg_nonictal", "auroc", "auroc_ci_lo", "auroc_ci_hi", "auprc", "auprc_ci_lo", "auprc_ci_hi"]
    (PAPER_DIR / "tab_ictal_classification.tex").write_text(
        revtex_table(endpoints[endpoint_cols].round(4), "Ictal (S) vs non-ictal (Z, F) classification, equalized 66-feature logistic readout.", "tab:ictal_classification")
    )
    contrast_cols = ["comparison", "n_segments", "delta_auroc", "delta_auroc_ci_lo", "delta_auroc_ci_hi", "p_holm", "delta_auprc", "delta_auprc_ci_lo", "delta_auprc_ci_hi"]
    (PAPER_DIR / "tab_ictal_classification_contrasts.tex").write_text(
        revtex_table(contrasts[contrast_cols].round(4), "Ictal classification paired contrasts, family eeg\\_ictal\\_classification, Holm-corrected.", "tab:ictal_classification_contrasts")
    )
    print("wrote paper/tab_ictal_classification.tex, paper/tab_ictal_classification_contrasts.tex")


if __name__ == "__main__":
    main()
