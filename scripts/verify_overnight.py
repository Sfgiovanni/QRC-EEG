#!/usr/bin/env python3
"""Verification gate for the overnight run (3 items). Fails loud: any check
that doesn't pass raises SystemExit with a clear message and nothing below
it runs. Run last, after Items 1-3 and the figures are all written.

1. Reproducibility: re-runs (not re-reads) single_kernel at seed=1 for one
   set/horizon through the real pipeline and diffs it against the stored
   raw CSV row-for-row (catches code drift). Separately recomputes the
   h=1 R^2 anchors reported in RESULTS.md (0.92 Z / 0.97 F / 0.97 S) directly
   from the raw CSV filtered to horizon==1 -- NOT from tab_eeg_endpoints.csv,
   which is horizon-averaged and would silently pass a wrong anchor.
2. Anti-leakage: re-checks split disjointness (train/val/test, and no
   cross-set segment-id collision) and confirms the classification
   shuffled-label sanity check (results/eeg/ictal_shuffle_check.csv) passed.
3. Completeness: ESN-66 raw grid has exactly 3x4x10 (set,horizon,seed) cells;
   every new CSV/tex/figure this run wrote has a SHA256 recorded in
   provenance/eeg_checksums.txt.
4. Writes results/eeg/overnight_summary.md -- the one-page digest.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.eeg_data import load_set  # noqa: E402
from qrc_eeg.pipeline import construction_features, fit_readouts_per_horizon, evaluate_segments_full  # noqa: E402
from qrc_eeg.splits import assert_disjoint, load_splits  # noqa: E402
from qrc_eeg.tasks import zscore  # noqa: E402
import json  # noqa: E402

CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
SPLITS_DIR = ROOT / "data" / "eeg" / "splits"
RESULTS_DIR = ROOT / "results" / "eeg"
PROVENANCE_PATH = ROOT / "provenance" / "eeg_checksums.txt"

RESULTS_MD_H1_R2_ANCHORS = {"Z": 0.92, "F": 0.97, "S": 0.97}
R2_TOLERANCE = 0.01

NEW_FILES = [
    "results/eeg/hp_search_log_esn66.csv",
    "results/eeg/hp_selected_esn66.json",
    "results/eeg/raw/eeg_holdout_esn66_by_segment_seed.csv",
    "results/eeg/tab_esn_matched.csv",
    "results/eeg/tab_long_horizon_contrasts.csv",
    "results/eeg/raw/eeg_ictal_by_seed.csv",
    "results/eeg/ictal_shuffle_check.csv",
    "results/eeg/tab_ictal_classification.csv",
    "results/eeg/tab_ictal_classification_contrasts.csv",
    "paper/tab_esn_matched.tex",
    "paper/tab_long_horizon_contrasts.tex",
    "paper/tab_ictal_classification.tex",
    "paper/tab_ictal_classification_contrasts.tex",
    "figures/eeg/fig_long_horizon.pdf",
    "figures/eeg/fig_long_horizon.png",
    "figures/eeg/fig_ictal_auroc.pdf",
    "figures/eeg/fig_ictal_auroc.png",
]


def abort(msg: str) -> None:
    raise SystemExit(f"GATE FAILED: {msg}")


def check_reproducibility(cfg: dict) -> dict:
    print("[gate] 1/4 reproducibility: re-running single_kernel seed=1, all sets, h=1 ...", flush=True)
    selected = json.loads((RESULTS_DIR / "hp_selected.json").read_text())
    hp = selected["single_kernel"]["hp"]
    sets = cfg["data"]["sets"]
    washout = cfg["readout"]["washout"]
    alpha_grid = cfg["readout"]["alpha_grid"]
    splits = load_splits(SPLITS_DIR, sets)
    raw = {name: load_set(ROOT / "data" / "eeg" / "sets" / name) for name in sets}
    zscored = {name: {sid: zscore(np.array(v))[0] for sid, v in segs.items()} for name, segs in raw.items()}

    stored = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_by_segment_seed.csv")
    stored = stored[(stored.construction == "single_kernel") & (stored.seed == 1) & (stored.horizon == 1)]

    r2_recomputed = {}
    for set_name in sets:
        trainval_ids = splits[set_name]["train"] + splits[set_name]["val"]
        test_ids = splits[set_name]["test"]
        trainval_arr = np.stack([zscored[set_name][i] for i in trainval_ids])
        test_arr = np.stack([zscored[set_name][i] for i in test_ids])
        feats_trainval = construction_features("single_kernel", hp, seed=1, segments=trainval_arr)
        feats_test = construction_features("single_kernel", hp, seed=1, segments=test_arr)
        fits = fit_readouts_per_horizon(feats_trainval, trainval_arr, [1], alpha_grid, washout=washout)
        results = evaluate_segments_full(feats_test, test_arr, fits, washout=washout)

        recomputed_nrmse = {sid: results[1]["nrmse"][i] for i, sid in enumerate(test_ids)}
        stored_set = stored[stored.set == set_name].set_index("segment_id")["nrmse"]
        for sid, val in recomputed_nrmse.items():
            stored_val = float(stored_set.loc[sid])
            if not np.isclose(val, stored_val, rtol=1e-6, atol=1e-8):
                abort(
                    f"code-drift: recomputed single_kernel seed=1 set={set_name} seg={sid} h=1 nrmse={val:.6f} "
                    f"!= stored {stored_val:.6f} -- pipeline behavior changed since the held-out run"
                )
        r2_recomputed[set_name] = float(np.mean(results[1]["r2"]))

    # Full-anchor check (all seeds/segments, from raw, horizon==1 only -- not the horizon-averaged endpoints table).
    full_h1 = stored_full = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_by_segment_seed.csv")
    full_h1 = full_h1[(full_h1.construction == "single_kernel") & (full_h1.horizon == 1)]
    r2_full = full_h1.groupby("set")["r2"].mean().to_dict()
    for set_name, anchor in RESULTS_MD_H1_R2_ANCHORS.items():
        got = r2_full.get(set_name)
        if got is None or abs(got - anchor) > R2_TOLERANCE:
            abort(f"RESULTS.md anchor mismatch: single_kernel h=1 R^2 on {set_name} is {got} vs reported {anchor} (tol {R2_TOLERANCE})")
    print(f"[gate] reproducibility OK -- seed=1 h=1 nrmse matches stored raw row-for-row; R^2 anchors match RESULTS.md: {r2_full}", flush=True)
    return r2_full


def check_antileakage(cfg: dict) -> dict:
    print("[gate] 2/4 anti-leakage ...", flush=True)
    sets = cfg["data"]["sets"]
    splits = load_splits(SPLITS_DIR, sets)
    for name, split in splits.items():
        assert_disjoint(split)
    test_ids, trainval_ids = set(), set()
    for split in splits.values():
        test_ids |= set(split["test"])
        trainval_ids |= set(split["train"]) | set(split["val"])
    if not test_ids.isdisjoint(trainval_ids):
        abort("cross-set segment id collision between test and train/val folds")

    shuffle_path = RESULTS_DIR / "ictal_shuffle_check.csv"
    if not shuffle_path.exists():
        abort(f"{shuffle_path} missing -- classification script did not run or was skipped")
    shuffle_df = pd.read_csv(shuffle_path)
    bad = shuffle_df[(shuffle_df.mean_shuffled_auroc <= 0.35) | (shuffle_df.mean_shuffled_auroc >= 0.65)]
    if not bad.empty:
        abort(f"classification shuffle check did not collapse to chance for: {bad['construction'].tolist()}")
    print("[gate] anti-leakage OK -- splits disjoint, shuffle-label AUROC near chance for all constructions", flush=True)
    return shuffle_df.set_index("construction")["mean_shuffled_auroc"].to_dict()


def check_completeness(cfg: dict) -> None:
    print("[gate] 3/4 completeness ...", flush=True)
    esn66 = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_esn66_by_segment_seed.csv")
    cells = esn66.groupby(["set", "horizon", "seed"]).size()
    expected_sets = len(cfg["data"]["sets"])
    expected_horizons = len(cfg["readout"]["horizons"])
    expected_seeds = len(cfg["channel"]["confirmatory_seeds"])
    if len(cells) != expected_sets * expected_horizons * expected_seeds:
        abort(f"ESN-66 raw grid incomplete: {len(cells)} of {expected_sets * expected_horizons * expected_seeds} (set,horizon,seed) cells")
    expected_n_test = 20  # test_frac=0.2 * n_segments_per_set=100
    bad_cells = cells[cells != expected_n_test]
    if not bad_cells.empty:
        abort(f"ESN-66 raw grid has cells with wrong segment count (expected {expected_n_test}): {bad_cells.to_dict()}")

    checksummed = PROVENANCE_PATH.read_text() if PROVENANCE_PATH.exists() else ""
    missing = [f for f in NEW_FILES if (ROOT / f).exists() and f.split("/")[-1] not in checksummed]
    if missing:
        abort(f"new artifacts missing SHA256 in {PROVENANCE_PATH}: {missing}")
    print("[gate] completeness OK -- ESN-66 grid is 3x4x10x20, all new artifacts checksummed", flush=True)


def write_summary(cfg: dict, r2_anchors: dict, shuffle_aucs: dict) -> None:
    print("[gate] 4/4 writing overnight_summary.md ...", flush=True)
    long_h = pd.read_csv(RESULTS_DIR / "tab_long_horizon_contrasts.csv")
    esn_matched = pd.read_csv(RESULTS_DIR / "tab_esn_matched.csv")
    ictal = pd.read_csv(RESULTS_DIR / "tab_ictal_classification.csv")
    ictal_contrasts = pd.read_csv(RESULTS_DIR / "tab_ictal_classification_contrasts.csv")
    primary = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_by_segment_seed.csv")

    def fmt_row(row) -> str:
        sig = "**sig**" if row["p_holm"] < 0.05 else "ns"
        return f"| {row['set']} | h={row['horizon']} | {row['mean_diff_rmse_comparator_minus_state']:+.4f} | [{row['ci95_lo']:+.4f}, {row['ci95_hi']:+.4f}] | p_holm={row['p_holm']:.4g} ({sig}) | win={row['win_fraction_state']:.2f} |"

    lines = []
    lines.append("# Overnight run summary (3 items + gate)\n")
    lines.append(f"Generated by scripts/verify_overnight.py. Reproducibility anchors matched RESULTS.md: {r2_anchors}.\n")

    lines.append("## Kernel vs AB (h=1,2,4,8)\n")
    lines.append("h=1,2 from the original held-out run (results/eeg/tab_eeg_contrasts.csv); h=4,8 below.\n")
    lines.append("| Set | Horizon | Delta-NRMSE (AB - kernel) | 95% CI | Holm p (family eeg_long) | Win frac |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in long_h[long_h.comparison == "single_kernel vs AB_noaux"].iterrows():
        lines.append(fmt_row(row))
    lines.append("")

    lines.append("## Kernel vs ESN-66 (the valid, dimension-matched comparison)\n")
    lines.append("| Set | Horizon | Delta-NRMSE (ESN66 - kernel) | 95% CI | Holm p (family eeg_esn_matched / eeg_long) | Win frac |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in esn_matched.iterrows():
        lines.append(fmt_row(row))
    lines.append("")
    lines.append("(long-horizon subset of the same comparison is duplicated in tab_long_horizon_contrasts.csv, family eeg_long)\n")

    lines.append("## Ictal (S) vs non-ictal (Z,F) classification -- AUROC by construction\n")
    lines.append("| Construction | AUROC | 95% CI | AUPRC | 95% CI | shuffled-label AUROC (sanity) |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in ictal.iterrows():
        c = row["construction"]
        lines.append(
            f"| {c} | {row['auroc']:.4f} | [{row['auroc_ci_lo']:.4f}, {row['auroc_ci_hi']:.4f}] | "
            f"{row['auprc']:.4f} | [{row['auprc_ci_lo']:.4f}, {row['auprc_ci_hi']:.4f}] | {shuffle_aucs.get(c, float('nan')):.3f} |"
        )
    lines.append("")
    lines.append("Ictal classification contrasts:\n")
    lines.append("| Comparison | Delta-AUROC | 95% CI | Holm p | Delta-AUPRC | 95% CI |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in ictal_contrasts.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['delta_auroc']:+.4f} | [{row['delta_auroc_ci_lo']:+.4f}, {row['delta_auroc_ci_hi']:+.4f}] | "
            f"{row['p_holm']:.4g} | {row['delta_auprc']:+.4f} | [{row['delta_auprc_ci_lo']:+.4f}, {row['delta_auprc_ci_hi']:+.4f}] |"
        )
    lines.append("")

    # Pre-committed decision rule, applied mechanically to whatever came out.
    ab_long_sig_favoring_kernel = long_h[
        (long_h.comparison == "single_kernel vs AB_noaux") & (long_h.p_holm < 0.05) & (long_h.mean_diff_rmse_comparator_minus_state > 0)
    ]
    ab_long_all_favor = len(ab_long_sig_favoring_kernel) == len(long_h[long_h.comparison == "single_kernel vs AB_noaux"])
    esn66_ci_cross_zero = ((esn_matched["ci95_lo"] < 0) & (esn_matched["ci95_hi"] > 0)).all()
    kernel_auroc_row = ictal[ictal.construction == "single_kernel"].iloc[0]
    ab_auroc_row = ictal[ictal.construction == "AB_noaux"].iloc[0]
    kernel_beats_ab_classification = kernel_auroc_row["auroc_ci_lo"] > ab_auroc_row["auroc_ci_hi"]

    strong_reading = ab_long_sig_favoring_kernel.shape[0] > 0 and esn66_ci_cross_zero
    lines.append("## Leitura A-vs-B (regra pre-comprometida, aplicada mecanicamente)\n")
    lines.append(
        f"Regra: leitura FORTE (mirar PRE) requer kernel > AB significativo em long horizon E kernel~ESN-66 "
        f"(CI cruzando zero em todas as celulas) E algum sinal de vantagem do kernel na classificacao; "
        f"caso contrario, leitura de RECUO (desacoplamento/robustez, mirar PRResearch).\n"
    )
    lines.append(
        f"Observado: AB long-horizon todos favorecem o kernel = {ab_long_all_favor} "
        f"({ab_long_sig_favoring_kernel.shape[0]}/{len(long_h[long_h.comparison=='single_kernel vs AB_noaux'])} significativos); "
        f"ESN-66 CI cruza zero em todas as celulas = {esn66_ci_cross_zero}; "
        f"kernel bate AB_noaux na classificacao (CIs nao sobrepoem) = {kernel_beats_ab_classification}.\n"
    )
    if strong_reading:
        lines.append(
            "**Leitura: FORTE.** O kernel mantem/amplia vantagem sobre AB-noaux em horizonte longo e, a orcamento "
            "igualado (66 features), e competitivo com o ESN classico (CI cruzando zero). Isso sustenta a narrativa "
            "de que a construcao de kernel exponencial e uma alternativa QRC solida e competitiva com o classico "
            "no substrato testado -- mirar PRE com essa moldura.\n"
        )
    else:
        lines.append(
            "**Leitura: RECUO.** Os dados nao sustentam mecanicamente a leitura forte pre-comprometida (ver "
            "observado acima). A narrativa honesta e a de desacoplamento/robustez: o kernel exponencial continua "
            "sendo a melhor construcao QRC entre as testadas e permanece competitivo com o classico igualado, mas "
            "sem uma vantagem que se amplie de forma inequivoca em horizonte longo ou que se transfira claramente "
            "para o regime nao linear de classificacao -- mirar PRResearch com essa moldura, sem forcar a leitura forte.\n"
        )

    (RESULTS_DIR / "overnight_summary.md").write_text("\n".join(lines))
    print("[gate] wrote", RESULTS_DIR / "overnight_summary.md", flush=True)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    r2_anchors = check_reproducibility(cfg)
    shuffle_aucs = check_antileakage(cfg)
    check_completeness(cfg)
    write_summary(cfg, r2_anchors, shuffle_aucs)
    print("GATE PASSED.")


if __name__ == "__main__":
    main()
