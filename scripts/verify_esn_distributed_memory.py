#!/usr/bin/env python3
"""Fail-high verifier for the follow-up classical distributed-memory ESN
control and crossed segment x seed inference (Section 7 of the task spec).

Recomputes tables from the raw CSVs independently and checks they match the
stored artifacts; checks protocol/config hashes are unchanged since freeze;
checks grid completeness/uniqueness; checks the crossed-inference family has
exactly 21 Holm-corrected tests.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.crossed_inference import crossed_bootstrap, interaction_matrix  # noqa: E402
from qrc_eeg.esn_distributed_memory import CONSTRUCTIONS  # noqa: E402
from qrc_eeg.statistics import holm  # noqa: E402

FOLLOWUP = ROOT / "results" / "eeg" / "followup"
CONTROL = FOLLOWUP / "classical_control"
CROSSED = FOLLOWUP / "crossed_inference"
CONFIG_PATH = ROOT / "config" / "esn_distributed_memory_frozen.yaml"
EEG_CONFIG_PATH = ROOT / "config" / "eeg_frozen.yaml"
PROTOCOL_PATHS = [
    ROOT / "docs/classical_distributed_memory_protocol.md",
    ROOT / "docs/crossed_inference_protocol.md",
    CONFIG_PATH,
]
HASHES_FILE = FOLLOWUP / "PROTOCOL_HASHES.sha256"

FAILURES: list[str] = []


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"FAIL: {message}")


def ok(message: str) -> None:
    print(f"OK: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol_hashes() -> None:
    recorded = {}
    for line in HASHES_FILE.read_text().splitlines():
        if not line.strip():
            continue
        digest, path = line.split(maxsplit=1)
        recorded[path.strip()] = digest.strip()
    for path in PROTOCOL_PATHS:
        rel = str(path.relative_to(ROOT))
        matches = [k for k in recorded if k.endswith(path.name)]
        if not matches:
            fail(f"protocol hash record missing for {rel}")
            continue
        current = sha256(path)
        if current != recorded[matches[0]]:
            fail(f"protocol file changed since freeze: {rel}")
        else:
            ok(f"protocol hash unchanged: {rel}")


def verify_grid_completeness() -> pd.DataFrame:
    raw_path = FOLLOWUP / "raw" / "esn_distributed_memory_holdout_by_segment_seed.csv"
    if not raw_path.exists():
        fail("missing raw holdout CSV")
        return pd.DataFrame()
    df = pd.read_csv(raw_path)
    ecfg = yaml.safe_load(EEG_CONFIG_PATH.read_text())
    sets = ecfg["data"]["sets"]
    horizons = ecfg["readout"]["horizons"]
    seeds = ecfg["channel"]["confirmatory_seeds"]
    modes = ["fixed_core", "retuned_core"]

    key_cols = ["construction", "analysis_mode", "set", "horizon", "seed", "segment_id"]
    dupes = df.duplicated(subset=key_cols).sum()
    if dupes:
        fail(f"{dupes} duplicated holdout cells")
    else:
        ok("no duplicated holdout cells")

    missing = 0
    for construction in CONSTRUCTIONS:
        for mode in modes:
            for set_name in sets:
                split = json.loads((ROOT / f"data/eeg/splits/{set_name}_split.json").read_text())
                test_ids = set(split["test"])
                for horizon in horizons:
                    for seed in seeds:
                        got = set(df[
                            (df.construction == construction) & (df.analysis_mode == mode)
                            & (df.set == set_name) & (df.horizon == horizon) & (df.seed == seed)
                        ]["segment_id"])
                        if got != test_ids:
                            missing += 1
    if missing:
        fail(f"{missing} incomplete (construction,mode,set,horizon,seed) cells")
    else:
        ok(f"all {len(CONSTRUCTIONS) * len(modes) * len(sets) * len(horizons) * len(seeds)} holdout cells complete")

    if not np.isfinite(df["nrmse"]).all():
        fail("non-finite nrmse values in holdout raw CSV")
    else:
        ok("no non-finite nrmse values")

    return df


def verify_aggregated_table_recomputable(raw: pd.DataFrame) -> None:
    table_path = CONTROL / "tab_classical_distributed_memory.csv"
    if not table_path.exists() or raw.empty:
        fail("missing tab_classical_distributed_memory.csv")
        return
    stored = pd.read_csv(table_path)
    per_segment = raw.groupby(["construction", "analysis_mode", "set", "horizon", "segment_id"], as_index=False)["nrmse"].mean()
    recomputed = per_segment.groupby(["construction", "analysis_mode", "set", "horizon"], as_index=False)["nrmse"].mean()
    merged = stored.merge(
        recomputed, on=["construction", "analysis_mode", "set", "horizon"], suffixes=("_stored", "_recomputed")
    )
    if len(merged) != len(stored):
        fail("aggregated table rows do not align with raw recomputation")
        return
    max_err = float(np.max(np.abs(merged["mean_nrmse"] - merged["nrmse"])))
    if max_err > 1e-9:
        fail(f"aggregated mean_nrmse does not match raw recomputation, max_err={max_err}")
    else:
        ok(f"aggregated table recomputes from raw CSV, max_err={max_err:.2e}")


def verify_crossed_bootstrap_reproducible() -> None:
    boot_path = CROSSED / "crossed_bootstrap.csv"
    if not boot_path.exists():
        fail("missing crossed_bootstrap.csv")
        return
    stored = pd.read_csv(boot_path)
    if len(stored) != 21:
        fail(f"crossed_inference family has {len(stored)} rows, expected 21")
    else:
        ok("crossed_inference family has exactly 21 tests")

    recomputed_p = holm(stored["p_bootstrap"].to_numpy())
    if not np.allclose(recomputed_p, stored["p_holm"].to_numpy(), atol=1e-12):
        fail("Holm-adjusted p-values in crossed_bootstrap.csv are not reproducible")
    else:
        ok("Holm correction reproducible")

    fcfg = yaml.safe_load(CONFIG_PATH.read_text())
    ci_cfg = fcfg["crossed_inference"]
    h_short, h_long, seed = ci_cfg["h_short"], ci_cfg["h_long"], ci_cfg["bootstrap_rng_seed"]

    primary = pd.read_csv(ROOT / "results/eeg/raw/eeg_holdout_by_segment_seed.csv")
    esn66 = pd.read_csv(ROOT / "results/eeg/raw/eeg_holdout_esn66_by_segment_seed.csv")
    qrc_df = pd.concat([primary, esn66], ignore_index=True)
    followup_raw_path = FOLLOWUP / "raw" / "esn_distributed_memory_holdout_by_segment_seed.csv"
    followup_df = pd.read_csv(followup_raw_path) if followup_raw_path.exists() else pd.DataFrame()

    rng = np.random.default_rng(seed)
    mismatches = 0
    for set_name in ci_cfg["sets"]:
        for contrast in ci_cfg["contrasts"]:
            kernel, comparator = contrast["kernel"], contrast["comparator"]
            for mode in contrast["modes"]:
                df = qrc_df if mode is None else followup_df[followup_df["analysis_mode"] == mode]
                if df.empty:
                    continue
                mat, _, _ = interaction_matrix(df, kernel, comparator, set_name, h_short, h_long)
                recomputed = crossed_bootstrap(mat, rng, n_replicates=ci_cfg["n_bootstrap_replicates"])
                mode_label = mode if mode is not None else "not_applicable"
                match = stored[
                    (stored["set"] == set_name) & (stored["kernel"] == kernel)
                    & (stored["comparator"] == comparator) & (stored["analysis_mode"] == mode_label)
                ]
                if match.empty:
                    fail(f"stored row missing for {kernel} vs {comparator} ({mode_label}), set {set_name}")
                    continue
                stored_row = match.iloc[0]
                if abs(stored_row["observed_mean"] - recomputed["observed_mean"]) > 1e-9:
                    mismatches += 1
                if abs(stored_row["bootstrap_mean"] - recomputed["bootstrap_mean"]) > 1e-9:
                    mismatches += 1
    if mismatches:
        fail(f"{mismatches} crossed-bootstrap values not exactly reproducible from the registered RNG seed")
    else:
        ok("crossed bootstrap exactly reproducible from the registered RNG seed, full iteration order")


def verify_figures_present() -> None:
    for name in ("fig_classical_distributed_memory", "fig_crossed_inference"):
        for suffix in ("pdf", "png"):
            path = ROOT / "figures" / "eeg" / f"{name}.{suffix}"
            if not path.exists():
                fail(f"missing figure: {path}")
            else:
                ok(f"figure present: {path.name}")


def main() -> None:
    verify_protocol_hashes()
    raw = verify_grid_completeness()
    verify_aggregated_table_recomputable(raw)
    verify_crossed_bootstrap_reproducible()
    verify_figures_present()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        raise SystemExit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
