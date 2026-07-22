#!/usr/bin/env python3
"""Crossed segment x seed sensitivity analysis (docs/crossed_inference_protocol.md).

Runs, for the frozen family `eeg_followup_crossed_sensitivity` (21 tests):

- the two-factor crossed bootstrap (primary sensitivity endpoint);
- a side-by-side replication of the canonical seed-averaged-then-segment
  analysis (comparison only -- does not touch results/eeg/gate_interactions.csv);
- an attempted crossed mixed model, with convergence/singularity/boundary
  diagnostics recorded verbatim, never hidden.

Does not modify any canonical Gate/Gate1B/Gate2/Gate3 artifact.

Writes, under results/eeg/followup/crossed_inference/:
  crossed_bootstrap.csv
  original_style_replication.csv
  mixed_model_results.csv
  mixed_model_diagnostics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.crossed_inference import (  # noqa: E402
    crossed_bootstrap,
    fit_crossed_mixed_model,
    interaction_matrix,
    original_style_interaction,
)
from qrc_eeg.statistics import holm  # noqa: E402

CONFIG_PATH = ROOT / "config" / "esn_distributed_memory_frozen.yaml"
RESULTS_DIR = ROOT / "results" / "eeg"
FOLLOWUP_DIR = RESULTS_DIR / "followup"
OUT_DIR = FOLLOWUP_DIR / "crossed_inference"


def main() -> None:
    fcfg = yaml.safe_load(CONFIG_PATH.read_text())
    ci_cfg = fcfg["crossed_inference"]
    h_short, h_long = ci_cfg["h_short"], ci_cfg["h_long"]
    n_replicates = ci_cfg["n_bootstrap_replicates"]
    bootstrap_seed = ci_cfg["bootstrap_rng_seed"]
    sets = ci_cfg["sets"]
    contrasts = ci_cfg["contrasts"]

    primary = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_by_segment_seed.csv")
    esn66 = pd.read_csv(RESULTS_DIR / "raw" / "eeg_holdout_esn66_by_segment_seed.csv")
    qrc_df = pd.concat([primary, esn66], ignore_index=True)

    followup_path = FOLLOWUP_DIR / "raw" / "esn_distributed_memory_holdout_by_segment_seed.csv"
    followup = pd.read_csv(followup_path)

    # Single shared RNG stream, advanced sequentially in this fixed iteration
    # order (set outer, contrast middle, mode inner) so the whole run is
    # exactly reproducible end to end from bootstrap_rng_seed.
    rng = np.random.default_rng(bootstrap_seed)
    rng_original = np.random.default_rng(bootstrap_seed)  # separate stream for the replication table

    bootstrap_rows, original_rows, mixed_rows, diagnostics = [], [], [], []

    for set_name in sets:
        for contrast in contrasts:
            kernel, comparator = contrast["kernel"], contrast["comparator"]
            for mode in contrast["modes"]:
                if mode is None:
                    df = qrc_df
                else:
                    df = followup[followup["analysis_mode"] == mode]

                mat, segs, seeds = interaction_matrix(df, kernel, comparator, set_name, h_short, h_long)
                boot = crossed_bootstrap(mat, rng, n_replicates=n_replicates)
                row = {
                    "set": set_name, "kernel": kernel, "comparator": comparator,
                    "analysis_mode": mode if mode is not None else "not_applicable",
                    "comparison": f"{kernel} vs {comparator}" + (f" ({mode})" if mode else ""),
                    **boot,
                }
                bootstrap_rows.append(row)

                orig = original_style_interaction(df, kernel, comparator, set_name, h_short, h_long, rng_original, n_replicates=n_replicates)
                original_rows.append({
                    "set": set_name, "kernel": kernel, "comparator": comparator,
                    "analysis_mode": mode if mode is not None else "not_applicable", **orig,
                })

                mm = fit_crossed_mixed_model(df, kernel, comparator, set_name, h_short, h_long)
                mm["analysis_mode"] = mode if mode is not None else "not_applicable"
                mixed_row = {k: v for k, v in mm.items() if k not in ("warnings", "variance_components")}
                mixed_row["kernel"] = kernel
                mixed_row["comparator"] = comparator
                mixed_rows.append(mixed_row)
                diagnostics.append(mm)
                print(
                    f"{set_name} {kernel} vs {comparator} ({row['analysis_mode']}): "
                    f"boot mean={boot['bootstrap_mean']:+.5f} CI=[{boot['ci95_lo']:+.5f},{boot['ci95_hi']:+.5f}] "
                    f"sign_frac={boot['sign_fraction']:.3f} | mixed converged={mm['converged']} boundary={mm['boundary_hit']}",
                    flush=True,
                )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    assert len(bootstrap_df) == 21, f"expected 21 tests in eeg_followup_crossed_sensitivity, got {len(bootstrap_df)}"
    bootstrap_df["p_holm"] = holm(bootstrap_df["p_bootstrap"].to_numpy())
    bootstrap_df["expected_direction"] = bootstrap_df["bootstrap_mean"] > 0
    bootstrap_df["significant_expected"] = (
        bootstrap_df["expected_direction"] & (bootstrap_df["ci95_lo"] > 0) & (bootstrap_df["p_holm"] < 0.05)
    )

    original_df = pd.DataFrame(original_rows)
    mixed_df = pd.DataFrame(mixed_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bootstrap_df.to_csv(OUT_DIR / "crossed_bootstrap.csv", index=False)
    original_df.to_csv(OUT_DIR / "original_style_replication.csv", index=False)
    mixed_df.to_csv(OUT_DIR / "mixed_model_results.csv", index=False)
    (OUT_DIR / "mixed_model_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str))

    print("wrote", OUT_DIR / "crossed_bootstrap.csv", f"({len(bootstrap_df)} rows)")
    print("wrote", OUT_DIR / "original_style_replication.csv", f"({len(original_df)} rows)")
    print("wrote", OUT_DIR / "mixed_model_results.csv", f"({len(mixed_df)} rows)")
    print("wrote", OUT_DIR / "mixed_model_diagnostics.json")

    n_boundary = sum(1 for d in diagnostics if d.get("boundary_hit"))
    n_nonconverged = sum(1 for d in diagnostics if not d.get("converged"))
    print(f"mixed model: {n_nonconverged}/21 non-converged, {n_boundary}/21 boundary-flagged")


if __name__ == "__main__":
    main()
