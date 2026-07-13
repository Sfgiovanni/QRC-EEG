#!/usr/bin/env python3
"""Append SHA256 of every new overnight-run artifact to
provenance/eeg_checksums.txt (never overwrites the existing block) and
append a dated section to results/eeg/PROVENANCE.md documenting the new
scripts, the ESN-66 post-freeze HP-search deviation, and where every new
number comes from. Idempotent: re-running recomputes hashes for files that
changed and does not duplicate an unchanged entry.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
    "results/eeg/overnight_summary.md",
    "paper/tab_esn_matched.tex",
    "paper/tab_long_horizon_contrasts.tex",
    "paper/tab_ictal_classification.tex",
    "paper/tab_ictal_classification_contrasts.tex",
    "figures/eeg/fig_long_horizon.pdf",
    "figures/eeg/fig_long_horizon.png",
    "figures/eeg/fig_ictal_auroc.pdf",
    "figures/eeg/fig_ictal_auroc.png",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    checksums_path = ROOT / "provenance" / "eeg_checksums.txt"
    text = checksums_path.read_text()
    existing_names = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            existing_names.add(parts[1])

    new_lines = []
    for rel in NEW_FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"skip (missing): {rel}")
            continue
        digest = sha256(p)
        # remove any stale entry for this path before appending the fresh one
        text = "\n".join(line for line in text.splitlines() if not line.endswith(f"  {rel}")) + "\n"
        new_lines.append(f"{digest}  {rel}")

    # provenance/eeg_checksums.txt is plain "hash  path" lines, no markdown fence -- just append.
    text = text.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"
    checksums_path.write_text(text)
    print(f"appended {len(new_lines)} checksums to {checksums_path}")

    provenance_md = ROOT / "results" / "eeg" / "PROVENANCE.md"
    if "## Overnight run addendum" in provenance_md.read_text():
        print(f"PROVENANCE.md addendum already present, skipping (idempotent re-run)")
        return
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    section = f"""
## Overnight run addendum ({stamp})

Three follow-up items, run in one pass, reusing the existing pipeline
(`src/qrc_eeg/pipeline.py`, `statistics.py`, `splits.py`, `readout.py`) end to
end; no reservoir/kernel/channel/ESN/capacity/bootstrap code was
reimplemented.

**Post-freeze deviation (logged per `config/eeg_frozen.yaml`'s own rule):**
`scripts/run_esn66_hp_search.py` runs a real HP search for the ESN
matched-dimension control (`n_reservoir=66`, exactly the readout feature
count of every quantum arm) over the *same* `spectral_radius` x `leak_rate`
grid values already frozen for ESN-200 (`config/eeg_frozen.yaml`'s
`hp_grids.ESN`), with the same `hp_search_seeds` and `hp_search_subsample`.
`n_reservoir` itself is the new experiment's independent variable, not a
tuned knob, and is fixed to 66 rather than searched. Selected HP:
`results/eeg/hp_selected_esn66.json`. This was *not* a reuse of the
ESN-200-tuned hyperparameters at a smaller size (which would have
handicapped the classical control); giving the classical arm its own fair
HP search is what makes the matched-budget comparison in Item 2 valid.

**Item 1 (long horizon, h=4/8):** `scripts/run_long_horizon_contrasts.py`.
single_kernel vs AB_noaux reuses the existing
`results/eeg/raw/eeg_holdout_by_segment_seed.csv` (h=4/8 were already run in
the original held-out grid, just not previously summarized as their own
family). single_kernel vs ESN is reported against **ESN-66** (matched,
`results/eeg/raw/eeg_holdout_esn66_by_segment_seed.csv`), not ESN-200 --
ESN-200 is included only as a labeled-unmatched reference column. Holm
correction is *recomputed fresh* within this 12-test family (`eeg_long`);
it does not reuse `eeg_primary`'s p_holm column, which was corrected across
a different (36-test) family.

**Item 2 (ESN dimension-matched control, complete):**
`scripts/run_esn66_hp_search.py` -> `scripts/run_esn66_holdout.py`
(3 sets x 4 horizons x 10 confirmatory seeds, same split/readout/alpha grid
as every other construction) -> `scripts/run_esn66_contrasts.py`
(paired contrasts vs single_kernel, family `eeg_esn_matched`, fresh Holm
within this 12-test family). `results/eeg/tab_esn_matched.csv` carries the
original ESN-200 endpoint (`esn200_nrmse_mean_unmatched`) alongside the
matched ESN-66 endpoint for direct transparency in one table.

**Item 3 (ictal classification, pre-registered secondary task):**
`scripts/run_ictal_classification.py`. Reuses
`qrc_eeg.pipeline.construction_features` for single_kernel, dual_kernel,
AB_noaux, and ESN_66 (equalized 66-feature budget on every arm) over the
existing frozen segment-level splits (train/val/test), pooled across
Z/F/S -- label is ictal (S) vs non-ictal (Z, F), fixed by which set a
segment belongs to, so no new leakage surface is introduced by pooling.
New (not reused, because nothing existed for it): `src/qrc_eeg/classification.py`
(mean-pool-over-time feature reduction, L2-penalized logistic readout via
Newton-Raphson, rank-based AUROC/AUPRC, segment-level bootstrap CIs and
paired bootstrap Delta-AUC -- mirrors `readout.py`'s regularization
convention and `statistics.py`'s bootstrap conventions rather than
inventing new ones). Alpha selected per construction/seed on the train/val
split by validation log-loss; final readout refit on train+val, evaluated
once on held-out test; probabilities ensemble-averaged across the 10
confirmatory seeds before computing AUROC/AUPRC. **Mandatory leakage sanity
check is embedded in the script itself** (`results/eeg/ictal_shuffle_check.csv`):
with segment labels shuffled and the model refit on a held-out split, AUROC
must land in [0.35, 0.65] for every construction or the script aborts with
`SystemExit` before writing any classification table.

**Gate:** `scripts/verify_overnight.py` re-runs (not just re-reads)
single_kernel seed=1 h=1 through the live pipeline and diffs every test
segment's NRMSE against the stored raw CSV row, then separately recomputes
the RESULTS.md h=1 R^2 anchors (0.92 Z / 0.97 F / 0.97 S) directly from the
raw CSV filtered to horizon==1 (not from the horizon-averaged endpoints
table). It re-verifies split disjointness and the classification shuffle
check, checks the ESN-66 grid is complete (3x4x10 cells x 20 segments each),
confirms every new artifact has a SHA256 in `provenance/eeg_checksums.txt`,
and writes `results/eeg/overnight_summary.md` with a pre-committed,
mechanically-applied PRE-vs-PRResearch decision rule (stated in the gate
script before results were read, applied honestly to whatever came out).

New tables: `results/eeg/tab_long_horizon_contrasts.csv`,
`results/eeg/tab_esn_matched.csv`, `results/eeg/tab_ictal_classification.csv`,
`results/eeg/tab_ictal_classification_contrasts.csv` (+ matching `paper/*.tex`).
New figures: `figures/eeg/fig_long_horizon.{{pdf,png}}`,
`figures/eeg/fig_ictal_auroc.{{pdf,png}}`. New raw/diagnostic files:
`results/eeg/raw/eeg_holdout_esn66_by_segment_seed.csv`,
`results/eeg/raw/eeg_ictal_by_seed.csv`,
`results/eeg/hp_search_log_esn66.csv`, `results/eeg/hp_selected_esn66.json`,
`results/eeg/ictal_shuffle_check.csv`. Progress log with timestamps:
`results/eeg/run_overnight.log`. No existing artifact was overwritten;
every filename above is new.
"""
    with open(provenance_md, "a") as f:
        f.write(section)
    print(f"appended overnight addendum to {provenance_md}")


if __name__ == "__main__":
    main()
