#!/usr/bin/env python3
"""Compute SHA256 of every generated CSV/figure and write
results/eeg/PROVENANCE.md + provenance/eeg_checksums.txt.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_GLOBS = [
    "results/eeg/*.csv",
    "results/eeg/raw/*.csv",
    "results/eeg/bugfix_diff_report.md",
    "results/eeg/fase1_diff_report.md",
    "results/eeg/_prefix_snapshot_hpselect/*",
    "results/eeg/_prefix_snapshot_hpselect/raw/*.csv",
    "figures/eeg/*.pdf",
    "figures/eeg/*.png",
    "paper/*.tex",
    "data/eeg/CHECKSUMS.txt",
    "data/eeg/splits/*.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries = []
    for pattern in OUTPUT_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                entries.append((path.relative_to(ROOT), sha256_file(path)))

    provenance_dir = ROOT / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    checksums_path = provenance_dir / "eeg_checksums.txt"
    with checksums_path.open("w") as f:
        for rel_path, digest in entries:
            f.write(f"{digest}  {rel_path}\n")

    md_lines = [
        "# EEG Study Provenance",
        "",
        "Every number in the EEG tables/figures traces to a script + source file below.",
        "",
        "## Pipeline (run in order by scripts/run_eeg.sh)",
        "",
        "1. `scripts/fetch_eeg.py` -> `data/eeg/sets/{Z,F,S}/*.txt`, `data/eeg/CHECKSUMS.txt`",
        "2. `scripts/run_sanity_checks.py` -> pytest mechanism suite (must pass to proceed)",
        "3. `scripts/run_hp_search.py`, `run_esn66_hp_search.py` -> segment-blocked HP selections",
        "4. `scripts/run_holdout_eval.py`, `run_esn66_holdout.py` -> held-out raw rows and selected-alpha logs",
        "5. `scripts/run_quadratic_capacity.py` -> validation-selected capacity tables",
        "6. `scripts/run_statistics.py`, `run_esn66_contrasts.py` -> paired segment contrasts",
        "7. `scripts/run_tables_figures.py`, `run_overnight_figures.py` -> tables and figures",
        "8. `scripts/update_results_fase1.py`, `make_fase1_diff_report.py` -> regenerated narrative and diff",
        "9. `scripts/verify_fase1.py` -> final tests, reproduction checks and these checksums",
        "",
        "## File checksums (SHA256)",
        "",
        "```",
    ]
    for rel_path, digest in entries:
        md_lines.append(f"{digest}  {rel_path}")
    md_lines.append("```")

    (ROOT / "results" / "eeg" / "PROVENANCE.md").write_text("\n".join(md_lines) + "\n")
    print(f"wrote {checksums_path} ({len(entries)} files)")
    print("wrote results/eeg/PROVENANCE.md")


if __name__ == "__main__":
    main()
