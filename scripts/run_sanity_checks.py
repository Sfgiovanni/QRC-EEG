#!/usr/bin/env python3
"""Run the mechanism sanity-check suite; abort the pipeline on any failure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_mechanism_checks.py",
            "tests/test_constructions_differ.py",
            "tests/test_batched_matches_reference.py",
            "tests/test_segment_grouped_selection.py",
            "-v",
        ],
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise SystemExit("sanity checks failed; aborting pipeline")
    print("sanity checks: ok")


if __name__ == "__main__":
    main()
