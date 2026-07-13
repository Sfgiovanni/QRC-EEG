#!/usr/bin/env python3
"""Fetch, verify, and extract the Bonn EEG Z/F/S sets.

Usage:
    python scripts/fetch_eeg.py [--url URL] [--sha256 HASH]

The URL defaults to a pinned GitHub mirror (see docs/eeg_preregistration.md
for why the canonical host is dead); override with --url or the
QRC_EEG_SOURCE_URL environment variable. The expected SHA256 defaults to the
value frozen after this repository's first successful fetch; override with
--sha256 or QRC_EEG_SOURCE_SHA256 if you point at a different archive
(the script then reports and freezes whatever hash it observes, but will
never silently proceed if a hash was already frozen in
data/eeg/CHECKSUMS.txt and does not match).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.eeg_data import (  # noqa: E402
    DEFAULT_URL,
    EXPECTED_SHA256,
    REQUIRED_SETS,
    download_archive,
    extract_sets,
    load_set,
    sha256_file,
    verify_archive,
)

DATA_DIR = ROOT / "data" / "eeg"
CHECKSUMS_PATH = DATA_DIR / "CHECKSUMS.txt"


def _frozen_hash() -> str | None:
    if not CHECKSUMS_PATH.exists():
        return None
    for line in CHECKSUMS_PATH.read_text().splitlines():
        if line.startswith("archive_sha256="):
            return line.split("=", 1)[1].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("QRC_EEG_SOURCE_URL", DEFAULT_URL))
    parser.add_argument("--sha256", default=os.environ.get("QRC_EEG_SOURCE_SHA256", EXPECTED_SHA256))
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    archive_path = DATA_DIR / "raw" / "bonn_eeg_dataset.zip"
    frozen = _frozen_hash()
    expected = frozen or args.sha256

    if not archive_path.exists() or args.force_download:
        print(f"downloading: {args.url}")
        download_archive(archive_path, url=args.url)
    else:
        print(f"using cached archive: {archive_path}")

    observed = sha256_file(archive_path)
    print(f"observed sha256: {observed}")

    if expected and observed != expected:
        raise SystemExit(
            f"SHA256 mismatch: expected {expected}, got {observed}. "
            "Aborting -- data integrity cannot be confirmed."
        )
    verify_archive(archive_path, expected_sha256=observed if not expected else expected)

    extracted = extract_sets(archive_path, DATA_DIR / "sets", sets=REQUIRED_SETS)
    for set_name, set_dir in extracted.items():
        segments = load_set(set_dir)
        print(f"set {set_name}: {len(segments)} segments verified in {set_dir}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKSUMS_PATH.write_text(
        f"# Frozen on first successful fetch; do not edit by hand.\n"
        f"source_url={args.url}\n"
        f"archive_sha256={observed}\n"
        f"sets={','.join(REQUIRED_SETS)}\n"
    )
    print(f"wrote {CHECKSUMS_PATH}")


if __name__ == "__main__":
    main()
