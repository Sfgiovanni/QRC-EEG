"""Bonn EEG dataset acquisition and verification.

Not present in either source repository (both target glucose data). The
canonical host (epileptologie-bonn.de / meb.uni-bonn.de) has been retired and
now redirects to a generic homepage; archive.ics.uci.edu and the UPF NTSA
mirror both return HTTP 403 to automated fetches. Per the maintainer
(2026-07-12), this module fetches the same original 2001 Andrzejak release
(unmodified 100 files x 4097 samples per set) from an unblocked GitHub
mirror, pinned to a specific commit; see docs/eeg_preregistration.md for the
full amendment note.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_URL = (
    "https://raw.githubusercontent.com/RYH2077/EEG-Epilepsy-Datasets/"
    "185859ab48bc701c9a10f6bb2b5f76d8e28e4003/Bonn%20EEG%20dataset.zip"
)
EXPECTED_SHA256 = "f4c2dc52fd5320d4404fcbc6ecb9db69a4a7e408df4e3d5456530343dbcb75ad"

SET_FOLDERS = {
    "Z": "A_Z",
    "O": "B_O",
    "N": "C_N",
    "F": "D_F",
    "S": "E_S",
}
REQUIRED_SETS = ("Z", "F", "S")
N_SEGMENTS_PER_SET = 100
N_SAMPLES_PER_SEGMENT = 4097
SAMPLING_RATE_HZ = 173.61


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_archive(dest: Path, url: str = DEFAULT_URL) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)
    return dest


def verify_archive(path: Path, expected_sha256: str = EXPECTED_SHA256) -> str:
    if not path.exists():
        raise FileNotFoundError(f"archive not found: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected_sha256}, got {observed}. "
            "Aborting -- do not proceed with unverified data."
        )
    return observed


def extract_sets(archive_path: Path, out_dir: Path, sets: tuple[str, ...] = REQUIRED_SETS) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        for set_name in sets:
            folder = SET_FOLDERS[set_name]
            members = [n for n in names if n.startswith(f"{folder}/") and n.endswith(".txt")]
            if len(members) != N_SEGMENTS_PER_SET:
                raise ValueError(f"set {set_name}: expected {N_SEGMENTS_PER_SET} files, found {len(members)}")
            set_dir = out_dir / set_name
            set_dir.mkdir(parents=True, exist_ok=True)
            for member in members:
                zf.extract(member, out_dir / "_raw")
            src_dir = out_dir / "_raw" / folder
            for f in sorted(src_dir.glob("*.txt")):
                target = set_dir / f.name
                if not target.exists():
                    f.replace(target)
            result[set_name] = set_dir
    return result


def load_segment(path: Path) -> "list[float]":
    with path.open() as fh:
        values = [float(line.strip()) for line in fh if line.strip()]
    if len(values) != N_SAMPLES_PER_SEGMENT:
        raise ValueError(f"{path}: expected {N_SAMPLES_PER_SEGMENT} samples, got {len(values)}")
    return values


def load_set(set_dir: Path) -> "dict[str, list[float]]":
    files = sorted(set_dir.glob("*.txt"))
    if len(files) != N_SEGMENTS_PER_SET:
        raise ValueError(f"{set_dir}: expected {N_SEGMENTS_PER_SET} segment files, found {len(files)}")
    return {f.stem: load_segment(f) for f in files}
