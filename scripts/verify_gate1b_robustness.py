#!/usr/bin/env python3
"""Fail-high verifier for Gate 1B — post-gate robustness of the effective-kernel mechanism.

Self-contained: it verifies only Gate 1B artifacts plus the integrity (unchanged hashes) of the
frozen Gate 1 canonical artifacts. It does NOT depend on the canonical Gate 1 verifier, whose
provenance state at HEAD is a pre-existing concern independent of this extension.

Fails if: not exactly 60 configurations; any seed/r/u0 missing; duplicates; unclassified NaN/inf;
Gate 1 canonical artifacts changed; protocol/config hashes mismatch; results not reproducible from
the stored arrays; or the frozen classification cannot be recomputed mechanically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results/eeg/gate1b_robustness"
CONFIG_PATH = ROOT / "config/effective_kernel_gate1b_robustness.json"
PROTOCOL_PATH = ROOT / "docs/gate1b_robustness_protocol.md"

TOLERANCE_METRICS = ("impulse_relative_frobenius", "step_relative_frobenius",
                     "frequency_relative_frobenius", "memory_function_l1")
DIAGNOSTIC_METRICS = ("impulse_cosine_similarity", "step_cosine_similarity")
TOLERANCES = {"impulse_relative_frobenius": 0.01, "step_relative_frobenius": 0.01,
              "frequency_relative_frobenius": 0.01, "memory_function_l1": 0.02}
CANONICAL_GATE1 = [
    "results/eeg/theory_vs_sim_check.csv", "results/eeg/theory_vs_sim_responses.npz",
    "results/eeg/theory_vs_sim_metadata.json", "results/eeg/theory_linearity_sweep.csv",
    "results/eeg/effective_kernel_symbolic.txt", "config/effective_kernel_gate1_frozen.json",
]


def fail(message: str) -> None:
    raise SystemExit(f"GATE 1B VERIFICATION FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(joint: pd.DataFrame) -> str:
    valid = joint[joint.valid]
    if len(valid) == 0:
        return "INVALID"
    tf = float(valid["tangent_all4"].mean())
    sf = float(valid["separable_all4"].mean())
    # Frozen rule, applied in spec order: NOT_ROBUST (tf<=0.50 OR sf>=0.50) takes precedence over
    # MIXED. In the residual MIXED region tf>0.50 and sf<0.50, so tangent always exceeds separable.
    if tf >= 0.90 and sf <= 0.10:
        return "ROBUST_WITHIN_GRID"
    if tf <= 0.50 or sf >= 0.50:
        return "NOT_ROBUST_WITHIN_GRID"
    return "MIXED"


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    for name in ("metrics_by_configuration.csv", "amplitude_sweep.csv",
                 "spectrum_by_configuration.csv", "summary.csv", "metadata.json", "report.md"):
        if not (OUTDIR / name).exists():
            fail(f"missing artifact: {name}")
    metadata = json.loads((OUTDIR / "metadata.json").read_text())

    # 1. provenance: config/protocol hashes match metadata
    if sha256(CONFIG_PATH) != metadata["config_sha256"]:
        fail("config SHA256 differs from metadata")
    if sha256(PROTOCOL_PATH) != metadata["protocol_sha256"]:
        fail("protocol SHA256 differs from metadata")

    # 2. canonical Gate 1 integrity: current hashes match the pre-run snapshot in metadata
    for rel in CANONICAL_GATE1:
        current = sha256(ROOT / rel)
        if current != metadata["canonical_gate1_sha256_before"].get(rel):
            fail(f"canonical Gate 1 artifact changed: {rel}")
        if current != metadata["canonical_gate1_sha256_after"].get(rel):
            fail(f"canonical Gate 1 before/after hash mismatch: {rel}")
    if not metadata.get("gate1_artifacts_unchanged"):
        fail("metadata reports Gate 1 artifacts changed")

    # 3. artifact hashes in metadata match files on disk
    for name, digest in metadata["artifact_sha256"].items():
        if sha256(OUTDIR / name) != digest:
            fail(f"artifact hash differs from metadata: {name}")

    metrics = pd.read_csv(OUTDIR / "metrics_by_configuration.csv")
    sweep = pd.read_csv(OUTDIR / "amplitude_sweep.csv")
    spectrum = pd.read_csv(OUTDIR / "spectrum_by_configuration.csv")

    # 4. exactly 60 configurations, complete grid, no duplicates
    seeds, rs, u0s = cfg["channel_seeds"], cfg["r"], cfg["operating_points"]
    expected_configs = {(int(s), float(r), float(u)) for s in seeds for r in rs for u in u0s}
    if len(expected_configs) != 60:
        fail(f"grid definition is not 60 configurations: {len(expected_configs)}")
    for frame, name in ((metrics, "metrics"), (spectrum, "spectrum")):
        got = {(int(s), float(r), float(u)) for s, r, u in zip(frame.seed, frame.r, frame.u0)}
        if got != expected_configs:
            fail(f"{name} grid mismatch: missing={expected_configs - got}, extra={got - expected_configs}")
    if len(spectrum) != 60:
        fail(f"spectrum must have exactly 60 rows, got {len(spectrum)}")
    if spectrum.duplicated(subset=["seed", "r", "u0"]).any():
        fail("duplicate configurations in spectrum")
    if len(metrics) != 60 * 12:
        fail(f"metrics must have 60x12=720 rows, got {len(metrics)}")
    if metrics.duplicated(subset=["seed", "r", "u0", "theory", "metric"]).any():
        fail("duplicate rows in metrics")

    # 5. amplitude sweep completeness
    n_eps = len(cfg["amplitude_sweep"])
    if len(sweep) != 60 * n_eps:
        fail(f"amplitude_sweep must have 60x{n_eps} rows, got {len(sweep)}")
    if not {"tangent_impulse_relative_frobenius", "tangent_step_relative_frobenius",
            "separable_impulse_relative_frobenius", "separable_step_relative_frobenius"}.issubset(sweep.columns):
        fail("amplitude_sweep missing tangent/separable impulse+step columns")

    # 6. no unclassified NaN/inf: every non-finite tolerance-metric value must have pass==False and valid==False
    tol_rows = metrics[metrics.metric.isin(TOLERANCE_METRICS)]
    nonfinite = tol_rows[~np.isfinite(tol_rows.value.astype(float))]
    if len(nonfinite):
        if nonfinite["pass"].astype(str).str.lower().eq("true").any():
            fail("non-finite tolerance metric marked as pass=True (unclassified NaN/inf)")
        if nonfinite["valid"].astype(bool).any():
            fail("non-finite tolerance metric on a configuration marked valid")
    # spectral radius must be finite OR the config invalid
    bad_radius = spectrum[~np.isfinite(spectrum.companion_spectral_radius.astype(float)) & spectrum.valid.astype(bool)]
    if len(bad_radius):
        fail("valid configuration has non-finite spectral radius")

    # 7. required columns present
    required = {"seed", "r", "K", "past_mass", "u0", "epsilon", "theory", "metric", "value",
                "tolerance", "pass", "fixed_point_converged", "fixed_iterations",
                "fixed_final_difference", "companion_spectral_radius", "companion_stable",
                "git_commit", "config_sha256", "protocol_sha256"}
    if not required.issubset(metrics.columns):
        fail(f"metrics missing required columns: {required - set(metrics.columns)}")

    # 8. pass flags reproducible from value+tolerance
    for _, row in tol_rows.iterrows():
        value = float(row.value)
        expected = bool(value <= TOLERANCES[row.metric]) if np.isfinite(value) else False
        if bool(str(row["pass"]).lower() == "true") != expected:
            fail(f"pass flag not reproducible: seed={row.seed} r={row.r} u0={row.u0} {row.theory}/{row.metric}")

    # 9. rebuild joint table and recompute classification; compare to metadata
    joint_rows = []
    for (seed, r, u0), group in metrics.groupby(["seed", "r", "u0"]):
        valid = bool(group["valid"].astype(bool).iloc[0])
        stable = bool(group["companion_stable"].astype(bool).iloc[0])

        def all4(theory: str) -> bool:
            sub = group[(group.theory == theory) & (group.metric.isin(TOLERANCE_METRICS))]
            return len(sub) == 4 and bool(sub["pass"].astype(str).str.lower().eq("true").all())
        joint_rows.append({"seed": seed, "r": r, "u0": u0, "valid": valid, "stable": stable,
                           "tangent_all4": all4("tangent_recurrence"),
                           "separable_all4": all4("separable_W_times_R")})
    joint = pd.DataFrame(joint_rows)

    recomputed = {"global": classify(joint)}
    for r in sorted(joint.r.unique()):
        recomputed[f"r={r}"] = classify(joint[joint.r == r])
    for u0 in sorted(joint.u0.unique()):
        recomputed[f"u0={u0}"] = classify(joint[joint.u0 == u0])
    for key, label in recomputed.items():
        stored = metadata["classification"].get(key, {}).get("classification")
        if stored != label:
            fail(f"classification not reproducible at {key}: stored={stored}, recomputed={label}")

    # 10. golden Gate 1 corner
    if not metadata.get("golden_gate1_corner_reproduced"):
        fail("Gate 1 corner (seed=1, r=0.7, u0=0) not reproduced")

    # 11. commit consistency across rows
    if set(metrics.git_commit) != {metadata["git_commit"]}:
        fail("metrics git_commit not uniform / differs from metadata")

    print("GATE 1B VERIFICATION: PASS")
    print(f"configurations: {len(joint)} (expected 60); duplicates: none")
    print(f"global classification: {recomputed['global']}")
    for key in ("r=0.7", "r=0.9", "u0=-0.5", "u0=0.0", "u0=0.5"):
        if key in recomputed:
            print(f"  {key}: {recomputed[key]}")
    print(f"Gate 1 canonical artifacts unchanged: {metadata['gate1_artifacts_unchanged']}")
    print(f"Gate 1 corner reproduced: {metadata['golden_gate1_corner_reproduced']}")


if __name__ == "__main__":
    main()
