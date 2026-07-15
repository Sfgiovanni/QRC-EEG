#!/usr/bin/env python3
"""Fail-high technical verifier for corrected r=0.7 Gate 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/eeg"
PROTOCOL = ROOT / "docs/effective_kernel_check_protocol.md"
FROZEN = ROOT / "config/effective_kernel_gate1_frozen.json"
ALLOWED = {"PASS", "FAIL_SEPARABLE_FACTORIZATION", "FAIL_LINEARIZATION"}
TOLERANCES = {
    "impulse_relative_frobenius": 0.01,
    "step_relative_frobenius": 0.01,
    "frequency_relative_frobenius": 0.01,
    "memory_function_l1": 0.02,
}


def fail(message: str) -> None:
    raise SystemExit(f"ROTA A GATE 1 FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_error(predicted: np.ndarray, measured: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - measured) / max(np.linalg.norm(measured), np.finfo(float).eps))


def memory_distribution(response: np.ndarray) -> np.ndarray:
    energy = np.sum(response * response, axis=1)
    return energy / energy.sum()


def recompute_metrics(predicted_i, predicted_s, measured_i, measured_s) -> dict[str, float]:
    return {
        "impulse_relative_frobenius": relative_error(predicted_i, measured_i),
        "step_relative_frobenius": relative_error(predicted_s, measured_s),
        "frequency_relative_frobenius": relative_error(
            np.fft.fft(predicted_i, axis=0), np.fft.fft(measured_i, axis=0)
        ),
        "memory_function_l1": float(np.abs(
            memory_distribution(predicted_i) - memory_distribution(measured_i)
        ).sum()),
    }


def check_protocol_config_and_outputs(release_audit: bool = False) -> tuple[dict, dict]:
    frozen = json.loads(FROZEN.read_text())
    protocol_hash = sha256(PROTOCOL)
    manifest = (RESULTS / "effective_kernel_protocol_frozen.sha256").read_text().split()[0]
    if protocol_hash != manifest:
        fail("frozen protocol changed after the simulation")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    committed = json.loads(subprocess.check_output(
        ["git", "show", f"{commit}:results/eeg/hp_selected.json"], cwd=ROOT, text=True
    ))
    if frozen.get("construction") != "single_kernel" or committed["single_kernel"]["hp"] != frozen.get("hp"):
        fail("authoritative committed single_kernel HP differs from frozen configuration")
    required = [
        RESULTS / "theory_vs_sim_check.csv", RESULTS / "theory_vs_sim_responses.npz",
        RESULTS / "theory_vs_sim_metadata.json", RESULTS / "theory_linearity_sweep.csv",
        RESULTS / "effective_kernel_symbolic.txt", ROOT / "docs/effective_kernel_theory.md",
        ROOT / "tests/test_effective_kernel_check.py",
        RESULTS / "_invalid_config_r09_snapshot/README.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        fail(f"missing artifacts: {missing}")
    if "INVALID_CONFIG" not in (RESULTS / "_invalid_config_r09_snapshot/README.md").read_text():
        fail("r=0.9 snapshot lacks INVALID_CONFIG classification")
    if not release_audit and (ROOT / "results/synth/theory_predictions_vs_measured.csv").exists():
        fail("Stage 2 artifact exists; mandatory Gate 1 stop was violated")
    return frozen, {"commit": commit, "protocol_hash": protocol_hash}


def check_metrics_and_arrays(frozen: dict, provenance: dict) -> tuple[str, float]:
    table = pd.read_csv(RESULTS / "theory_vs_sim_check.csv")
    required_columns = {
        "configuration", "theory", "metric", "value", "tolerance", "pass", "K", "r",
        "past_mass", "seed", "epsilon", "response_length", "git_commit", "automatic_verdict",
    }
    if not required_columns.issubset(table.columns) or len(table) != 12:
        fail("CSV schema/row count differs from six metrics for each of two theories")
    for key, value in frozen["hp"].items():
        if not np.allclose(table[key].astype(float), float(value)):
            fail(f"CSV configuration differs at {key}")
    if set(table.git_commit) != {provenance["commit"]}:
        fail("CSV commit differs from HEAD")
    verdicts = set(table.automatic_verdict)
    if len(verdicts) != 1 or next(iter(verdicts)) not in ALLOWED:
        fail(f"invalid automatic verdict: {verdicts}")
    verdict = next(iter(verdicts))
    arrays = np.load(RESULTS / "theory_vs_sim_responses.npz")
    response_keys = (
        "measured_impulse", "tangent_impulse", "separable_impulse", "measured_step",
        "tangent_step", "separable_step", "k0_impulse",
    )
    for key in response_keys:
        if arrays[key].shape != (256, 66) or not np.isfinite(arrays[key]).all():
            fail(f"bad response array {key}: {arrays[key].shape}")
    for key in ("kernel_weights", "A_eigenvalues", "companion_eigenvalues"):
        if not np.isfinite(arrays[key]).all():
            fail(f"non-finite {key}")
    if not np.isclose(arrays["kernel_weights"].sum(), 1.0, atol=1e-14):
        fail("kernel weights do not sum to one")
    if relative_error(np.cumsum(arrays["tangent_impulse"], axis=0), arrays["tangent_step"]) > 1e-10:
        fail("tangent impulse cumulative sum does not reproduce tangent step")
    flags = {}
    for theory, prefix in (("tangent_recurrence", "tangent"), ("separable_W_times_R", "separable")):
        computed = recompute_metrics(
            arrays[f"{prefix}_impulse"], arrays[f"{prefix}_step"],
            arrays["measured_impulse"], arrays["measured_step"],
        )
        flags[theory] = []
        for metric, value in computed.items():
            row = table[(table.theory == theory) & (table.metric == metric)].iloc[0]
            if not np.isclose(value, row.value, rtol=1e-12, atol=1e-14):
                fail(f"stored metric differs from arrays: {theory}/{metric}")
            passed = value <= TOLERANCES[metric]
            if bool(row["pass"]) != passed or not np.isclose(row.tolerance, TOLERANCES[metric]):
                fail(f"stored tolerance/pass differs: {theory}/{metric}")
            flags[theory].append(passed)
    mechanical = (
        "FAIL_LINEARIZATION" if not all(flags["tangent_recurrence"])
        else "FAIL_SEPARABLE_FACTORIZATION" if not all(flags["separable_W_times_R"])
        else "PASS"
    )
    if verdict != mechanical:
        fail(f"verdict {verdict} differs from mechanical classification {mechanical}")
    radius = float(np.max(np.abs(arrays["companion_eigenvalues"])))
    return verdict, radius


def check_metadata_sweep_docs(frozen: dict, provenance: dict, verdict: str, radius: float,
                              release_audit: bool = False) -> None:
    metadata = json.loads((RESULTS / "theory_vs_sim_metadata.json").read_text(), parse_constant=lambda x: fail(f"non-JSON constant {x}"))
    if metadata["hp"] != frozen["hp"] or metadata["construction"] != "single_kernel":
        fail("metadata configuration differs")
    if metadata["git_commit"] != provenance["commit"] or metadata["protocol_sha256"] != provenance["protocol_hash"]:
        fail("metadata commit/protocol provenance differs")
    if metadata["automatic_verdict"] != verdict:
        fail("metadata verdict differs")
    companion = metadata["companion"]
    if companion["dimension"] != 4080 or not np.isclose(companion["spectral_radius"], radius, rtol=1e-12):
        fail("companion metadata differs from stored spectrum")
    if companion["locally_stable"] != (radius < 1):
        fail("companion stability classification differs")
    for relative, digest in metadata["artifact_sha256"].items():
        if release_audit and relative == "docs/rotaA_plan.md":
            continue  # cross-stage navigation is intentionally updated after the frozen Gate 1 run
        if sha256(ROOT / relative) != digest:
            fail(f"metadata artifact hash differs: {relative}")
    sweep = pd.read_csv(RESULTS / "theory_linearity_sweep.csv")
    expected_eps = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    if len(sweep) != len(expected_eps) or not np.allclose(sweep.epsilon, expected_eps, rtol=0, atol=1e-15):
        fail("post-gate sweep amplitudes differ from protocol")
    if set(sweep.confirmatory_verdict_unchanged) != {verdict}:
        fail("sweep changed or omitted confirmatory verdict")
    theory = (ROOT / "docs/effective_kernel_theory.md").read_text()
    for marker in (
        "H_{\\mathrm{actual}}", "H_{\\mathrm{sep}}", "not generally equal", "apparent pole at",
        "not by itself sufficient", "T_eff` sozinho", verdict, "Estágio 2 não foi executado",
    ):
        if marker not in theory:
            fail(f"theory document lacks required statement: {marker}")


def tests_and_provenance() -> None:
    if subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode:
        fail("pytest failed")
    if subprocess.run([sys.executable, "scripts/make_provenance.py"], cwd=ROOT).returncode:
        fail("provenance generation failed")
    checksums = (ROOT / "provenance/eeg_checksums.txt").read_text()
    required = (
        "config/effective_kernel_gate1_frozen.json", "docs/effective_kernel_check_protocol.md",
        "docs/effective_kernel_theory.md", "results/eeg/theory_vs_sim_check.csv",
        "results/eeg/theory_vs_sim_responses.npz", "results/eeg/theory_vs_sim_metadata.json",
        "results/eeg/theory_linearity_sweep.csv", "scripts/run_effective_kernel_check.py",
        "tests/test_effective_kernel_check.py", "results/eeg/_invalid_config_r09_snapshot/README.md",
    )
    for relative in required:
        if relative not in checksums:
            fail(f"SHA256 missing: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-audit", action="store_true",
                        help="verify frozen Gate 1 evidence after later-stage artifacts exist")
    args = parser.parse_args()
    frozen, provenance = check_protocol_config_and_outputs(args.release_audit)
    verdict, radius = check_metrics_and_arrays(frozen, provenance)
    check_metadata_sweep_docs(frozen, provenance, verdict, radius, args.release_audit)
    tests_and_provenance()
    print(f"\nROTA A GATE 1 VERIFICATION: PASS; SCIENTIFIC VERDICT: {verdict}")
    print(f"configuration: {frozen['construction']} {frozen['hp']}; commit: {provenance['commit']}")
    print(f"companion spectral radius: {radius:.12g}; locally stable: {radius < 1}")
    print("r=0.9 artifacts: preserved as INVALID_CONFIG")
    print("release audit: later-stage artifacts allowed" if args.release_audit else "Stage 2 artifacts: absent")


if __name__ == "__main__":
    main()
