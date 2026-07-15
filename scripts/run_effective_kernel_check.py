#!/usr/bin/env python3
"""Corrected, provenance-locked effective-kernel Gate 1 and post-gate epsilon sweep."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.batched import batched_channel_step, run_batched_reservoir  # noqa: E402
from qrc_eeg.channels import build_input_channel  # noqa: E402
from qrc_eeg.models import pure_zero_state  # noqa: E402
from qrc_eeg.observables import local_pauli_observables  # noqa: E402
from qrc_eeg.state_kernels import single_exponential_weights  # noqa: E402

RESULTS = ROOT / "results/eeg"
FROZEN_CONFIG = ROOT / "config/effective_kernel_gate1_frozen.json"
PROTOCOL = ROOT / "docs/effective_kernel_check_protocol.md"
HP_PATH = "results/eeg/hp_selected.json"
U0 = 0.0
FIXED_TOLERANCE = 1.0e-13
FIXED_MAX_ITERATIONS = 5000
TOLERANCES = {
    "impulse_relative_frobenius": 0.01,
    "step_relative_frobenius": 0.01,
    "frequency_relative_frobenius": 0.01,
    "memory_function_l1": 0.02,
}
SWEEP_EPSILONS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


class GateInvalid(RuntimeError):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise GateInvalid("INVALID_PROVENANCE", f"git provenance failed for {args}: {exc}") from exc


def validate_configuration(document: dict, frozen: dict) -> tuple[str, dict]:
    """Fail instead of silently running a construction or HP different from the freeze."""

    construction = frozen.get("construction")
    if construction != "single_kernel" or construction not in document:
        raise GateInvalid("INVALID_CONFIG", f"required construction single_kernel is absent: {construction}")
    hp = document[construction].get("hp")
    if hp != frozen.get("hp"):
        raise GateInvalid("INVALID_CONFIG", f"committed HP changed: loaded={hp}, frozen={frozen.get('hp')}")
    return construction, hp


def load_official_configuration() -> tuple[dict, dict]:
    """Load HP from the committed hp_selected.json; never mutate the dirty working copy."""

    frozen = json.loads(FROZEN_CONFIG.read_text())
    commit = git_output("rev-parse", "HEAD")
    try:
        document = json.loads(git_output("show", f"{commit}:{HP_PATH}"))
    except (json.JSONDecodeError, GateInvalid) as exc:
        raise GateInvalid("INVALID_PROVENANCE", f"cannot read committed {HP_PATH}: {exc}") from exc
    construction, hp = validate_configuration(document, frozen)
    print(f"loaded construction: {construction}")
    print(f"loaded hyperparameters: {json.dumps(hp, sort_keys=True)}")
    print(f"git commit: {commit}")
    return frozen, {"commit": commit, "construction": construction, "hp": hp, "document": document}


def hermitian_traceless_basis(dimension: int) -> np.ndarray:
    """Orthonormal generalized Gell-Mann basis, excluding normalized identity."""

    basis: list[np.ndarray] = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            symmetric = np.zeros((dimension, dimension), dtype=np.complex128)
            symmetric[i, j] = symmetric[j, i] = 1.0 / np.sqrt(2.0)
            basis.append(symmetric)
            antisymmetric = np.zeros((dimension, dimension), dtype=np.complex128)
            antisymmetric[i, j] = -1j / np.sqrt(2.0)
            antisymmetric[j, i] = 1j / np.sqrt(2.0)
            basis.append(antisymmetric)
    for k in range(1, dimension):
        diagonal = np.zeros((dimension, dimension), dtype=np.complex128)
        diagonal[np.arange(k), np.arange(k)] = 1.0
        diagonal[k, k] = -float(k)
        diagonal /= np.sqrt(k * (k + 1.0))
        basis.append(diagonal)
    result = np.asarray(basis)
    if result.shape != (dimension * dimension - 1, dimension, dimension):
        raise RuntimeError(f"unexpected tangent basis shape: {result.shape}")
    gram = np.real(np.einsum("aij,bji->ab", result, result))
    if not np.allclose(gram, np.eye(len(result)), atol=1e-12):
        raise RuntimeError("traceless-Hermitian basis is not orthonormal")
    return result


def normalize_density(rho: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (rho + rho.conj().T)
    return hermitian / np.trace(hermitian).real


def fixed_state(channel, dimension: int) -> tuple[np.ndarray, int, float]:
    rho = pure_zero_state(dimension)
    difference = float("inf")
    for iteration in range(1, FIXED_MAX_ITERATIONS + 1):
        updated = normalize_density(batched_channel_step(channel, np.array([U0]), rho[None])[0])
        difference = float(np.linalg.norm(updated - rho))
        rho = updated
        if difference < FIXED_TOLERANCE:
            return rho, iteration, difference
    raise GateInvalid("FAIL_LINEARIZATION", f"fixed state failed to converge: {difference}")


def coordinates(basis: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """Coordinates of one matrix or a batch in the Hermitian basis."""

    values = np.asarray(matrices)
    if values.ndim == 2:
        return np.real(np.einsum("aij,ji->a", basis, values))
    return np.real(np.einsum("aij,bji->ab", basis, values))


def build_abc(channel, rho_star: np.ndarray, basis: np.ndarray, observables: np.ndarray, derivative_epsilon: float):
    propagated_basis = batched_channel_step(channel, np.full(len(basis), U0), basis)
    A = coordinates(basis, propagated_basis)
    plus = normalize_density(batched_channel_step(channel, np.array([U0 + derivative_epsilon]), rho_star[None])[0])
    minus = normalize_density(batched_channel_step(channel, np.array([U0 - derivative_epsilon]), rho_star[None])[0])
    B = coordinates(basis, (plus - minus) / (2.0 * derivative_epsilon))
    C = np.real(np.einsum("kij,aji->ka", observables, basis))
    return A, B, C


def tangent_response(A, B, C, weights: np.ndarray, signal: np.ndarray) -> np.ndarray:
    dimension = len(B)
    K = len(weights) - 1
    current = np.zeros(dimension)
    history = np.zeros((K + 1, dimension))
    response = np.empty((len(signal), C.shape[0]))
    for step, value in enumerate(signal):
        mixed = weights[0] * current
        for lag in range(1, K + 1):
            mixed += weights[lag] * history[-1 - lag]
        current = A @ mixed + B * value
        history = np.concatenate([history[1:], current[None]], axis=0)
        response[step] = C @ current
    return response


def nonlinear_response(kernel, channel, rho_star, signal: np.ndarray, epsilon: float) -> np.ndarray:
    baseline = np.full((1, len(signal)), U0)
    perturbed = baseline + epsilon * signal[None]
    base = run_batched_reservoir(kernel, channel, rho_star, baseline, check_every=len(signal)).features[0]
    measured = run_batched_reservoir(kernel, channel, rho_star, perturbed, check_every=len(signal)).features[0]
    return (measured - base) / epsilon


def separable_response(k0_impulse: np.ndarray, weights: np.ndarray) -> np.ndarray:
    response = np.empty_like(k0_impulse)
    for feature in range(k0_impulse.shape[1]):
        response[:, feature] = np.convolve(k0_impulse[:, feature], weights, mode="full")[: len(k0_impulse)]
    return response


def relative_frobenius(predicted: np.ndarray, measured: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - measured) / max(np.linalg.norm(measured), np.finfo(float).eps))


def cosine_similarity(predicted: np.ndarray, measured: np.ndarray) -> float:
    a, b = predicted.ravel(), measured.ravel()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), np.finfo(float).eps))


def memory_distribution(response: np.ndarray) -> np.ndarray:
    energy = np.sum(response * response, axis=1)
    return energy / max(float(energy.sum()), np.finfo(float).eps)


def theory_metrics(predicted_impulse, predicted_step, measured_impulse, measured_step) -> dict[str, float]:
    return {
        "impulse_relative_frobenius": relative_frobenius(predicted_impulse, measured_impulse),
        "step_relative_frobenius": relative_frobenius(predicted_step, measured_step),
        "frequency_relative_frobenius": relative_frobenius(
            np.fft.fft(predicted_impulse, axis=0), np.fft.fft(measured_impulse, axis=0)
        ),
        "memory_function_l1": float(np.sum(np.abs(memory_distribution(predicted_impulse) - memory_distribution(measured_impulse)))),
        "impulse_cosine_similarity": cosine_similarity(predicted_impulse, measured_impulse),
        "step_cosine_similarity": cosine_similarity(predicted_step, measured_step),
    }


def classify_verdict(rows: list[dict]) -> str:
    confirmatory = [row for row in rows if row["metric"] in TOLERANCES]
    tangent = [bool(row["pass"]) for row in confirmatory if row["theory"] == "tangent_recurrence"]
    separable = [bool(row["pass"]) for row in confirmatory if row["theory"] == "separable_W_times_R"]
    if len(tangent) != 4 or len(separable) != 4:
        return "INVALID_PROVENANCE"
    if not all(tangent):
        return "FAIL_LINEARIZATION"
    if not all(separable):
        return "FAIL_SEPARABLE_FACTORIZATION"
    return "PASS"


def companion_spectrum(A: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact eigenvalue union from modal companion polynomials."""

    a_eigenvalues = np.linalg.eigvals(A)
    roots = []
    for eigenvalue in a_eigenvalues:
        coefficients = np.concatenate([[1.0 + 0j], -eigenvalue * weights.astype(complex)])
        roots.extend(np.roots(coefficients))
    companion = np.asarray(roots)
    if not np.isfinite(companion).all():
        raise GateInvalid("FAIL_LINEARIZATION", "non-finite companion spectrum")
    return a_eigenvalues, companion, float(np.max(np.abs(companion)))


def symbolic_record(K: int, conditional_lag: float, full_lag: float) -> str:
    z, r, m, a, W = sp.symbols("z r m a W", nonzero=True)
    normalization = sum(r**lag for lag in range(1, K + 1))
    normalization_closed = r * (1 - r**K) / (1 - r)
    delayed = sum((r / z) ** lag for lag in range(1, K + 1))
    delayed_closed = (r / z) * (1 - (r / z) ** K) / (1 - r / z)
    if sp.simplify(normalization - normalization_closed) != 0 or sp.simplify(delayed - delayed_closed) != 0:
        raise GateInvalid("INVALID_PROVENANCE", "SymPy finite-geometric identity failed")
    difference = sp.factor(sp.together(1 / (z - a * W) - W / (z - a)))
    return "\n".join([
        f"K = {K}", "S_K(r) = r*(1-r**K)/(1-r)",
        "W_K(z) = (1-m) + m*((r/z)*(1-(r/z)**K)/(1-r/z))/S_K(r)",
        "H_actual modal = 1/(z-a*W_K(z))", "H_separable modal = W_K(z)/(z-a)",
        f"generic difference = {sp.sstr(difference)}", f"generic equality: {difference == 0}",
        f"conditional delayed T_eff = {conditional_lag:.12g}", f"full mean lag including w0 = {full_lag:.12g}",
    ]) + "\n"


def ensure_arrays(arrays: dict[str, np.ndarray], shape: tuple[int, int]) -> None:
    for name, array in arrays.items():
        if name.endswith("eigenvalues") or name == "kernel_weights":
            if not np.isfinite(array).all():
                raise GateInvalid("INVALID_PROVENANCE", f"non-finite array: {name}")
            continue
        if array.shape != shape or not np.isfinite(array).all():
            raise GateInvalid("INVALID_PROVENANCE", f"bad shape/finiteness for {name}: {array.shape}")


def main() -> None:
    frozen, provenance = load_official_configuration()
    hp = provenance["hp"]
    seed = int(frozen["seed"])
    epsilon = float(frozen["epsilon"])
    response_length = int(frozen["response_length"])
    kernel = single_exponential_weights(**hp)
    weights = np.concatenate([[kernel.present], kernel.delayed])
    if not np.isclose(weights.sum(), 1.0, atol=1e-14):
        raise GateInvalid("INVALID_CONFIG", f"kernel weights do not sum to one: {weights.sum()}")

    channel = build_input_channel(n_qubits=4, seed=seed)
    _, observable_list = local_pauli_observables(4)
    observables = np.asarray(observable_list)
    if observables.shape != (66, 16, 16):
        raise GateInvalid("INVALID_PROVENANCE", f"expected 66 Pauli observables, got {observables.shape}")
    rho_star, fixed_iterations, fixed_difference = fixed_state(channel, 16)
    basis = hermitian_traceless_basis(16)
    A, B, C = build_abc(channel, rho_star, basis, observables, derivative_epsilon=epsilon)

    impulse = np.zeros(response_length); impulse[0] = 1.0
    step = np.ones(response_length)
    tangent_impulse = tangent_response(A, B, C, weights, impulse)
    tangent_step = tangent_response(A, B, C, weights, step)
    k0_impulse = tangent_response(A, B, C, np.array([1.0]), impulse)
    separable_impulse = separable_response(k0_impulse, weights)
    separable_step = np.cumsum(separable_impulse, axis=0)
    measured_impulse = nonlinear_response(kernel, channel, rho_star, impulse, epsilon)
    measured_step = nonlinear_response(kernel, channel, rho_star, step, epsilon)

    a_eigenvalues, companion_eigenvalues, spectral_radius = companion_spectrum(A, weights)
    locally_stable = spectral_radius < 1.0
    cumulative_consistency = relative_frobenius(np.cumsum(tangent_impulse, axis=0), tangent_step)
    if cumulative_consistency > 1e-10:
        raise GateInvalid("INVALID_PROVENANCE", f"tangent impulse/step LTI inconsistency: {cumulative_consistency}")

    arrays = {
        "measured_impulse": measured_impulse, "tangent_impulse": tangent_impulse,
        "separable_impulse": separable_impulse, "measured_step": measured_step,
        "tangent_step": tangent_step, "separable_step": separable_step, "k0_impulse": k0_impulse,
        "kernel_weights": weights, "A_eigenvalues": a_eigenvalues,
        "companion_eigenvalues": companion_eigenvalues,
    }
    ensure_arrays(arrays, (response_length, 66))

    configuration = frozen["configuration"]
    rows = []
    for theory, predicted_impulse, predicted_step in (
        ("tangent_recurrence", tangent_impulse, tangent_step),
        ("separable_W_times_R", separable_impulse, separable_step),
    ):
        for metric, value in theory_metrics(predicted_impulse, predicted_step, measured_impulse, measured_step).items():
            tolerance = TOLERANCES.get(metric)
            rows.append({
                "configuration": configuration, "theory": theory, "metric": metric,
                "value": value, "tolerance": tolerance,
                "pass": bool(value <= tolerance) if tolerance is not None else np.nan,
                "K": hp["K"], "r": hp["r"], "past_mass": hp["past_mass"], "seed": seed,
                "epsilon": epsilon, "response_length": response_length, "git_commit": provenance["commit"],
            })
    verdict = classify_verdict(rows)
    for row in rows:
        row["automatic_verdict"] = verdict

    csv_path = RESULTS / "theory_vs_sim_check.csv"
    npz_path = RESULTS / "theory_vs_sim_responses.npz"
    sweep_path = RESULTS / "theory_linearity_sweep.csv"
    symbolic_path = RESULTS / "effective_kernel_symbolic.txt"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **arrays)

    conditional_lag = float(np.dot(np.arange(1, kernel.K + 1), kernel.delayed) / kernel.past_mass)
    full_lag = float(np.dot(np.arange(1, kernel.K + 1), kernel.delayed))
    symbolic_path.write_text(symbolic_record(kernel.K, conditional_lag, full_lag), encoding="utf-8")

    # Secondary post-gate robustness. It cannot alter `verdict` above.
    sweep_rows = []
    for sweep_epsilon in SWEEP_EPSILONS:
        sweep_impulse = nonlinear_response(kernel, channel, rho_star, impulse, sweep_epsilon)
        sweep_step = nonlinear_response(kernel, channel, rho_star, step, sweep_epsilon)
        sweep_rows.append({
            "configuration": configuration, "epsilon": sweep_epsilon,
            "impulse_relative_frobenius": relative_frobenius(tangent_impulse, sweep_impulse),
            "step_relative_frobenius": relative_frobenius(tangent_step, sweep_step),
            "confirmatory_epsilon": sweep_epsilon == epsilon, "confirmatory_verdict_unchanged": verdict,
            "K": hp["K"], "r": hp["r"], "past_mass": hp["past_mass"], "git_commit": provenance["commit"],
        })
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)

    status_lines = git_output("status", "--porcelain").splitlines()
    working_hp = json.loads((ROOT / HP_PATH).read_text())["single_kernel"]["hp"]
    artifact_paths = (
        csv_path, npz_path, sweep_path, symbolic_path, PROTOCOL, FROZEN_CONFIG,
        ROOT / "docs/effective_kernel_theory.md", ROOT / "docs/rotaA_plan.md",
        ROOT / "scripts/run_effective_kernel_check.py",
    )
    artifact_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths}
    metadata = {
        "configuration": configuration, "construction": provenance["construction"], "hp": hp,
        "hp_source": frozen["hp_source"], "working_tree_hp": working_hp,
        "working_tree_hp_diverges_from_official": working_hp != hp,
        "git_commit": provenance["commit"], "git_working_tree_clean": not status_lines,
        "git_status_porcelain": status_lines, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "sympy_version": sp.__version__,
        "timestamp": datetime.now(ZoneInfo(frozen["timezone"])).isoformat(), "timezone": frozen["timezone"],
        "seed": seed, "epsilon": epsilon, "response_length": response_length,
        "fixed_state": {"tolerance": FIXED_TOLERANCE, "max_iterations": FIXED_MAX_ITERATIONS,
                        "iterations": fixed_iterations, "final_difference": fixed_difference, "converged": True},
        "kernel": {"weights_sum": float(weights.sum()), "conditional_delayed_T_eff": conditional_lag,
                   "full_mean_lag_including_w0": full_lag},
        "tangent": {"state_dimension": len(B), "A_shape": list(A.shape), "B_shape": list(B.shape),
                    "C_shape": list(C.shape), "impulse_cumsum_step_error": cumulative_consistency},
        "companion": {"dimension": int((kernel.K + 1) * len(B)), "spectral_radius": spectral_radius,
                      "locally_stable": locally_stable,
                      "dominant_eigenvalues": [
                          {"real": float(value.real), "imag": float(value.imag), "abs": float(abs(value))}
                          for value in companion_eigenvalues[np.argsort(np.abs(companion_eigenvalues))[-20:][::-1]]
                      ]},
        "automatic_verdict": verdict,
        "confirmatory_metrics": [
            {key: (None if isinstance(value, float) and np.isnan(value) else value) for key, value in row.items()}
            for row in rows
        ],
        "post_gate_sweep_does_not_change_verdict": True,
        "protocol_sha256": sha256(PROTOCOL), "frozen_config_sha256": sha256(FROZEN_CONFIG),
        "artifact_sha256": artifact_hashes,
    }
    metadata_path = RESULTS / "theory_vs_sim_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print(pd.DataFrame(rows)[["theory", "metric", "value", "tolerance", "pass"]].to_string(index=False))
    print(f"automatic verdict: {verdict}")
    print(f"fixed point: iterations={fixed_iterations}, difference={fixed_difference:.3e}, tolerance={FIXED_TOLERANCE:.1e}")
    print(f"companion: dimension={(kernel.K + 1) * len(B)}, spectral_radius={spectral_radius:.12g}, stable={locally_stable}")
    print(f"conditional T_eff={conditional_lag:.12g}; full mean lag={full_lag:.12g}")


if __name__ == "__main__":
    try:
        main()
    except GateInvalid as exc:
        print(f"automatic verdict: {exc.verdict}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
