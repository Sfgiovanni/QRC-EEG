#!/usr/bin/env python3
"""Falsifiable Stage-1 check of tangent and separable effective-kernel theories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.batched import batched_channel_step, run_batched_reservoir  # noqa: E402
from qrc_eeg.channels import build_input_channel  # noqa: E402
from qrc_eeg.models import pure_zero_state  # noqa: E402
from qrc_eeg.observables import local_pauli_observables  # noqa: E402
from qrc_eeg.state_kernels import no_memory_weights, single_exponential_weights  # noqa: E402

RESULTS = ROOT / "results/eeg"
EPSILON = 1.0e-4
N_STEPS = 256
U0 = 0.0
SEED = 1
TOL_RRMSE = 0.01
TOL_FREQUENCY = 0.01
TOL_MEMORY_L1 = 0.02


def normalize(rho: np.ndarray) -> np.ndarray:
    out = 0.5 * (rho + rho.conj().T)
    return out / np.trace(out).real


def fixed_state(channel, dimension: int) -> tuple[np.ndarray, int, float]:
    rho = pure_zero_state(dimension)
    error = float("inf")
    for iteration in range(1, 5001):
        updated = normalize(batched_channel_step(channel, np.array([U0]), rho[None])[0])
        error = float(np.linalg.norm(updated - rho))
        rho = updated
        if error < 1e-13:
            return rho, iteration, error
    raise RuntimeError(f"constant-input fixed state did not converge: final error={error}")


def feature_map(delta_rho: np.ndarray, observables: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("ij,kji->k", delta_rho, observables))


def input_derivative(channel, rho_star: np.ndarray) -> np.ndarray:
    plus = normalize(batched_channel_step(channel, np.array([U0 + EPSILON]), rho_star[None])[0])
    minus = normalize(batched_channel_step(channel, np.array([U0 - EPSILON]), rho_star[None])[0])
    return (plus - minus) / (2.0 * EPSILON)


def tangent_response(kernel, channel, rho_star, observables, signal: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(rho_star)
    buffer = np.zeros((kernel.K + 1, *rho_star.shape), dtype=np.complex128)
    b = input_derivative(channel, rho_star)
    output = np.empty((len(signal), len(observables)), dtype=float)
    for step, value in enumerate(signal):
        mixed = kernel.present * delta
        for lag, weight in enumerate(kernel.delayed, start=1):
            mixed = mixed + weight * buffer[-1 - lag]
        # At fixed input the CPTP channel is linear in rho. Trace-zero tangent
        # perturbations therefore propagate without a normalization correction.
        propagated = batched_channel_step(channel, np.array([U0]), mixed[None])[0]
        delta = propagated + b * value
        buffer = np.concatenate([buffer[1:], delta[None]], axis=0)
        output[step] = feature_map(delta, observables)
    return output


def nonlinear_response(kernel, channel, rho_star, signal: np.ndarray) -> np.ndarray:
    baseline = np.full((1, len(signal)), U0)
    perturbed = baseline + EPSILON * signal[None]
    base_features = run_batched_reservoir(kernel, channel, rho_star, baseline, check_every=N_STEPS).features[0]
    pert_features = run_batched_reservoir(kernel, channel, rho_star, perturbed, check_every=N_STEPS).features[0]
    return (pert_features - base_features) / EPSILON


def separable_response(base_impulse: np.ndarray, kernel) -> np.ndarray:
    weights = np.concatenate([[kernel.present], kernel.delayed])
    out = np.zeros_like(base_impulse)
    for feature in range(base_impulse.shape[1]):
        out[:, feature] = np.convolve(base_impulse[:, feature], weights, mode="full")[: len(base_impulse)]
    return out


def relative_error(predicted: np.ndarray, measured: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - measured) / max(np.linalg.norm(measured), np.finfo(float).eps))


def memory_distribution(response: np.ndarray) -> np.ndarray:
    energy = np.sum(response * response, axis=1)
    return energy / max(float(energy.sum()), np.finfo(float).eps)


def metrics(theory: str, impulse_pred, step_pred, impulse_measured, step_measured) -> list[dict]:
    frequency_pred = np.fft.fft(impulse_pred, axis=0)
    frequency_measured = np.fft.fft(impulse_measured, axis=0)
    values = {
        "impulse_relative_rmse": relative_error(impulse_pred, impulse_measured),
        "step_relative_rmse": relative_error(step_pred, step_measured),
        "frequency_relative_error": relative_error(frequency_pred, frequency_measured),
        "memory_function_l1": float(np.sum(np.abs(memory_distribution(impulse_pred) - memory_distribution(impulse_measured)))),
    }
    tolerances = {
        "impulse_relative_rmse": TOL_RRMSE,
        "step_relative_rmse": TOL_RRMSE,
        "frequency_relative_error": TOL_FREQUENCY,
        "memory_function_l1": TOL_MEMORY_L1,
    }
    return [{"theory": theory, "check": key, "value": value, "tolerance": tolerances[key], "passed": value <= tolerances[key]} for key, value in values.items()]


def symbolic_summary(kernel) -> str:
    z, r, m, a = sp.symbols("z r m a", nonzero=True)
    K = int(kernel.K)
    normalization_sum = sum(r**tau for tau in range(1, K + 1))
    delayed_sum = sum((r / z) ** tau for tau in range(1, K + 1))
    normalization_closed = r * (1 - r**K) / (1 - r)
    delayed_closed = (r / z) * (1 - (r / z) ** K) / (1 - r / z)
    teff_sum = sum(tau * r**tau for tau in range(1, K + 1)) / normalization_sum
    teff_closed = (1 - (K + 1) * r**K + K * r ** (K + 1)) / ((1 - r) * (1 - r**K))
    if sp.simplify(normalization_sum - normalization_closed) != 0:
        raise RuntimeError("SymPy failed normalization closed-form identity")
    if sp.simplify(delayed_sum - delayed_closed) != 0:
        raise RuntimeError("SymPy failed delayed-kernel closed-form identity")
    if sp.simplify(teff_sum - teff_closed) != 0:
        raise RuntimeError("SymPy failed T_eff closed-form identity")
    W_abstract = sp.symbols("W", nonzero=True)
    actual = 1 / (z - a * W_abstract)
    separable = W_abstract / (z - a)
    difference = sp.factor(sp.together(actual - separable))
    difference_is_zero = difference == 0
    return "\n".join([
        f"K = {K}",
        "S_K(r) = r*(1-r**K)/(1-r)",
        "D_K(z,r) = (r/z)*(1-(r/z)**K)/(1-r/z)",
        "W_K(z) = (1-m) + m*D_K(z,r)/S_K(r)",
        "T_eff(K,r) = (1-(K+1)*r**K+K*r**(K+1))/((1-r)*(1-r**K))",
        "scalar H_actual(z) = 1/(z-a*W_K(z))",
        "scalar H_separable(z) = W_K(z)/(z-a)",
        f"abstract difference = {sp.sstr(difference)}",
        f"SymPy proves H_actual == H_separable generically: {difference_is_zero}",
        f"numeric T_eff = {sum((i + 1) * w for i, w in enumerate(kernel.delayed)) / kernel.past_mass:.12g}",
    ]) + "\n"


def main() -> None:
    hp = json.loads((RESULTS / "hp_selected.json").read_text())["single_kernel"]["hp"]
    kernel = single_exponential_weights(**hp)
    base_kernel = no_memory_weights()
    channel = build_input_channel(n_qubits=4, seed=SEED)
    _, observable_list = local_pauli_observables(4)
    observables = np.asarray(observable_list)
    rho_star, iterations, fixed_error = fixed_state(channel, 16)

    impulse = np.zeros(N_STEPS); impulse[0] = 1.0
    step = np.ones(N_STEPS)
    measured_impulse = nonlinear_response(kernel, channel, rho_star, impulse)
    measured_step = nonlinear_response(kernel, channel, rho_star, step)
    tangent_impulse = tangent_response(kernel, channel, rho_star, observables, impulse)
    tangent_step = tangent_response(kernel, channel, rho_star, observables, step)
    base_impulse = tangent_response(base_kernel, channel, rho_star, observables, impulse)
    separable_impulse = separable_response(base_impulse, kernel)
    separable_step = np.cumsum(separable_impulse, axis=0)

    rows = metrics("tangent_recurrence", tangent_impulse, tangent_step, measured_impulse, measured_step)
    rows += metrics("separable_W_times_R", separable_impulse, separable_step, measured_impulse, measured_step)
    tangent_pass = all(row["passed"] for row in rows if row["theory"] == "tangent_recurrence")
    separable_pass = all(row["passed"] for row in rows if row["theory"] == "separable_W_times_R")
    verdict = "PASS" if tangent_pass and separable_pass else "FAIL_SEPARABLE_FACTORIZATION" if tangent_pass else "FAIL_LINEARIZATION"
    for row in rows:
        row.update({
            "overall_verdict": verdict, "epsilon": EPSILON, "n_steps": N_STEPS, "seed": SEED,
            "fixed_point_iterations": iterations, "fixed_point_error": fixed_error,
            "K": kernel.K, "r": hp["r"], "past_mass": hp["past_mass"],
        })
    pd.DataFrame(rows).to_csv(RESULTS / "theory_vs_sim_check.csv", index=False)
    (RESULTS / "effective_kernel_symbolic.txt").write_text(symbolic_summary(kernel), encoding="utf-8")
    np.savez_compressed(
        RESULTS / "theory_vs_sim_responses.npz", measured_impulse=measured_impulse,
        tangent_impulse=tangent_impulse, separable_impulse=separable_impulse,
        measured_step=measured_step, tangent_step=tangent_step, separable_step=separable_step,
    )
    print(pd.DataFrame(rows)[["theory", "check", "value", "tolerance", "passed"]].to_string(index=False))
    print(f"STAGE 1 THEORY CHECK VERDICT: {verdict}")
    print(f"fixed point: iterations={iterations}, error={fixed_error:.3e}")


if __name__ == "__main__":
    main()
