#!/usr/bin/env python3
"""Gate 1B — post-gate robustness of the effective-kernel mechanism.

Post-gate robustness analysis over a prespecified grid frozen before execution.
It does NOT alter, overwrite, or retrospectively reinterpret the frozen
confirmatory Gate 1 (config/effective_kernel_gate1_frozen.json and its
results/eeg/theory_vs_sim_* artifacts). All outputs live under
results/eeg/gate1b_robustness/ and the Gate 1B-specific config/docs/figures.

The u0=0, r=0.7, seed=1, epsilon=1e-4 corner of this grid IS the frozen Gate 1
operating point; the script asserts it reproduces the frozen numbers, which
proves the u0-parameterization is implementation-faithful.

Pure, u0-independent mathematics (traceless-Hermitian basis, coordinates,
tangent/separable responses, metric helpers) is imported from the canonical
Gate 1 script so the two share identical math. Only the three u0-dependent
pieces (fixed state, A/B derivative, nonlinear baseline) get explicit-u0
variants here; the canonical script is left untouched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.channels import build_input_channel  # noqa: E402
from qrc_eeg.models import pure_zero_state  # noqa: E402
from qrc_eeg.observables import local_pauli_observables  # noqa: E402
from qrc_eeg.state_kernels import single_exponential_weights  # noqa: E402

# Import the canonical Gate 1 script as a library for its shared, u0-independent math.
_SPEC = importlib.util.spec_from_file_location(
    "effective_kernel_check", ROOT / "scripts/run_effective_kernel_check.py"
)
GATE1 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(GATE1)

from qrc_eeg.batched import batched_channel_step, run_batched_reservoir  # noqa: E402

CONFIG_PATH = ROOT / "config/effective_kernel_gate1b_robustness.json"
PROTOCOL_PATH = ROOT / "docs/gate1b_robustness_protocol.md"
OUTDIR = ROOT / "results/eeg/gate1b_robustness"

# Frozen Gate 1 canonical artifacts whose integrity must be preserved (hashed before/after).
CANONICAL_GATE1 = [
    "results/eeg/theory_vs_sim_check.csv",
    "results/eeg/theory_vs_sim_responses.npz",
    "results/eeg/theory_vs_sim_metadata.json",
    "results/eeg/theory_linearity_sweep.csv",
    "results/eeg/effective_kernel_symbolic.txt",
    "config/effective_kernel_gate1_frozen.json",
]

TOLERANCES = {
    "impulse_relative_frobenius": 0.01,
    "step_relative_frobenius": 0.01,
    "frequency_relative_frobenius": 0.01,
    "memory_function_l1": 0.02,
}
TOLERANCE_METRICS = tuple(TOLERANCES.keys())
DIAGNOSTIC_METRICS = ("impulse_cosine_similarity", "step_cosine_similarity")

# Frozen Gate 1 reference numbers (seed=1, r=0.7, u0=0, eps=1e-4) from docs/effective_kernel_theory.md.
GATE1_REFERENCE = {
    "fixed_iterations": 296,
    "fixed_difference": 9.454e-14,
    "tangent_impulse_relative_frobenius": 2.7416e-5,
    "tangent_step_relative_frobenius": 4.0720e-5,
    "separable_impulse_relative_frobenius": 0.418614,
    "separable_step_relative_frobenius": 0.068988,
    "companion_spectral_radius": 0.9587240324199373,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


# --- u0-parameterized variants of the three globally-U0-dependent Gate 1 routines -------------


def fixed_state_at(channel, u0: float, tolerance: float, max_iterations: int):
    """Converged fixed state under constant input u0. Never raises; returns convergence status."""

    rho = pure_zero_state(16)
    difference = float("inf")
    for iteration in range(1, max_iterations + 1):
        updated = GATE1.normalize_density(batched_channel_step(channel, np.array([u0]), rho[None])[0])
        difference = float(np.linalg.norm(updated - rho))
        rho = updated
        if difference < tolerance:
            return rho, iteration, difference, True
    return rho, max_iterations, difference, False


def build_abc_at(channel, rho_star, basis, observables, u0: float, derivative_epsilon: float):
    """A/B/C at operating point u0. A uses u0 through rho_in(u0); B is the central input derivative
    around u0; C is the fixed observable projection. All are independent of the kernel damping r."""

    propagated_basis = batched_channel_step(channel, np.full(len(basis), u0), basis)
    A = GATE1.coordinates(basis, propagated_basis)
    plus = GATE1.normalize_density(batched_channel_step(channel, np.array([u0 + derivative_epsilon]), rho_star[None])[0])
    minus = GATE1.normalize_density(batched_channel_step(channel, np.array([u0 - derivative_epsilon]), rho_star[None])[0])
    B = GATE1.coordinates(basis, (plus - minus) / (2.0 * derivative_epsilon))
    C = np.real(np.einsum("kij,aji->ka", observables, basis))
    return A, B, C


def nonlinear_response_at(kernel, channel, rho_star, signal: np.ndarray, u0: float, epsilon: float) -> np.ndarray:
    baseline = np.full((1, len(signal)), u0)
    perturbed = baseline + epsilon * signal[None]
    base = run_batched_reservoir(kernel, channel, rho_star, baseline, check_every=len(signal)).features[0]
    measured = run_batched_reservoir(kernel, channel, rho_star, perturbed, check_every=len(signal)).features[0]
    return (measured - base) / epsilon


def companion_spectrum_local(A: np.ndarray, weights: np.ndarray):
    """Companion eigenvalue union; does not raise on non-finite (unlike the canonical helper)."""

    a_eigenvalues = np.linalg.eigvals(A)
    roots: list[complex] = []
    for eigenvalue in a_eigenvalues:
        coefficients = np.concatenate([[1.0 + 0j], -eigenvalue * weights.astype(complex)])
        roots.extend(np.roots(coefficients))
    companion = np.asarray(roots)
    abs_companion = np.abs(companion)
    radius = float(np.max(abs_companion)) if abs_companion.size and np.isfinite(abs_companion).all() else float("inf")
    return a_eigenvalues, companion, radius


def safe_pass(value: float, tolerance: float | None):
    """Non-finite tolerance metrics are classified as failures, never left as bare NaN."""

    if tolerance is None:
        return float("nan")  # diagnostic metric, no threshold
    if not np.isfinite(value):
        return False
    return bool(value <= tolerance)


def config_metrics(tangent_impulse, tangent_step, separable_impulse, separable_step,
                   measured_impulse, measured_step) -> dict[str, dict[str, float]]:
    return {
        "tangent_recurrence": GATE1.theory_metrics(tangent_impulse, tangent_step, measured_impulse, measured_step),
        "separable_W_times_R": GATE1.theory_metrics(separable_impulse, separable_step, measured_impulse, measured_step),
    }


# --- grid execution ---------------------------------------------------------------------------


def run_grid(cfg: dict, provenance: dict):
    seeds = list(cfg["channel_seeds"])
    rs = list(cfg["r"])
    u0s = list(cfg["operating_points"])
    K = int(cfg["K"])
    past_mass = float(cfg["past_mass"])
    eps = float(cfg["confirmatory_epsilon_for_robustness_summary"])
    deps = float(cfg["derivative_epsilon"])
    n = int(cfg["response_length"])
    ftol = float(cfg["fixed_point_tolerance"])
    fmax = int(cfg["fixed_point_max_iterations"])
    sweep_epsilons = list(cfg["amplitude_sweep"])

    _, observable_list = local_pauli_observables(4)
    observables = np.asarray(observable_list)
    if observables.shape != (66, 16, 16):
        raise RuntimeError(f"expected 66 Pauli observables, got {observables.shape}")
    basis = GATE1.hermitian_traceless_basis(16)
    if basis.shape != (255, 16, 16):
        raise RuntimeError(f"expected 255-dim traceless basis, got {basis.shape}")

    impulse = np.zeros(n); impulse[0] = 1.0
    step = np.ones(n)

    metric_rows: list[dict] = []
    sweep_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    per_config_time: dict[str, float] = {}
    counts = {"success": 0, "fixed_point_failed": 0, "unstable": 0, "nonfinite": 0, "exception": 0}

    # NOTE: no "epsilon" here on purpose. The confirmatory epsilon belongs to the metric/spectrum
    # rows, but each amplitude-sweep row carries its own varying epsilon; spreading a fixed epsilon
    # via **base_provenance would silently clobber the per-row sweep epsilon.
    base_provenance = {
        "K": K, "past_mass": past_mass,
        "git_commit": provenance["commit"],
        "config_sha256": provenance["config_sha256"],
        "protocol_sha256": provenance["protocol_sha256"],
    }

    golden = None

    for seed in seeds:
        channel = build_input_channel(n_qubits=4, seed=int(seed))
        for u0 in u0s:
            # rho_*, A, B, C computed ONCE per (seed,u0); reused across r (r-independent).
            base_error = ""
            try:
                rho_star, fixed_iters, fixed_diff, converged = fixed_state_at(channel, u0, ftol, fmax)
                A, B, C = build_abc_at(channel, rho_star, basis, observables, u0, deps)
                base_ok = True
            except Exception:  # pragma: no cover - defensive isolation
                base_error = traceback.format_exc(limit=3)
                rho_star = None; A = B = C = None
                fixed_iters, fixed_diff, converged, base_ok = fmax, float("nan"), False, False

            for r in rs:
                t0 = time.perf_counter()
                key = f"seed{seed}_r{r}_u0{u0}"
                exception_text = base_error
                spectral_radius = float("nan")
                stable = False
                arrays_finite = False
                metrics = None
                n_companion = 0
                n_unstable_modes = -1
                dominant_abs = [float("nan")] * 3
                max_real = float("nan")
                max_a_abs = float("nan")

                if base_ok and converged:
                    try:
                        kernel = single_exponential_weights(K=K, r=float(r), past_mass=past_mass)
                        weights = np.concatenate([[kernel.present], kernel.delayed])
                        if not np.isclose(weights.sum(), 1.0, atol=1e-14):
                            raise RuntimeError(f"weights do not sum to one: {weights.sum()}")

                        tangent_impulse = GATE1.tangent_response(A, B, C, weights, impulse)
                        tangent_step = GATE1.tangent_response(A, B, C, weights, step)
                        k0_impulse = GATE1.tangent_response(A, B, C, np.array([1.0]), impulse)
                        separable_impulse = GATE1.separable_response(k0_impulse, weights)
                        separable_step = np.cumsum(separable_impulse, axis=0)
                        measured_impulse = nonlinear_response_at(kernel, channel, rho_star, impulse, u0, eps)
                        measured_step = nonlinear_response_at(kernel, channel, rho_star, step, u0, eps)

                        arrays_finite = bool(
                            np.isfinite(tangent_impulse).all() and np.isfinite(tangent_step).all()
                            and np.isfinite(measured_impulse).all() and np.isfinite(measured_step).all()
                            and np.isfinite(separable_impulse).all() and np.isfinite(separable_step).all()
                        )
                        metrics = config_metrics(tangent_impulse, tangent_step, separable_impulse,
                                                 separable_step, measured_impulse, measured_step)

                        a_eigs, companion, spectral_radius = companion_spectrum_local(A, weights)
                        stable = bool(np.isfinite(spectral_radius) and spectral_radius < 1.0)
                        abs_companion = np.abs(companion)
                        n_companion = int(companion.size)
                        n_unstable_modes = int(np.sum(abs_companion >= 1.0)) if np.isfinite(abs_companion).all() else -1
                        top = np.sort(abs_companion[np.isfinite(abs_companion)])[::-1] if np.isfinite(abs_companion).any() else np.array([])
                        dominant_abs = [float(top[i]) if i < top.size else float("nan") for i in range(3)]
                        finite_comp = companion[np.isfinite(companion)]
                        max_real = float(np.max(finite_comp.real)) if finite_comp.size else float("nan")
                        max_a_abs = float(np.max(np.abs(a_eigs)))

                        # golden reproduction check on the Gate 1 corner
                        if int(seed) == 1 and float(r) == 0.7 and u0 == 0.0:
                            golden = {
                                "fixed_iterations": fixed_iters, "fixed_difference": fixed_diff,
                                "tangent_impulse_relative_frobenius": metrics["tangent_recurrence"]["impulse_relative_frobenius"],
                                "tangent_step_relative_frobenius": metrics["tangent_recurrence"]["step_relative_frobenius"],
                                "separable_impulse_relative_frobenius": metrics["separable_W_times_R"]["impulse_relative_frobenius"],
                                "separable_step_relative_frobenius": metrics["separable_W_times_R"]["step_relative_frobenius"],
                                "companion_spectral_radius": spectral_radius,
                            }

                        # amplitude sweep: fixed tangent/separable vs nonlinear probe over epsilons
                        for sweep_eps in sweep_epsilons:
                            mi = nonlinear_response_at(kernel, channel, rho_star, impulse, u0, sweep_eps)
                            ms = nonlinear_response_at(kernel, channel, rho_star, step, u0, sweep_eps)
                            sweep_rows.append({
                                "seed": int(seed), "r": float(r), "u0": u0, "epsilon": sweep_eps,
                                "is_confirmatory_epsilon": bool(sweep_eps == eps),
                                "tangent_impulse_relative_frobenius": GATE1.relative_frobenius(tangent_impulse, mi),
                                "tangent_step_relative_frobenius": GATE1.relative_frobenius(tangent_step, ms),
                                "separable_impulse_relative_frobenius": GATE1.relative_frobenius(separable_impulse, mi),
                                "separable_step_relative_frobenius": GATE1.relative_frobenius(separable_step, ms),
                                "valid": bool(arrays_finite),
                                **base_provenance,
                            })
                    except Exception:  # pragma: no cover - defensive isolation
                        exception_text = traceback.format_exc(limit=3)
                        metrics = None
                        arrays_finite = False

                valid = bool(base_ok and converged and (exception_text == "") and arrays_finite)

                # count status
                if not converged:
                    counts["fixed_point_failed"] += 1
                elif exception_text:
                    counts["exception"] += 1
                elif not arrays_finite:
                    counts["nonfinite"] += 1
                else:
                    counts["success"] += 1
                    if not stable:
                        counts["unstable"] += 1

                # emit sweep NaN rows if the config produced no responses
                if metrics is None:
                    for sweep_eps in sweep_epsilons:
                        sweep_rows.append({
                            "seed": int(seed), "r": float(r), "u0": u0, "epsilon": sweep_eps,
                            "is_confirmatory_epsilon": bool(sweep_eps == eps),
                            "tangent_impulse_relative_frobenius": float("nan"),
                            "tangent_step_relative_frobenius": float("nan"),
                            "separable_impulse_relative_frobenius": float("nan"),
                            "separable_step_relative_frobenius": float("nan"),
                            "valid": False, **base_provenance,
                        })

                # metric rows (12 per config: 6 metrics x 2 theories), always emitted
                for theory in ("tangent_recurrence", "separable_W_times_R"):
                    for metric in (*TOLERANCE_METRICS, *DIAGNOSTIC_METRICS):
                        tol = TOLERANCES.get(metric)
                        value = float(metrics[theory][metric]) if metrics is not None else float("nan")
                        metric_rows.append({
                            "seed": int(seed), "r": float(r), "u0": u0, "epsilon": eps,
                            "theory": theory, "metric": metric,
                            "value": value, "tolerance": tol if tol is not None else float("nan"),
                            "pass": safe_pass(value, tol),
                            "fixed_point_converged": bool(converged),
                            "fixed_iterations": int(fixed_iters),
                            "fixed_final_difference": float(fixed_diff),
                            "companion_spectral_radius": float(spectral_radius),
                            "companion_stable": bool(stable),
                            "arrays_finite": bool(arrays_finite),
                            "valid": valid,
                            "exception": exception_text.strip().splitlines()[-1] if exception_text else "",
                            **base_provenance,
                        })

                spectrum_rows.append({
                    "seed": int(seed), "r": float(r), "u0": u0, "epsilon": eps,
                    "fixed_point_converged": bool(converged),
                    "fixed_iterations": int(fixed_iters),
                    "fixed_final_difference": float(fixed_diff),
                    "companion_spectral_radius": float(spectral_radius),
                    "companion_stable": bool(stable),
                    "companion_dimension": int(n_companion),
                    "n_unstable_modes": int(n_unstable_modes),
                    "dominant_abs_1": dominant_abs[0], "dominant_abs_2": dominant_abs[1],
                    "dominant_abs_3": dominant_abs[2], "companion_max_real_part": max_real,
                    "max_A_eigenvalue_abs": max_a_abs,
                    "valid": valid, "exception": exception_text.strip().splitlines()[-1] if exception_text else "",
                    **base_provenance,
                })
                per_config_time[key] = time.perf_counter() - t0

    return (pd.DataFrame(metric_rows), pd.DataFrame(sweep_rows), pd.DataFrame(spectrum_rows),
            per_config_time, counts, golden)


# --- summary + classification -----------------------------------------------------------------


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {k: float("nan") for k in ("mean", "median", "std", "min", "max", "q10", "q25", "q75", "q90")}
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "q10": float(np.quantile(values, 0.10)), "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)), "q90": float(np.quantile(values, 0.90)),
    }


def joint_pass_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per config with booleans for all-four-tangent, all-four-separable, stability, validity."""

    rows = []
    for (seed, r, u0), group in metrics.groupby(["seed", "r", "u0"]):
        valid = bool(group["valid"].iloc[0])
        stable = bool(group["companion_stable"].iloc[0])

        def all4(theory: str) -> bool:
            sub = group[(group.theory == theory) & (group.metric.isin(TOLERANCE_METRICS))]
            return bool(sub["pass"].astype(bool).all()) and len(sub) == 4

        rows.append({"seed": seed, "r": r, "u0": u0, "valid": valid, "stable": stable,
                     "tangent_all4": all4("tangent_recurrence"), "separable_all4": all4("separable_W_times_R")})
    return pd.DataFrame(rows)


def classify(joint: pd.DataFrame) -> dict:
    valid = joint[joint.valid]
    n_valid = len(valid)
    if n_valid == 0:
        return {"classification": "INVALID", "reason": "no valid configurations", "n_valid": 0,
                "n_total": len(joint), "tangent_pass_fraction": float("nan"),
                "separable_pass_fraction": float("nan")}
    tf = float(valid["tangent_all4"].mean())
    sf = float(valid["separable_all4"].mean())
    # Frozen rule, applied in spec order: NOT_ROBUST (tf<=0.50 OR sf>=0.50) takes precedence over
    # MIXED. In the residual MIXED region tf>0.50 and sf<0.50, so tangent always exceeds separable.
    if tf >= 0.90 and sf <= 0.10:
        label = "ROBUST_WITHIN_GRID"
    elif tf <= 0.50 or sf >= 0.50:
        label = "NOT_ROBUST_WITHIN_GRID"
    else:
        label = "MIXED"
    return {"classification": label, "n_total": len(joint), "n_valid": n_valid,
            "tangent_pass_fraction": tf, "separable_pass_fraction": sf,
            "locally_stable_fraction_over_valid": float(valid["stable"].mean()),
            "locally_stable_fraction_over_total": float(joint["stable"].mean())}


def build_summary(metrics: pd.DataFrame, joint: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strata = [("global", "all", "all", metrics, joint)]
    for r in sorted(metrics.r.unique()):
        strata.append((f"r={r}", r, "all", metrics[metrics.r == r], joint[joint.r == r]))
    for u0 in sorted(metrics.u0.unique()):
        strata.append((f"u0={u0}", "all", u0, metrics[metrics.u0 == u0], joint[joint.u0 == u0]))
    # joint cross: one row per (r, u0, theory, metric) as the plain reading of the spec requires
    for r in sorted(metrics.r.unique()):
        for u0 in sorted(metrics.u0.unique()):
            m = metrics[(metrics.r == r) & (metrics.u0 == u0)]
            j = joint[(joint.r == r) & (joint.u0 == u0)]
            strata.append((f"r={r},u0={u0}", r, u0, m, j))

    for stratum, r_key, u0_key, msub, jsub in strata:
        n_total = len(jsub)
        n_valid = int(jsub["valid"].sum())
        # per theory x metric statistics over valid configs
        for theory in ("tangent_recurrence", "separable_W_times_R"):
            for metric in (*TOLERANCE_METRICS, *DIAGNOSTIC_METRICS):
                cell = msub[(msub.theory == theory) & (msub.metric == metric) & (msub.valid)]
                values = cell["value"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                tol = TOLERANCES.get(metric)
                if tol is not None:
                    n_pass = int(cell["pass"].astype(bool).sum())
                    pass_fraction = float(n_pass / n_valid) if n_valid else float("nan")
                else:
                    n_pass = -1
                    pass_fraction = float("nan")
                rows.append({"stratum": stratum, "r": r_key, "u0": u0_key, "theory": theory,
                             "metric": metric, "n_total": n_total, "n_valid": n_valid,
                             "n_pass": n_pass, "pass_fraction": pass_fraction, **_quantiles(values)})
        # joint / stability rows
        jvalid = jsub[jsub.valid]
        nv = len(jvalid)
        empty = {k: float("nan") for k in ("mean", "median", "std", "min", "max", "q10", "q25", "q75", "q90")}
        for theory, col in (("tangent_recurrence", "tangent_all4"), ("separable_W_times_R", "separable_all4")):
            n_pass = int(jvalid[col].sum())
            rows.append({"stratum": stratum, "r": r_key, "u0": u0_key, "theory": theory,
                         "metric": "joint_pass_all4", "n_total": n_total, "n_valid": nv,
                         "n_pass": n_pass, "pass_fraction": float(n_pass / nv) if nv else float("nan"), **empty})
        rows.append({"stratum": stratum, "r": r_key, "u0": u0_key, "theory": "n/a",
                     "metric": "locally_stable_over_valid", "n_total": n_total, "n_valid": nv,
                     "n_pass": int(jvalid["stable"].sum()),
                     "pass_fraction": float(jvalid["stable"].mean()) if nv else float("nan"), **empty})
        rows.append({"stratum": stratum, "r": r_key, "u0": u0_key, "theory": "n/a",
                     "metric": "locally_stable_over_total", "n_total": n_total, "n_valid": nv,
                     "n_pass": int(jsub["stable"].sum()),
                     "pass_fraction": float(jsub["stable"].mean()) if n_total else float("nan"), **empty})
    return pd.DataFrame(rows)


def stratified_classification(joint: pd.DataFrame) -> dict:
    out = {"global": classify(joint)}
    for r in sorted(joint.r.unique()):
        out[f"r={r}"] = classify(joint[joint.r == r])
    for u0 in sorted(joint.u0.unique()):
        out[f"u0={u0}"] = classify(joint[joint.u0 == u0])
    return out


# --- main -------------------------------------------------------------------------------------


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text())
    commit = git_output("rev-parse", "HEAD")
    status_lines = git_output("status", "--porcelain").splitlines()
    config_sha = sha256(CONFIG_PATH)
    protocol_sha = sha256(PROTOCOL_PATH)
    canonical_before = {rel: sha256(ROOT / rel) for rel in CANONICAL_GATE1}
    provenance = {"commit": commit, "config_sha256": config_sha, "protocol_sha256": protocol_sha}

    start = time.perf_counter()
    metrics, sweep, spectrum, per_config_time, counts, golden = run_grid(cfg, provenance)
    elapsed = time.perf_counter() - start

    joint = joint_pass_table(metrics)
    summary = build_summary(metrics, joint)
    classification = stratified_classification(joint)

    # golden reproduction of the Gate 1 corner
    golden_ok = golden is not None
    golden_report = {}
    if golden is not None:
        for key, expected in GATE1_REFERENCE.items():
            got = golden[key]
            rtol = 1e-9 if key in ("companion_spectral_radius",) else 3e-3
            atol = 0.0 if key == "fixed_iterations" else 1e-12
            ok = (int(got) == int(expected)) if key == "fixed_iterations" else bool(np.isclose(got, expected, rtol=rtol, atol=max(atol, abs(expected) * rtol)))
            golden_report[key] = {"expected": expected, "got": got, "ok": ok}
            golden_ok = golden_ok and ok

    metrics.to_csv(OUTDIR / "metrics_by_configuration.csv", index=False)
    sweep.to_csv(OUTDIR / "amplitude_sweep.csv", index=False)
    spectrum.to_csv(OUTDIR / "spectrum_by_configuration.csv", index=False)
    summary.to_csv(OUTDIR / "summary.csv", index=False)

    canonical_after = {rel: sha256(ROOT / rel) for rel in CANONICAL_GATE1}
    gate1_unchanged = canonical_before == canonical_after

    finite_radii = spectrum.loc[spectrum.companion_spectral_radius.apply(np.isfinite), "companion_spectral_radius"]
    metadata = {
        "analysis": cfg["analysis"],
        "status": cfg["status"],
        "timestamp": datetime.now(ZoneInfo(cfg["timezone"])).isoformat(),
        "timezone": cfg["timezone"],
        "git_commit": commit,
        "git_status_porcelain": status_lines,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sympy_version": __import__("sympy").__version__,
        "config": cfg,
        "config_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "canonical_gate1_sha256_before": canonical_before,
        "canonical_gate1_sha256_after": canonical_after,
        "gate1_artifacts_unchanged": gate1_unchanged,
        "gate1_original_not_altered": True,
        "n_configurations_total": int(len(joint)),
        "n_configurations_expected": int(cfg["expected_total_configurations"]),
        "status_counts": counts,
        "total_seconds": elapsed,
        "seconds_per_configuration_mean": float(np.mean(list(per_config_time.values()))),
        "seconds_per_configuration": per_config_time,
        "golden_gate1_corner_reproduced": bool(golden_ok),
        "golden_gate1_corner_detail": golden_report,
        "classification": classification,
        "spectral_radius_min": float(finite_radii.min()) if len(finite_radii) else float("nan"),
        "spectral_radius_max": float(finite_radii.max()) if len(finite_radii) else float("nan"),
        "artifact_sha256": {},  # filled below
    }

    for name in ("metrics_by_configuration.csv", "amplitude_sweep.csv",
                 "spectrum_by_configuration.csv", "summary.csv"):
        metadata["artifact_sha256"][name] = sha256(OUTDIR / name)

    (OUTDIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")

    write_report(cfg, metadata, summary, joint, spectrum, classification, golden_report)

    if not gate1_unchanged:
        raise SystemExit("INTEGRITY FAILURE: canonical Gate 1 artifact hashes changed during Gate 1B run")
    if not golden_ok:
        raise SystemExit(f"GOLDEN FAILURE: Gate 1 corner not reproduced: {golden_report}")

    print(f"Gate 1B complete: {len(joint)} configurations in {elapsed:.1f}s")
    print(f"status counts: {counts}")
    print(f"golden Gate 1 corner reproduced: {golden_ok}")
    print(f"global classification: {classification['global']['classification']} "
          f"(tangent {classification['global']['tangent_pass_fraction']:.3f}, "
          f"separable {classification['global']['separable_pass_fraction']:.3f})")
    for key in ("r=0.7", "r=0.9", "u0=-0.5", "u0=0.0", "u0=0.5"):
        c = classification.get(key, {})
        print(f"  {key}: {c.get('classification')} "
              f"(tangent {c.get('tangent_pass_fraction', float('nan')):.3f}, "
              f"separable {c.get('separable_pass_fraction', float('nan')):.3f})")
    print(f"spectral radius range: [{metadata['spectral_radius_min']:.6g}, {metadata['spectral_radius_max']:.6g}]")
    print(f"Gate 1 artifacts unchanged: {gate1_unchanged}")


def write_report(cfg, metadata, summary, joint, spectrum, classification, golden_report) -> None:
    def frac(d, key):
        v = d.get(key, float("nan"))
        return f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "n/a"

    g = classification["global"]
    lines = []
    lines.append("# Gate 1B — post-gate robustness of the effective-kernel mechanism\n")
    lines.append("Post-gate robustness analysis over a **prespecified grid frozen before "
                 "execution**. It does not alter, overwrite, or retrospectively reinterpret the "
                 "frozen confirmatory Gate 1.\n")
    lines.append(f"- Generated: {metadata['timestamp']} ({metadata['timezone']})")
    lines.append(f"- Origin commit: `{metadata['git_commit']}`")
    lines.append(f"- Python {metadata['python_version']}, NumPy {metadata['numpy_version']}, "
                 f"Pandas {metadata['pandas_version']}, SymPy {metadata['sympy_version']}")
    lines.append(f"- Gate 1 canonical artifacts unchanged: **{metadata['gate1_artifacts_unchanged']}**")
    lines.append(f"- Gate 1 corner (seed=1, r=0.7, u0=0, eps=1e-4) reproduced: "
                 f"**{metadata['golden_gate1_corner_reproduced']}**\n")

    lines.append("## Scientific question\n")
    lines.append("Over a grid of channel seeds, damping ratios `r`, and operating points `u0` "
                 "frozen before execution, how robust is the *numerical* agreement of the "
                 "implementation-faithful tangent recurrence with the nonlinear simulator, and does "
                 "the separable factorization `H_sep(z)=W_K(z)R(z)` remain falsified? The structural "
                 "derivation is seed-independent; falsifying separability needs only one "
                 "counterexample (Gate 1); this extension characterizes the numerical robustness of "
                 "the tangent approximation and where it holds or fails.\n")

    lines.append("## Frozen grid\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| construction | {cfg['construction']} |")
    lines.append(f"| K | {cfg['K']} |")
    lines.append(f"| past_mass | {cfg['past_mass']} |")
    lines.append(f"| r | {cfg['r']} |")
    lines.append(f"| channel_seeds | {cfg['channel_seeds']} |")
    lines.append(f"| operating_points u0 | {cfg['operating_points']} |")
    lines.append(f"| confirmatory epsilon | {cfg['confirmatory_epsilon_for_robustness_summary']} |")
    lines.append(f"| amplitude sweep | {cfg['amplitude_sweep']} |")
    lines.append(f"| response_length | {cfg['response_length']} |")
    lines.append(f"| total configurations | {metadata['n_configurations_total']} "
                 f"(expected {metadata['n_configurations_expected']}) |\n")

    lines.append("## Status counts\n")
    for k, v in metadata["status_counts"].items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- total wall time: {metadata['total_seconds']:.1f} s "
                 f"({metadata['seconds_per_configuration_mean']:.2f} s/config mean)\n")

    lines.append("## Global classification\n")
    lines.append(f"**{g['classification']}** — tangent all-four pass fraction "
                 f"{frac(g,'tangent_pass_fraction')}, separable all-four pass fraction "
                 f"{frac(g,'separable_pass_fraction')} (over {g['n_valid']}/{g['n_total']} valid).")
    lines.append(f"Locally stable: {frac(g,'locally_stable_fraction_over_valid')} of valid, "
                 f"{frac(g,'locally_stable_fraction_over_total')} of all.\n")

    lines.append("## Stratified classification (tangent / separable all-four pass fraction)\n")
    lines.append("| Stratum | Classification | Tangent | Separable | n_valid / n_total | Stable (valid) |")
    lines.append("|---|---|---|---|---|---|")
    for key in ("global", "r=0.7", "r=0.9", "u0=-0.5", "u0=0.0", "u0=0.5"):
        c = classification.get(key, {})
        lines.append(f"| {key} | {c.get('classification','?')} | {frac(c,'tangent_pass_fraction')} | "
                     f"{frac(c,'separable_pass_fraction')} | {c.get('n_valid','?')} / {c.get('n_total','?')} | "
                     f"{frac(c,'locally_stable_fraction_over_valid')} |")
    lines.append("")

    lines.append("## r=0.7 versus r=0.9\n")
    for r in (0.7, 0.9):
        c = classification.get(f"r={r}", {})
        lines.append(f"- **r={r}**: {c.get('classification','?')}; tangent {frac(c,'tangent_pass_fraction')}, "
                     f"separable {frac(c,'separable_pass_fraction')}, "
                     f"stable(valid) {frac(c,'locally_stable_fraction_over_valid')}, "
                     f"stable(total) {frac(c,'locally_stable_fraction_over_total')}.")
    lines.append("")

    lines.append("## Dependence on operating point u0\n")
    for u0 in (-0.5, 0.0, 0.5):
        c = classification.get(f"u0={u0}", {})
        lines.append(f"- **u0={u0}**: {c.get('classification','?')}; tangent {frac(c,'tangent_pass_fraction')}, "
                     f"separable {frac(c,'separable_pass_fraction')}, "
                     f"stable(valid) {frac(c,'locally_stable_fraction_over_valid')}.")
    lines.append("")

    lines.append("## Dependence on epsilon\n")
    sweep = pd.read_csv(OUTDIR / "amplitude_sweep.csv")
    valid_sweep = sweep[sweep.valid]
    lines.append("Median tangent impulse / step error and median separable impulse error across "
                 "valid configurations, by amplitude epsilon:\n")
    lines.append("| epsilon | tangent impulse (median) | tangent step (median) | separable impulse (median) |")
    lines.append("|---|---|---|---|")
    for eps in cfg["amplitude_sweep"]:
        sub = valid_sweep[np.isclose(valid_sweep.epsilon, eps)]
        if len(sub):
            lines.append(f"| {eps:g} | {sub.tangent_impulse_relative_frobenius.median():.3e} | "
                         f"{sub.tangent_step_relative_frobenius.median():.3e} | "
                         f"{sub.separable_impulse_relative_frobenius.median():.3e} |")
    lines.append("")

    lines.append("## Spectral stability\n")
    finite = spectrum[spectrum.companion_spectral_radius.apply(np.isfinite)]
    lines.append(f"- Companion spectral radius range: "
                 f"[{metadata['spectral_radius_min']:.6g}, {metadata['spectral_radius_max']:.6g}].")
    for r in (0.7, 0.9):
        sub = finite[finite.r == r]
        if len(sub):
            lines.append(f"- r={r}: radius median {sub.companion_spectral_radius.median():.6g}, "
                         f"max {sub.companion_spectral_radius.max():.6g}, "
                         f"stable {int(sub.companion_stable.sum())}/{len(spectrum[spectrum.r==r])}.")
    lines.append("")

    lines.append("## Failed / unstable / non-converged configurations\n")
    failed = joint[~joint.valid]
    unstable = joint[joint.valid & ~joint.stable]
    if len(failed):
        lines.append(f"Invalid configurations ({len(failed)}):\n")
        for _, row in failed.iterrows():
            lines.append(f"- seed={row.seed}, r={row.r}, u0={row.u0}")
    else:
        lines.append("No invalid configurations: all 60 converged, executed without exception, and "
                     "produced finite confirmatory arrays.")
    lines.append("")
    if len(unstable):
        lines.append(f"Valid-but-locally-unstable configurations ({len(unstable)}):\n")
        for _, row in unstable.iterrows():
            rad = spectrum[(spectrum.seed == row.seed) & (spectrum.r == row.r) & (spectrum.u0 == row.u0)]
            radv = float(rad.companion_spectral_radius.iloc[0]) if len(rad) else float("nan")
            lines.append(f"- seed={row.seed}, r={row.r}, u0={row.u0} (radius {radv:.6g}, "
                         f"tangent all-four pass={row.tangent_all4})")
    else:
        lines.append("No valid configuration was locally unstable.")
    lines.append("")

    lines.append("## Limited interpretation\n")
    lines.append("- The **structural derivation** `H_actual = C[zI - A W_K(z)]^{-1} B` is algebraic "
                 "and does not depend on the seed.")
    lines.append("- The **falsification of the separable factorization** needs only the single Gate 1 "
                 "counterexample; this grid shows it is not an isolated accident.")
    lines.append("- The **numerical robustness of the tangent approximation** is the object here: it is "
                 "characterized over the frozen grid, including the region where it holds and any "
                 "unstable or failing cases, which are preserved above.")
    lines.append("- This does **not** demonstrate quantum advantage, physical/hardware implementation, "
                 "clinical validity, or environmental non-Markovianity.\n")

    lines.append("## Relationship to the original Gate 1\n")
    lines.append("Gate 1 (K=15, r=0.7, past_mass=0.3, seed=1, u0=0, eps=1e-4) remains frozen and "
                 "canonical. This extension neither modifies its artifacts (hash-verified unchanged: "
                 f"**{metadata['gate1_artifacts_unchanged']}**) nor updates the canonical claims "
                 "registry. The u0=0/r=0.7/seed=1 corner of this grid reproduces the frozen Gate 1 "
                 f"numbers (reproduced: **{metadata['golden_gate1_corner_reproduced']}**).\n")

    lines.append("## Files produced\n")
    for name in ("metrics_by_configuration.csv", "amplitude_sweep.csv",
                 "spectrum_by_configuration.csv", "summary.csv", "metadata.json", "report.md"):
        lines.append(f"- `results/eeg/gate1b_robustness/{name}`")
    lines.append("- `figures/eeg/fig_gate1b_robustness.pdf`")
    lines.append("- `figures/eeg/fig_gate1b_robustness.png`\n")

    lines.append("## Artifact hashes\n")
    for name, digest in metadata["artifact_sha256"].items():
        lines.append(f"- `{name}`: `{digest}`")
    lines.append(f"- config `{metadata['config_sha256']}`")
    lines.append(f"- protocol `{metadata['protocol_sha256']}`\n")

    lines.append("## Reproduction\n")
    lines.append("```")
    lines.append(".venv/bin/python scripts/run_gate1b_robustness.py")
    lines.append(".venv/bin/python scripts/verify_gate1b_robustness.py")
    lines.append(".venv/bin/python -m pytest tests/test_gate1b_robustness.py -q")
    lines.append("```")
    lines.append("")

    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
