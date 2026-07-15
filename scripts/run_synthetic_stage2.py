#!/usr/bin/env python3
"""Rota A Stage 2: freeze H_actual predictions, then measure nonlinear synthetic forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qrc_eeg.batched import run_batched_reservoir  # noqa: E402
from qrc_eeg.channels import build_input_channel  # noqa: E402
from qrc_eeg.classical_baselines import evaluate_feature_model, select_ridge_blocked  # noqa: E402
from qrc_eeg.preprocessing import fit_training_scaler  # noqa: E402
from qrc_eeg.state_kernels import (  # noqa: E402
    matched_delay_weights, no_memory_weights, single_exponential_weights,
    triangular_weights, uniform_weights,
)
from run_effective_kernel_check import (  # noqa: E402
    build_abc, companion_spectrum, fixed_state, hermitian_traceless_basis, tangent_response,
)

CONFIG = ROOT / "config/rotaA_stage2_frozen.json"
PROTOCOL = ROOT / "docs/synthetic_stage2_protocol.md"
RESULTS = ROOT / "results/synth"
FIGURES = ROOT / "figures/synth"
HP_PATH = "results/eeg/hp_selected.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_configuration() -> tuple[dict, dict, str]:
    cfg = json.loads(CONFIG.read_text())
    commit = git("rev-parse", "HEAD")
    if commit != cfg["gate1_commit"]:
        raise RuntimeError(f"commit changed since freeze: {commit} != {cfg['gate1_commit']}")
    selected = json.loads(git("show", f"{commit}:{HP_PATH}"))
    required = {
        "QRC_K0": {},
        "AB_noaux": selected["AB_noaux"]["hp"],
        "single_kernel": selected["single_kernel"]["hp"],
        "triangular": selected["triangular"]["hp"],
        "uniform": selected["uniform"]["hp"],
    }
    if list(required) != cfg["models"]:
        raise RuntimeError("frozen model order/configuration changed")
    if required["single_kernel"] != {"K": 15, "r": 0.7, "past_mass": 0.3}:
        raise RuntimeError(f"official Gate 1 HP changed: {required['single_kernel']}")
    print(f"Stage 2 commit: {commit}")
    print(f"loaded committed model HP: {json.dumps(required, sort_keys=True)}")
    return cfg, required, commit


def kernel_for(model: str, hp: dict):
    if model == "QRC_K0":
        return no_memory_weights()
    if model == "AB_noaux":
        return matched_delay_weights(hp["tau"], hp["tau"], hp["delayed_mass"])
    if model == "single_kernel":
        return single_exponential_weights(**hp)
    if model == "triangular":
        return triangular_weights(**hp)
    if model == "uniform":
        return uniform_weights(**hp)
    raise ValueError(model)


def stable_rng(label: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def ar1(length: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    burn = 256
    noise = rng.normal(scale=np.sqrt(1 - phi * phi), size=length + burn)
    x = np.empty(length + burn)
    x[0] = rng.normal()
    for t in range(1, len(x)):
        x[t] = phi * x[t - 1] + noise[t]
    return x[burn:]


def ar2(length: int, rho: float, frequency: float, rng: np.random.Generator) -> np.ndarray:
    burn = 512
    a1 = 2 * rho * np.cos(2 * np.pi * frequency)
    a2 = -(rho**2)
    noise = rng.normal(size=length + burn)
    x = np.zeros(length + burn)
    x[:2] = rng.normal(size=2)
    for t in range(2, len(x)):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + noise[t]
    return x[burn:]


def colored(length: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    nfreq = length // 2 + 1
    frequency = np.fft.rfftfreq(length)
    amplitude = np.ones(nfreq)
    amplitude[1:] = frequency[1:] ** (-beta / 2)
    spectrum = (rng.normal(size=nfreq) + 1j * rng.normal(size=nfreq)) * amplitude
    spectrum[0] = 0
    if length % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    return np.fft.irfft(spectrum, n=length)


def phase_surrogate(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(values)
    phase = rng.uniform(0, 2 * np.pi, len(spectrum))
    phase[0] = 0 if spectrum[0].real >= 0 else np.pi
    if len(values) % 2 == 0:
        phase[-1] = 0 if spectrum[-1].real >= 0 else np.pi
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phase), n=len(values))


def generate_one(spec: dict, index: int, length: int) -> np.ndarray:
    rng = stable_rng(f"{spec['name']}:{index}")
    family = spec["family"]
    if family == "AR1":
        return ar1(length, spec["phi"], rng)
    if family == "AR2":
        return ar2(length, spec["rho"], spec["frequency"], rng)
    if family == "colored":
        return colored(length, spec["beta"], rng)
    if family == "higher_order":
        x = ar1(length, spec["phi"], rng)
        return x + spec["quadratic"] * (x * x - 1.0)
    if family == "phase_surrogate":
        source = {"name": "nonlinear_ar1_phi085", "family": "higher_order", "phi": 0.85, "quadratic": 0.35}
        x = generate_one(source, index, length)
        return phase_surrogate(x, stable_rng(f"phase:{spec['name']}:{index}"))
    raise ValueError(family)


def scenario_data(cfg: dict, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = cfg["split"]
    total = sum(sizes.values())
    raw = np.stack([generate_one(spec, i, cfg["segment_length"]) for i in range(total)])
    ntr, nv = sizes["train"], sizes["validation"]
    scaler = fit_training_scaler(raw[:ntr])
    scaled = scaler.transform(raw)
    return scaled[:ntr], scaled[ntr:ntr + nv], scaled[ntr + nv:]


def causal_convolution(inputs: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    length = inputs.shape[1]
    nfft = 1 << (length + len(impulse) - 2).bit_length()
    signal_fft = np.fft.rfft(inputs, n=nfft)[:, None, :]
    impulse_fft = np.fft.rfft(impulse.T, n=nfft, axis=1)[None, :, :]
    output = np.fft.irfft(signal_fft * impulse_fft, n=nfft, axis=2)[:, :, :length]
    return np.transpose(output, (0, 2, 1))


def evaluate_all(features: tuple[np.ndarray, np.ndarray, np.ndarray], segments, cfg, model, scenario, seed, source):
    train_f, val_f, test_f = features
    train, val, test = segments
    rows = []
    for horizon in cfg["horizons"]:
        fit = select_ridge_blocked(
            train_f, train, val_f, val, horizon, cfg["washout"], cfg["alpha_grid"]
        )
        metrics = evaluate_feature_model(test_f, test, horizon, cfg["washout"], fit.weights)
        for segment_index, value in enumerate(metrics["nrmse"]):
            rows.append({
                "source": source, "scenario": scenario, "model": model, "seed": seed,
                "test_segment": segment_index, "horizon": horizon, "nrmse": value,
                "selected_alpha": fit.alpha,
            })
    return rows


def slope(values: pd.DataFrame) -> float:
    curve = values.groupby("horizon", sort=True).nrmse.mean()
    return float(np.polyfit(np.log2(curve.index.to_numpy(float)), curve.to_numpy(), 1)[0])


def bootstrap_slope(values: pd.DataFrame, resamples: int, label: str) -> tuple[float, float]:
    matrix = values.pivot(index=["seed", "test_segment"], columns="horizon", values="nrmse")
    matrix = matrix.reindex(sorted(matrix.columns), axis=1).to_numpy()
    rng = stable_rng(f"bootstrap:{label}")
    sampled = matrix[rng.integers(0, len(matrix), size=(resamples, len(matrix)))].mean(axis=1)
    x = np.log2(np.asarray(sorted(values.horizon.unique()), dtype=float))
    centered_x = x - x.mean()
    estimates = ((sampled - sampled.mean(axis=1, keepdims=True)) * centered_x).sum(axis=1) / np.sum(centered_x**2)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def summarize(raw: pd.DataFrame, dynamics: pd.DataFrame, cfg: dict, prefix: str) -> pd.DataFrame:
    rows = []
    for (scenario, model), group in raw.groupby(["scenario", "model"], sort=True):
        estimate = slope(group)
        low, high = bootstrap_slope(group, cfg["bootstrap_resamples"], f"{prefix}:{scenario}:{model}")
        dyn = dynamics[dynamics.model == model]
        rows.append({
            "scenario": scenario, "model": model, f"{prefix}_slope": estimate,
            f"{prefix}_slope_ci_low": low, f"{prefix}_slope_ci_high": high,
            "conditional_T_eff": dyn.conditional_T_eff.mean(),
            "full_mean_lag": dyn.full_mean_lag.mean(),
            "companion_spectral_radius_mean": dyn.companion_spectral_radius.mean(),
        })
    out = pd.DataFrame(rows)
    out[f"{prefix}_rank"] = out.groupby("scenario")[f"{prefix}_slope"].rank(method="average")
    return out


def system_objects(cfg: dict, hp_map: dict):
    objects = {}
    dynamics = []
    for seed in cfg["seeds"]:
        channel = build_input_channel(n_qubits=4, seed=seed)
        rho_star, iterations, difference = fixed_state(channel, 16)
        basis = hermitian_traceless_basis(16)
        from qrc_eeg.observables import local_pauli_observables
        observables = local_pauli_observables(4)[1]
        A, B, C = build_abc(channel, rho_star, basis, observables, derivative_epsilon=1e-4)
        for model, hp in hp_map.items():
            kernel = kernel_for(model, hp)
            weights = np.r_[kernel.present, kernel.delayed]
            impulse_signal = np.zeros(cfg["impulse_length"]); impulse_signal[0] = 1
            impulse = tangent_response(A, B, C, weights, impulse_signal)
            _, eigenvalues, radius = companion_spectrum(A, weights)
            delayed_mass = kernel.past_mass
            conditional = 0.0 if delayed_mass == 0 else float(
                np.dot(np.arange(1, kernel.K + 1), kernel.delayed) / delayed_mass
            )
            full = float(np.dot(np.arange(1, kernel.K + 1), kernel.delayed))
            objects[(seed, model)] = (channel, rho_star, kernel, impulse)
            dynamics.append({
                "seed": seed, "model": model, "K": kernel.K, "conditional_T_eff": conditional,
                "full_mean_lag": full, "companion_spectral_radius": radius,
                "companion_stable": radius < 1, "fixed_iterations": iterations,
                "fixed_final_difference": difference, "companion_dimension": len(eigenvalues),
            })
    return objects, pd.DataFrame(dynamics)


def predict(cfg: dict, hp_map: dict, commit: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    objects, dynamics = system_objects(cfg, hp_map)
    raw_rows = []
    for spec in cfg["scenarios"]:
        segments = scenario_data(cfg, spec)
        all_segments = np.vstack(segments)
        cuts = (len(segments[0]), len(segments[0]) + len(segments[1]))
        for seed in cfg["seeds"]:
            for model in cfg["models"]:
                impulse = objects[(seed, model)][3]
                features_all = causal_convolution(all_segments, impulse)
                features = (features_all[:cuts[0]], features_all[cuts[0]:cuts[1]], features_all[cuts[1]:])
                raw_rows.extend(evaluate_all(features, segments, cfg, model, spec["name"], seed, "theory_H_actual"))
        print(f"theory predictions complete: {spec['name']}", flush=True)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(RESULTS / "theory_predictions_raw.csv", index=False)
    dynamics.to_csv(RESULTS / "stage2_model_dynamics.csv", index=False)
    summary = summarize(raw, dynamics, cfg, "predicted")
    summary["git_commit"] = commit
    prediction = RESULTS / "theory_predictions_frozen.csv"
    summary.to_csv(prediction, index=False)
    manifest = RESULTS / "stage2_predictions_frozen.sha256"
    manifest.write_text(f"{sha256(prediction)}  results/synth/theory_predictions_frozen.csv\n")
    print(f"froze {prediction}: {sha256(prediction)}")


def aggregate_statistics(combined: pd.DataFrame, cfg: dict) -> dict:
    rho = float(spearmanr(combined.predicted_slope, combined.measured_slope).statistic)
    scenario_rho = combined.groupby("scenario").apply(
        lambda frame: spearmanr(frame.predicted_slope, frame.measured_slope).statistic,
        include_groups=False,
    )
    scenarios = combined.scenario.unique()
    rng = stable_rng("aggregate-spearman-bootstrap")
    boot = []
    scenario_frames = {scenario: combined[combined.scenario == scenario] for scenario in scenarios}
    for _ in range(cfg["bootstrap_resamples"]):
        chosen = rng.choice(scenarios, len(scenarios), replace=True)
        predicted = np.concatenate([scenario_frames[scenario].predicted_slope.to_numpy() for scenario in chosen])
        measured = np.concatenate([scenario_frames[scenario].measured_slope.to_numpy() for scenario in chosen])
        boot.append(spearmanr(predicted, measured).statistic)
    low, high = np.nanquantile(boot, [0.025, 0.975])
    best = combined.loc[combined.groupby("scenario").predicted_slope.idxmin(), ["scenario", "model"]]
    measured = combined.loc[combined.groupby("scenario").measured_slope.idxmin(), ["scenario", "model"]]
    matches = best.merge(measured, on="scenario", suffixes=("_predicted", "_measured"))
    fraction = float((matches.model_predicted == matches.model_measured).mean())
    median = float(np.nanmedian(scenario_rho))
    verdict = "SUPPORTED" if low > 0 and median >= 0.5 and fraction >= 0.6 else (
        "PARTIAL" if rho > 0 and median > 0 else "NOT_SUPPORTED"
    )
    return {
        "aggregate_spearman": rho, "aggregate_spearman_ci_low": float(low),
        "aggregate_spearman_ci_high": float(high), "median_within_scenario_spearman": median,
        "best_model_match_fraction": fraction, "verdict": verdict,
        "scenario_spearman": {str(k): float(v) for k, v in scenario_rho.items()},
        "best_matches": matches.to_dict(orient="records"),
    }


def make_figure(combined: pd.DataFrame, stats: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = dict(zip(sorted(combined.model.unique()), plt.cm.viridis(np.linspace(0.08, 0.9, 5))))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for model, group in combined.groupby("model"):
        axes[0].scatter(group.predicted_slope, group.measured_slope, label=model, color=colors[model], s=35)
        axes[2].scatter(group.companion_spectral_radius_mean, group.measured_slope, color=colors[model], s=35)
    lo = min(combined.predicted_slope.min(), combined.measured_slope.min())
    hi = max(combined.predicted_slope.max(), combined.measured_slope.max())
    axes[0].plot([lo, hi], [lo, hi], "--", color="0.35", lw=1)
    axes[0].set(xlabel="H_actual predicted NRMSE slope", ylabel="Nonlinear measured NRMSE slope",
                title=f"All processes: Spearman={stats['aggregate_spearman']:.2f}")
    scenario = pd.Series(stats["scenario_spearman"]).sort_values()
    axes[1].barh(scenario.index, scenario.values, color="#4477AA")
    axes[1].axvline(0, color="0.25", lw=1)
    axes[1].set(xlabel="Within-process Spearman", title="Ordering fidelity")
    axes[2].set(xlabel="Companion spectral radius", ylabel="Measured NRMSE slope",
                title="Dynamics versus degradation")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_theory_predictions_vs_measured.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def measure(cfg: dict, hp_map: dict, commit: str) -> None:
    prediction = RESULTS / "theory_predictions_frozen.csv"
    manifest = (RESULTS / "stage2_predictions_frozen.sha256").read_text().split()[0]
    if not prediction.exists() or sha256(prediction) != manifest:
        raise RuntimeError("frozen predictions missing or changed before nonlinear measurement")
    objects, dynamics = system_objects(cfg, hp_map)
    raw_rows = []
    for spec in cfg["scenarios"]:
        segments = scenario_data(cfg, spec)
        all_segments = np.vstack(segments)
        cuts = (len(segments[0]), len(segments[0]) + len(segments[1]))
        for seed in cfg["seeds"]:
            for model in cfg["models"]:
                channel, rho_star, kernel, _ = objects[(seed, model)]
                features_all = run_batched_reservoir(
                    kernel, channel, rho_star, all_segments, check_every=cfg["segment_length"]
                ).features
                features = (features_all[:cuts[0]], features_all[cuts[0]:cuts[1]], features_all[cuts[1]:])
                raw_rows.extend(evaluate_all(features, segments, cfg, model, spec["name"], seed, "nonlinear_simulator"))
        print(f"nonlinear measurements complete: {spec['name']}", flush=True)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(RESULTS / "measured_forecasts_raw.csv", index=False)
    measured = summarize(raw, dynamics, cfg, "measured")
    predicted = pd.read_csv(prediction)
    combined = predicted.merge(
        measured[["scenario", "model", "measured_slope", "measured_slope_ci_low", "measured_slope_ci_high"]],
        on=["scenario", "model"], validate="one_to_one",
    )
    combined["rank_match"] = combined.predicted_rank == combined.groupby("scenario").measured_slope.rank(method="average")
    combined.to_csv(RESULTS / "theory_predictions_vs_measured.csv", index=False)
    curves = raw.groupby(["scenario", "model", "horizon"], as_index=False).nrmse.mean().rename(columns={"nrmse": "measured_nrmse"})
    theory_raw = pd.read_csv(RESULTS / "theory_predictions_raw.csv")
    theory_curves = theory_raw.groupby(["scenario", "model", "horizon"], as_index=False).nrmse.mean().rename(columns={"nrmse": "predicted_nrmse"})
    theory_curves.merge(curves, on=["scenario", "model", "horizon"]).to_csv(
        RESULTS / "theory_vs_measured_curves.csv", index=False
    )
    stats = aggregate_statistics(combined, cfg)
    make_figure(combined, stats)
    nonlinear = combined[combined.scenario == "nonlinear_ar1_phi085"].set_index("model")
    surrogate = combined[combined.scenario == "phase_surrogate_nonlinear_ar1_phi085"].set_index("model")
    surrogate_delta = (surrogate.measured_slope - nonlinear.measured_slope).to_dict()
    report = [
        "# Rota A Gate 2 — synthetic theory validation", "",
        f"**Mechanical verdict: {stats['verdict']}.**", "",
        "The predictions were frozen from `H_actual` before nonlinear simulation. The external",
        "`W_K(z)R(z)` factorization was not used.", "", "## Frozen aggregate checks", "",
        f"- Aggregate Spearman: {stats['aggregate_spearman']:.6f} "
        f"(bootstrap 95% CI [{stats['aggregate_spearman_ci_low']:.6f}, {stats['aggregate_spearman_ci_high']:.6f}]).",
        f"- Median within-scenario Spearman: {stats['median_within_scenario_spearman']:.6f}.",
        f"- Predicted/measured best-model match fraction: {stats['best_model_match_fraction']:.3f}.",
        "- Caveat: the aggregate coefficient includes between-process slope-scale differences and",
        "  therefore is not an isolated measure of within-process model ordering; the frozen rule",
        "  also requires the within-scenario median and best-model match reported above.",
        "", "## Per-scenario ordering", "",
        "| Scenario | Spearman | Predicted best | Measured best |", "|---|---:|---|---|",
    ]
    match_map = {row["scenario"]: row for row in stats["best_matches"]}
    for scenario, rho in stats["scenario_spearman"].items():
        match = match_map[scenario]
        report.append(f"| {scenario} | {rho:.3f} | {match['model_predicted']} | {match['model_measured']} |")
    report += ["", "## Phase-surrogate diagnostic", "",
               "Measured slope change (surrogate minus higher-order source):"]
    for model, delta in sorted(surrogate_delta.items()):
        report.append(f"- `{model}`: {delta:+.6f}.")
    report += ["", "## Interpretation and limits", "",
               "SUPPORTED is the mechanical frozen classification, not uniform agreement. Negative",
               "within-scenario correlations are explicit failures of the local ordering prediction.",
               "The verdict applies only to the frozen processes, amplitudes, channel seeds and linear readout.",
               "Agreement supports local spectral/dynamical prediction; disagreement marks nonlinear,",
               "observability or finite-sample effects not captured by the tangent theory. This is not a",
               "claim of quantum advantage or universal superiority.", "", "Stage 2 stops here. No EEG rerun,",
               "shots, physical-resource analysis or manuscript work was performed."]
    (RESULTS / "gate2_report.md").write_text("\n".join(report) + "\n")
    metadata = {
        "configuration": cfg["configuration"], "git_commit": commit,
        "timestamp": datetime.now(ZoneInfo(cfg["timezone"])).isoformat(), "timezone": cfg["timezone"],
        "python": platform.python_version(), "numpy": np.__version__, "protocol_sha256": sha256(PROTOCOL),
        "config_sha256": sha256(CONFIG), "prediction_sha256": manifest, "statistics": stats,
        "working_tree_clean": not bool(git("status", "--porcelain")),
    }
    artifacts = [
        RESULTS / "theory_predictions_frozen.csv", RESULTS / "theory_predictions_vs_measured.csv",
        RESULTS / "theory_vs_measured_curves.csv", RESULTS / "stage2_model_dynamics.csv",
        RESULTS / "gate2_report.md", FIGURES / "fig_theory_predictions_vs_measured.pdf",
        FIGURES / "fig_theory_predictions_vs_measured.png",
    ]
    metadata["artifact_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in artifacts}
    (RESULTS / "stage2_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "measure", "all"))
    args = parser.parse_args()
    cfg, hp_map, commit = load_configuration()
    if args.phase in ("predict", "all"):
        predict(cfg, hp_map, commit)
    if args.phase in ("measure", "all"):
        measure(cfg, hp_map, commit)


if __name__ == "__main__":
    main()
