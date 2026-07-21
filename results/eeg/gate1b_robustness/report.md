# Gate 1B — post-gate robustness of the effective-kernel mechanism

Post-gate robustness analysis over a **prespecified grid frozen before execution**. It does not alter, overwrite, or retrospectively reinterpret the frozen confirmatory Gate 1.

- Generated: 2026-07-21T17:45:27.294386-03:00 (America/Sao_Paulo)
- Origin commit: `cfb8e08f769bfe5cc2961e3a93421a8490cc9796`
- Python 3.13.9, NumPy 2.5.1, Pandas 3.0.3, SymPy 1.14.0
- Gate 1 canonical artifacts unchanged: **True**
- Gate 1 corner (seed=1, r=0.7, u0=0, eps=1e-4) reproduced: **True**

## Scientific question

Over a grid of channel seeds, damping ratios `r`, and operating points `u0` frozen before execution, how robust is the *numerical* agreement of the implementation-faithful tangent recurrence with the nonlinear simulator, and does the separable factorization `H_sep(z)=W_K(z)R(z)` remain falsified? The structural derivation is seed-independent; falsifying separability needs only one counterexample (Gate 1); this extension characterizes the numerical robustness of the tangent approximation and where it holds or fails.

## Frozen grid

| Field | Value |
|---|---|
| construction | single_kernel |
| K | 15 |
| past_mass | 0.3 |
| r | [0.7, 0.9] |
| channel_seeds | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] |
| operating_points u0 | [-0.5, 0.0, 0.5] |
| confirmatory epsilon | 0.0001 |
| amplitude sweep | [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01] |
| response_length | 256 |
| total configurations | 60 (expected 60) |

## Status counts

- success: 60
- fixed_point_failed: 0
- unstable: 0
- nonfinite: 0
- exception: 0
- total wall time: 167.2 s (2.70 s/config mean)

## Global classification

**ROBUST_WITHIN_GRID** — tangent all-four pass fraction 1.000, separable all-four pass fraction 0.000 (over 60/60 valid).
Locally stable: 1.000 of valid, 1.000 of all.

## Stratified classification (tangent / separable all-four pass fraction)

| Stratum | Classification | Tangent | Separable | n_valid / n_total | Stable (valid) |
|---|---|---|---|---|---|
| global | ROBUST_WITHIN_GRID | 1.000 | 0.000 | 60 / 60 | 1.000 |
| r=0.7 | ROBUST_WITHIN_GRID | 1.000 | 0.000 | 30 / 30 | 1.000 |
| r=0.9 | ROBUST_WITHIN_GRID | 1.000 | 0.000 | 30 / 30 | 1.000 |
| u0=-0.5 | ROBUST_WITHIN_GRID | 1.000 | 0.000 | 20 / 20 | 1.000 |
| u0=0.0 | ROBUST_WITHIN_GRID | 1.000 | 0.000 | 20 / 20 | 1.000 |
| u0=0.5 | ROBUST_WITHIN_GRID | 1.000 | 0.000 | 20 / 20 | 1.000 |

## r=0.7 versus r=0.9

- **r=0.7**: ROBUST_WITHIN_GRID; tangent 1.000, separable 0.000, stable(valid) 1.000, stable(total) 1.000.
- **r=0.9**: ROBUST_WITHIN_GRID; tangent 1.000, separable 0.000, stable(valid) 1.000, stable(total) 1.000.

## Dependence on operating point u0

- **u0=-0.5**: ROBUST_WITHIN_GRID; tangent 1.000, separable 0.000, stable(valid) 1.000.
- **u0=0.0**: ROBUST_WITHIN_GRID; tangent 1.000, separable 0.000, stable(valid) 1.000.
- **u0=0.5**: ROBUST_WITHIN_GRID; tangent 1.000, separable 0.000, stable(valid) 1.000.

## Dependence on epsilon

Median tangent impulse / step error and median separable impulse error across valid configurations, by amplitude epsilon:

| epsilon | tangent impulse (median) | tangent step (median) | separable impulse (median) |
|---|---|---|---|
| 1e-05 | 2.865e-06 | 5.149e-06 | 3.983e-01 |
| 3e-05 | 8.595e-06 | 1.545e-05 | 3.983e-01 |
| 0.0001 | 2.865e-05 | 5.149e-05 | 3.983e-01 |
| 0.0003 | 8.595e-05 | 1.545e-04 | 3.983e-01 |
| 0.001 | 2.865e-04 | 5.152e-04 | 3.983e-01 |
| 0.003 | 8.594e-04 | 1.547e-03 | 3.983e-01 |
| 0.01 | 2.864e-03 | 5.173e-03 | 3.982e-01 |

## Spectral stability

- Companion spectral radius range: [0.947146, 0.995212].
- r=0.7: radius median 0.980237, max 0.993127, stable 30/30.
- r=0.9: radius median 0.986291, max 0.995212, stable 30/30.

## Failed / unstable / non-converged configurations

No invalid configurations: all 60 converged, executed without exception, and produced finite confirmatory arrays.

No valid configuration was locally unstable.

## Limited interpretation

- The **structural derivation** `H_actual = C[zI - A W_K(z)]^{-1} B` is algebraic and does not depend on the seed.
- The **falsification of the separable factorization** needs only the single Gate 1 counterexample; this grid shows it is not an isolated accident.
- The **numerical robustness of the tangent approximation** is the object here: it is characterized over the frozen grid, including the region where it holds and any unstable or failing cases, which are preserved above.
- This does **not** demonstrate quantum advantage, physical/hardware implementation, clinical validity, or environmental non-Markovianity.

## Relationship to the original Gate 1

Gate 1 (K=15, r=0.7, past_mass=0.3, seed=1, u0=0, eps=1e-4) remains frozen and canonical. This extension neither modifies its artifacts (hash-verified unchanged: **True**) nor updates the canonical claims registry. The u0=0/r=0.7/seed=1 corner of this grid reproduces the frozen Gate 1 numbers (reproduced: **True**).

## Files produced

- `results/eeg/gate1b_robustness/metrics_by_configuration.csv`
- `results/eeg/gate1b_robustness/amplitude_sweep.csv`
- `results/eeg/gate1b_robustness/spectrum_by_configuration.csv`
- `results/eeg/gate1b_robustness/summary.csv`
- `results/eeg/gate1b_robustness/metadata.json`
- `results/eeg/gate1b_robustness/report.md`
- `figures/eeg/fig_gate1b_robustness.pdf`
- `figures/eeg/fig_gate1b_robustness.png`

## Artifact hashes

- `metrics_by_configuration.csv`: `948929e502a6e33dc65d1de05c7033f0df03a0b8e28b64a4612194f949e4e0db`
- `amplitude_sweep.csv`: `deaa64bb92496fc8e67a50109ea5c2210375b9b0593a418f7d3d2a4e81d6144b`
- `spectrum_by_configuration.csv`: `def0b13e2b787d8038e966cfbaead61f1dcfd45be3ebfdb7f55616240c18bdab`
- `summary.csv`: `c29e0f362c89140ce550768f4304c084774a287c87980bfc9562b8200b11f4d6`
- config `e78b988afa66247e10d48da3e4fa69361eca6daca1a4d41d487e79c2a8ab763d`
- protocol `a383649cbc6ebc418d984f6026f983148fb75321f8e9c112b87a20ea3c1d1418`

## Reproduction

```
.venv/bin/python scripts/run_gate1b_robustness.py
.venv/bin/python scripts/verify_gate1b_robustness.py
.venv/bin/python -m pytest tests/test_gate1b_robustness.py -q
```
