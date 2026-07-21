# Gate 1B — post-gate robustness of the effective-kernel mechanism

Status: **prespecified grid frozen before execution** (2026-07-21,
America/Sao_Paulo). This is a **post-gate robustness analysis**, not a
preregistered confirmatory gate. It does not alter, overwrite, or
retrospectively reinterpret the frozen Gate 1.

## Relationship to the frozen Gate 1

The original confirmatory Gate 1 stays frozen and canonical:

- construction `single_kernel`, K=15, r=0.7, past_mass=0.3;
- channel seed 1, operating point u0=0, confirmatory epsilon 1e-4.

Gate 1 falsified the separable factorization `H_sep(z)=W_K(z)R(z)` and
numerically validated the implementation-faithful tangent recurrence at that
single operating point. Its artifacts
(`results/eeg/theory_vs_sim_check.csv`, `..._responses.npz`,
`..._metadata.json`, `theory_linearity_sweep.csv`,
`effective_kernel_symbolic.txt`, `config/effective_kernel_gate1_frozen.json`)
and the `INVALID_CONFIG` r=0.9 snapshot are **not** touched by this analysis.
Gate 1B writes only under `results/eeg/gate1b_robustness/` and its own
config/docs/scripts/tests/figures.

## Scientific question

Over a grid of channel seeds, damping ratios `r`, and operating points `u0`
**frozen before execution**, how robust is the *numerical* agreement of the
tangent recurrence with the nonlinear simulator, and does the separable
factorization remain falsified? Three things are logically distinct and are
reported separately:

1. the **structural derivation** (`H_actual = C[zI - A W_K(z)]^{-1} B`), which
   does not depend on the seed;
2. the **falsification of separability**, for which a single counterexample
   already suffices (Gate 1 provides it);
3. the **numerical robustness of the tangent approximation** across the grid —
   the actual object of this extension — together with the region of parameter
   space where it holds and the unstable/failing cases.

This analysis makes no claim of quantum advantage, clinical validation,
hardware readiness, or environmental non-Markovianity.

## Frozen grid

| Field | Value |
|---|---|
| construction | `single_kernel` |
| K | 15 |
| past_mass | 0.3 |
| r | {0.7, 0.9} |
| channel_seeds | {1,…,10} |
| operating_points u0 | {-0.5, 0.0, 0.5} |
| confirmatory epsilon | 1e-4 |
| derivative epsilon | 1e-4 |
| amplitude sweep | {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2} |
| response_length | 256 |
| fixed-point tolerance | 1e-13 |
| fixed-point max iterations | 5000 |

Total: 10 seeds × 2 `r` × 3 `u0` = **60 configurations**.

Frozen tolerances (identical to Gate 1):
`impulse_relative_frobenius <= 0.01`, `step_relative_frobenius <= 0.01`,
`frequency_relative_frobenius <= 0.01`, `memory_function_l1 <= 0.02`.
Cosine similarities are diagnostic only.

## Operating-point convention (verified before freezing)

The frozen four-qubit input channel encodes a *squashed* input:
`x = logistic(u)` then `|psi(x)> = sqrt(1-x)|0> + sqrt(x)|1>`, `x in (0,1)`.
`u0` is therefore the raw pre-squash (z-scored) constant input. The three
frozen points map to encoded amplitudes

- u0 = -0.5 → x = 0.37754,
- u0 =  0.0 → x = 0.50000 (the Gate 1 point),
- u0 = +0.5 → x = 0.62246,

all interior to `(0,1)` with no clipping. The points were chosen purely from
the model input domain, symmetric about the Gate 1 point, before observing any
result.

## Linearization reuse (mathematical justification)

For constant input `u0`, the fixed state `rho_*`, tangent operator
`A = D_rho Phi_{u0}|_{rho_*}`, input derivative `B = D_u Phi_u(rho_*)|_{u0}`,
and observable map `C` depend only on `(seed, u0)` and are **independent of the
kernel damping `r`**: `r` enters only the history-mixing weights `w_tau`, which
act *after* the channel in the recurrence and in the companion polynomial.
Gate 1B therefore computes `rho_*, A, B, C` once per `(seed, u0)` and reuses
them for both `r=0.7` and `r=0.9`. `C` further depends on neither the seed nor
`u0` (it is the fixed 66-observable projection). This reuse is exact, not an
approximation, and is asserted in code.

## Per-configuration procedure

For each `(seed, r, u0)`:

1. build the channel with the given seed;
2. compute the fixed state under constant `u0`; record convergence, iteration
   count and final Frobenius difference;
3. build the orthonormal 255-dimensional traceless-Hermitian basis;
4. compute `A`, `B`, `C` at the operating point (`derivative_epsilon = 1e-4`);
5. compute tangent impulse/step, nonlinear impulse/step (baseline `u0`,
   central subtraction, division by epsilon), and the separable `W_K(z)R(z)`
   response;
6. compute the six Gate 1 indicators (four tolerance metrics + two diagnostic
   cosines) for the tangent and separable theories;
7. compute the companion spectrum, spectral radius and local stability;
8. run the amplitude sweep (nonlinear probe over the seven epsilons) without
   allowing it to change the epsilon = 1e-4 classification.

Nothing is silenced: convergence failure, spectral radius ≥ 1, NaN/inf,
tangent failure, a passing separable construction, or an invalid configuration
are all recorded. Each configuration is isolated in `try/except`; an exception
becomes a recorded failure row and does not abort the grid.

## Validity and denominators

A configuration is **valid** iff its fixed point converged, no exception
occurred, and all four confirmatory response arrays are finite. Non-finite
metrics are tagged `pass = False` (never left as bare NaN). Classification
thresholds are computed over valid configurations, but every summary reports
`n_total` (all 60), `n_valid`, and — separately — the locally-stable subset, so
that unstable or non-converged configurations never silently vanish from the
denominator.

## Frozen descriptive classification

Purely for this extension; it cannot modify Gate 1.

- **ROBUST_WITHIN_GRID**: ≥90% of valid configurations pass all four tangent
  tolerances simultaneously AND ≤10% pass all four separable tolerances
  simultaneously.
- **MIXED**: outside those bounds, but the tangent passes all four
  simultaneously in >50% of valid configurations and clearly exceeds the
  separable hypothesis.
- **NOT_ROBUST_WITHIN_GRID**: the tangent passes all four simultaneously in
  ≤50% of valid configurations, OR the separable construction passes all four
  simultaneously in ≥50%.
- **INVALID**: incomplete grid, non-finite unclassified artifacts, invalid
  provenance, or integrity failure.

The classification is reported globally, for r=0.7, for r=0.9, and for each
operating point. It is recomputed mechanically by the verifier from the CSVs.

## Artifacts

Under `results/eeg/gate1b_robustness/`:
`metrics_by_configuration.csv`, `amplitude_sweep.csv`,
`spectrum_by_configuration.csv`, `summary.csv`, `metadata.json`, `report.md`.
Figures: `figures/eeg/fig_gate1b_robustness.pdf` and `.png`.

## Reproduction

```
.venv/bin/python scripts/run_gate1b_robustness.py
.venv/bin/python scripts/verify_gate1b_robustness.py
.venv/bin/python -m pytest tests/test_gate1b_robustness.py -q
```

This analysis stops at the robustness artifacts and the report. It does not
update the canonical claims registry, does not integrate into the canonical
release, and produces no manuscript or `.tex` artifact.
