# Confirmatory effective-kernel protocol — corrected frozen configuration

Status: **FROZEN BEFORE the corrected r=0.7 simulation** (2026-07-14,
America/Sao_Paulo).

## Reason for correction and configuration authority

The prior numerical check used K=15, r=0.9 and past_mass=0.3. It remains preserved under
`results/eeg/_invalid_config_r09_snapshot/` with formal status `INVALID_CONFIG`. This repetition
corrects configuration only; it does not change the hypothesis, metrics, tolerances, seed,
amplitude or verdict rule after observing results.

By explicit human decision, the official Gate 1 configuration is the `single_kernel.hp` entry in
the committed `HEAD:results/eeg/hp_selected.json`, produced by segment-blocked ridge selection:
K=15, r=0.7, past_mass=0.3. The current working-tree copy of that path contains an uncommitted
r=0.9 rerun. It must not be modified. The script therefore reads the committed blob automatically,
prints it, verifies its exact expected values and records both the authoritative source and the
working-tree divergence. Any other committed value yields `INVALID_CONFIG`; failure to resolve
commit/configuration yields `INVALID_PROVENANCE`.

## Frozen system and signals

- Existing four-qubit input channel and 66 Pauli observables; seed 1.
- Expansion point: constant standardized input u0=0 and its converged fixed state.
- Fixed-state convergence tolerance: Frobenius difference below 1e-13, maximum 5000 iterations.
- Confirmatory perturbation epsilon=1e-4; response length 256 samples.
- Initial current state and all K+1 history-buffer entries equal the fixed state.
- Nonlinear impulse at t=0 and step from t=0, with baseline subtraction followed by division by
  epsilon.

No Bonn signal or EEG result is read or modified by this mechanism check.

## Independently implemented models

The tangent state uses an orthonormal 255-dimensional traceless-Hermitian basis. A is built by
applying the fixed-input CPTP derivative to every basis matrix; B is a central input derivative;
C is the 66-observable linear map. Tangent evolution uses only the resulting numeric A, B, C and
kernel weights, not the nonlinear simulator's response arrays or state-update function:

`delta x[t+1] = A sum_(tau=0)^K w_tau delta x[t-tau] + B delta u[t]`.

The falsifiable separable ansatz is constructed separately by convolving the K=0 tangent impulse
from the same A, B, C with `[w0,...,wK]`: `H_sep(z)=W_K(z)R(z)`.

## Frozen confirmatory metrics and tolerances

All arrays must be finite and shape `(256,66)`. Norms combine time and features.

- impulse relative Frobenius error <= 0.01;
- step relative Frobenius error <= 0.01;
- 256-point complex-FFT relative Frobenius error <= 0.01;
- L1 error between normalized impulse-energy memory distributions <= 0.02.

Cosine similarity is diagnostic only and has no threshold. Additional invariants are: kernel
weights sum to one; cumulative tangent impulse reproduces tangent step; the companion spectrum is
finite; and local stability is decided by companion spectral radius <1.

## Verdict rule

- `TANGENT_PASS`: all four confirmatory tangent metrics pass.
- `SEPARABLE_PASS`: all four confirmatory separable metrics pass.
- `PASS`: both pass.
- `FAIL_SEPARABLE_FACTORIZATION`: tangent passes and separable fails at least one metric.
- `FAIL_LINEARIZATION`: tangent fails at least one metric.
- `INVALID_CONFIG`: authoritative HP is not K=15, r=0.7, past_mass=0.3.
- `INVALID_PROVENANCE`: commit/configuration or required artifacts cannot be identified.

No result is presumed. The rule is applied mechanically.

## Post-gate amplitude robustness

Only after the epsilon=1e-4 verdict is fixed, evaluate epsilon in
`{1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2}`. Report tangent-versus-nonlinear impulse and step errors in
`theory_linearity_sweep.csv`. This secondary sweep cannot select a new confirmatory epsilon or
change the verdict.

The process stops at Gate 1. It does not create Stage 2, shots, physical-resource or manuscript
artifacts.
