# Rota A Stage 2 — frozen synthetic validation protocol

Status: **FROZEN BEFORE nonlinear synthetic reservoir simulations** on 2026-07-14
(America/Sao_Paulo).

## Question and theory used

This gate tests whether the local implementation-faithful transfer function

\[
H_{\mathrm{actual}}(z)=C[zI-AW_K(z)]^{-1}B
\]

predicts the measured horizon-degradation ordering. The falsified external factorization
`W_K(z)R(z)` is not used. No quantum advantage or universal kernel superiority is hypothesized.

The theory phase runs first, writes `results/synth/theory_predictions_frozen.csv`, and freezes its
SHA256. The nonlinear phase must verify that hash before it runs. Measured results cannot alter a
theoretical prediction, scenario, tolerance, split, model, or decision rule.

## Frozen design

- Models: QRC K=0, discrete AB delay, exponential distributed kernel, recent-triangular kernel,
  and uniform kernel. Their HP are loaded from the committed `HEAD:results/eeg/hp_selected.json`;
  K=0 is parameter-free. The dirty working copy is not used or modified.
- Four-qubit channel, 66 observables, channel seeds 1, 2 and 3.
- Per scenario: 12 train, 6 validation and 12 held-out test segments, each 768 samples.
- Training-only global scaling; complete segments remain disjoint. Ridge alpha is selected on the
  complete validation segments and refitted on train+validation.
- Washout 50; horizons `{1,2,4,8,16,32,64,128}`; alpha grid frozen in
  `config/rotaA_stage2_frozen.json`.
- Theory features are causal convolutions of each standardized input with the 512-sample impulse
  response of `H_actual`. Nonlinear features use the existing batched state-history simulator,
  initialized at the same constant-input fixed state. Neither route uses future samples.

## Processes

AR(1) uses phi `{0.30,0.60,0.85,0.95}`. AR(2) uses complex-conjugate roots with frequency 0.12
cycles/sample and radii `{0.75,0.90}`. Colored Gaussian noise uses PSD exponents beta `{0.5,1.5}`.
A quadratically distorted AR(1), `x+0.35(x^2-E[x^2])`, supplies higher-order structure. Its
phase-randomized surrogate preserves each finite segment's Fourier magnitudes while destroying
phase relations; this construction step is not a fitted preprocessing operation.

## Frozen outcomes and decision

For every scenario/model, degradation is the OLS slope of mean NRMSE against `log2(h)`. Report
predicted and nonlinear-measured slopes, bootstrap 95% CI over held-out `(seed,segment)` blocks,
mean companion radius, delayed conditional `T_eff`, full mean lag, rank, and whether the predicted
lowest-degradation model matches the measured one.

Across all scenario/model points compute Spearman correlation between predicted and measured
slopes, with a paired bootstrap CI. Also report the median within-scenario Spearman correlation
and the fraction of scenarios whose predicted best model is measured best.

- **SUPPORTED:** aggregate Spearman lower 95% bound >0, median within-scenario Spearman >=0.5,
  and best-model match fraction >=0.6.
- **PARTIAL:** otherwise, aggregate Spearman point estimate >0 and median within-scenario
  Spearman >0.
- **NOT_SUPPORTED:** otherwise.

The surrogate comparison is reported separately: preservation of the predicted/measured ordering
supports a predominantly linear-PSD explanation; a material change indicates higher-order effects
outside the local theory. There is no post-hoc model exclusion.

The process stops at Gate 2 after tests, report, figure and SHA256. No shots, physical-resource
analysis, EEG rerun, second EEG database or manuscript is permitted in this stage.
