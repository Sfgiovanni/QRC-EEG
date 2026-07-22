# Claims registry

This registry is normative for future writing. It does not assert quantum advantage.

## C1 — SUPPORTED

Distributed state memory acts inside feedback and changes augmented-system dynamics.

- Permitted: “Distributed state memory acts inside feedback and changes the augmented-system dynamics.”
- Prohibited: “The kernel is only an external W(z) filter applied to K=0.”
- Evidence: `results/eeg/theory_vs_sim_check.csv`
- Limitation: Local linear mechanism; not hardware realization.

## C2 — SUPPORTED_LOCAL

The tangent recurrence locally reproduces the simulator.

- Permitted: “The tangent recurrence reproduces the simulator locally.”
- Prohibited: “The linear theory explains all EEG forecasting performance.”
- Evidence: `results/eeg/theory_vs_sim_check.csv`
- Limitation: Local small-signal statement only.

## C3 — SUPPORTED_WITH_LIMITS

Theory predicts between-process differences and partially recovers within-process ordering.

- Permitted: “Theory predicts process differences and partially recovers model ordering within processes.”
- Prohibited: “Theory universally predicts the best kernel.”
- Evidence: `results/synth/theory_predictions_vs_measured.csv;results/synth/gate2_postgate_sensitivity.csv`
- Limitation: Within-scenario evidence is moderate and has explicit failures.

## C4 — SUPPORTED_F_Z

Distributed memory changes horizon dependence in F and Z.

- Permitted: “Distributed memory alters error dependence on horizon in F and Z.”
- Prohibited: “There is quantum advantage or universal forecasting superiority.”
- Evidence: `results/eeg/gate_interactions.csv`
- Limitation: Single Bonn database; segment-level; h=64 lacks absolute skill.

## C5 — NULL

S is null in the primary causal test.

- Permitted: “S yielded a null result in the primary causal test.”
- Prohibited: “The effect was demonstrated in all three sets.”
- Evidence: `results/eeg/gate_interactions.csv`
- Limitation: Null is a valid result and remains visible.

## C6 — PARTIAL

Distributed-memory class differs from discrete delay in some comparisons.

- Permitted: “Results favor distributed memory over discrete delay in part of the comparisons.”
- Prohibited: “The exponential form is universally superior to triangular and uniform forms.”
- Evidence: `results/eeg/gate_nrmse_curves.csv;results/eeg/gate_interactions.csv`
- Limitation: Exponential, triangular and uniform shapes overlap.

## C7 — MIXED_SHOT_SENSITIVITY

Finite-shot sensitivity is heterogeneous; principal F/Z signs persist.

- Permitted: “Shot robustness is heterogeneous and principal F/Z signs are qualitatively preserved.”
- Prohibited: “The method is globally shot-robust or hardware-ready.”
- Evidence: `results/eeg/shot_sensitivity_classification.csv;results/eeg/shot_sensitivity_contrasts.csv`
- Limitation: Large tail; readout sampling only; not hardware.

## C8 — SUPPORTED_F_Z_GENERIC_MECHANISM (follow-up, additive)

Grafting the same causal state-mixing structure onto a dimension-matched classical leaky-integrator
ESN (`ESN66_kernel` vs `ESN66_K0`) reproduces the F/Z slower-degradation pattern seen in the QRC
kernel vs `QRC_K0`, under the same h=2→h=64 crossed segment×seed bootstrap: Z +0.0228 (95% CI
[0.0062, 0.0438]), F +0.0412 (95% CI [0.0199, 0.0671]), both Holm-significant in the
`eeg_followup_crossed_sensitivity` family. In S the classical effect is small and does not survive
Holm correction (+0.0087, 95% CI [0.0007, 0.0162], Holm p=0.19), consistent with the QRC S-null
(C5).

- Permitted: “The slower-degradation pattern versus a no-memory control also appears in a
  dimension-matched classical ESN control in F and Z, which is compatible with a mechanism generic
  to distributed recurrent mixing rather than specific to the quantum substrate; S remains null in
  both substrates.”
- Prohibited: “The classical control proves the QRC mechanism is non-quantum,” “the ESN control
  demonstrates quantum advantage,” or any claim that treats the classical-ESN replication as
  evidence *for* rather than *compatible with* a substrate-independent mechanism.
- Evidence: `results/eeg/followup/crossed_inference/crossed_bootstrap.csv` (rows `ESN66_kernel vs
  ESN66_K0`), `results/eeg/followup/classical_control/tab_classical_distributed_memory.csv`,
  `results/eeg/followup/technical_report.md`.
- Limitation: One classical substrate (leaky-integrator ESN), one HP-search budget (two modes
  tested, both converged to the same core HP in this run); Bonn segment-to-subject mapping remains
  unavailable, no patient-level claim.

## C9 — NULL_OR_REVERSED_VS_CONCENTRATED_DELAY (follow-up, additive)

A comparison not present in the frozen `eeg_gate` family — the exponential distributed kernel
against the concentrated single-lag control (`AB_noaux`/`ESN66_AB`) under the same h=2→h=64
crossed metric — does **not** favor the distributed kernel in either substrate. `single_kernel` vs
`AB_noaux`: Z −0.0443, F −0.0148, S −0.0881 (all Holm-significant, all in the direction of
AB_noaux degrading *more slowly* than the kernel). `ESN66_kernel` vs `ESN66_AB`: Z −0.0300
(Holm-significant), F −0.0033 (not significant, CI includes 0), S −0.0769 (Holm-significant), same
sign pattern.

- Permitted: “Under the h=2→h=64 degradation contrast, the exponential distributed kernel does not
  degrade more slowly than the concentrated single-lag control in either substrate; in most
  set×substrate cells the concentrated control degrades more slowly.”
- Prohibited: “Distributed memory is universally superior to concentrated/discrete delay,” or
  citing only the `QRC_K0`/`ESN66_K0` comparisons (C8) while omitting this one.
- Evidence: `results/eeg/followup/crossed_inference/crossed_bootstrap.csv` (rows `... vs AB_noaux` /
  `... vs ESN66_AB`), `results/eeg/followup/technical_report.md` Section 2-3.
- Limitation: Single frozen `AB_noaux`/`ESN66_AB` configuration (`tau=5, delayed_mass=0.3`); a
  different concentrated-delay configuration could behave differently. This nuances, and must be
  read alongside, C6 (kernel-*shape* comparisons among distributed kernels remain a separate,
  already-partial finding).

## C10 — ROBUST_TO_CROSSED_RESAMPLING_WITH_ONE_EXCEPTION (follow-up, additive)

A segment×seed crossed bootstrap sensitivity analysis (`docs/crossed_inference_protocol.md`),
preserving both variance sources instead of averaging seeds first, was applied to 21 tests spanning
the canonical `single_kernel` contrasts and the two new classical-control contrasts. Of 10 tests
significant under a same-design raw-CI replication, 8 remained significant after crossed
resampling and Holm correction; the 2 that did not were both `ESN66_kernel vs ESN66_K0` in S
(the classical analogue of the already-null QRC S result, C5). A secondary crossed mixed-model
check (`nrmse ~ construction*horizon + (1|segment) + (1|seed)`, fit via a statsmodels
variance-components approximation since no R/lme4 is available in this environment) agreed in sign
with the bootstrap in all 21/21 cells where it produced a finite estimate, but converged in only
18/21 and flagged a variance-component boundary in 21/21 — recorded as a diagnostic limitation, not
hidden; the crossed bootstrap remains the primary sensitivity endpoint per the frozen protocol.

- Permitted: “The F/Z slower-degradation findings are robust to joint segment×seed resampling; a
  marginal classical-ESN effect in S does not survive this stricter check, reinforcing rather than
  contradicting the pre-existing S-null (C5).”
- Prohibited: “The crossed bootstrap validates every original claim unconditionally,” or citing the
  mixed model as an independently converged confirmation given its near-universal boundary/singular
  diagnostics.
- Evidence: `results/eeg/followup/crossed_inference/crossed_bootstrap.csv`,
  `results/eeg/followup/crossed_inference/original_style_replication.csv`,
  `results/eeg/followup/crossed_inference/mixed_model_diagnostics.json`.
- Limitation: Mixed-model diagnostics are largely unfavorable (boundary/singular) in this small
  (20 segments × 10 seeds) design; treat the mixed model as a secondary, mostly-inconclusive check,
  not as independent confirmation.
