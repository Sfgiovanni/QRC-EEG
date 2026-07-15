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
