# Analysis provenance

## Frozen empirical gate

`results/eeg/gate_report.md` is the immutable output of the preregistered EEG gate. Its frozen
SHA256 is recorded in `results/eeg/gate_report_frozen.sha256` as
`9f1d7717c4061ee31ee19e0dc9f666a27d1c7ec9a7c743bba56a8f8820c79247`.
Rota A does not regenerate, revise or reinterpret that artifact.

## Rota A Stage 0

Stage 0 consumes the already-generated held-out segment rows and does not fit or evaluate any
new model. `scripts/make_useful_horizon_v2.py` averages confirmatory seeds within each test
segment, applies the same bootstrap-over-segments criterion to every construction, and writes
`results/eeg/useful_horizon_v2.csv`. `scripts/update_results_gate.py` regenerates `RESULTS.md`
from gate CSVs and the v2 table. `scripts/verify_rotaA_gate0.py` independently recomputes the
criterion, checks narrative consistency and the frozen gate hash, runs tests, and updates SHA256.

Canonical Stage 0 command: `bash scripts/run_rotaA_stage0.sh`. It stops at Gate 0. No effective-
kernel theory, synthetic process, shot experiment or manuscript artifact is produced in this
stage.

## Rota A Stage 1

The corrected confirmatory settings were frozen before simulation in
`docs/effective_kernel_check_protocol.md`; its SHA256 is pinned in
`results/eeg/effective_kernel_protocol_frozen.sha256`. The Stage 1 simulation reads no Bonn data.
By human decision, the authoritative configuration is read automatically from the committed
`HEAD:results/eeg/hp_selected.json`: `single_kernel`, K=15, r=0.7, past_mass=0.3. The dirty
working-tree copy at that path is recorded but not modified. The former r=0.9 outputs are
preserved under `results/eeg/_invalid_config_r09_snapshot/` with status `INVALID_CONFIG`.

The script linearizes the frozen four-qubit channel around its constant-input fixed state and
compares nonlinear impulse/step responses with (i) the implementation-faithful tangent recurrence
and (ii) the separable `W(z)R(z)` proposal. SymPy verifies the geometric closed forms and that the
two scalar transfer expressions are not generically identical.

`scripts/run_effective_kernel_check.py` writes the metric CSV, metadata JSON, symbolic record,
compressed response arrays and a separately marked post-gate amplitude sweep.
`docs/effective_kernel_theory.md` records the derivation and the corrected confirmatory
**FAIL_SEPARABLE_FACTORIZATION** result. Canonical command:
`bash scripts/run_rotaA_stage1.sh`. It stops at Gate 1; no Stage 2 artifact is produced.

## Rota A Stage 2

The protocol and configuration are frozen in `docs/synthetic_stage2_protocol.md` and
`config/rotaA_stage2_frozen.json`. `scripts/run_synthetic_stage2.py predict` uses the corrected
`H_actual` recurrence and writes a prediction CSV plus SHA256 before any nonlinear synthetic
reservoir is evaluated. The `measure` phase refuses to run if that prediction hash changes, then
uses training-only scaling, complete disjoint train/validation/test segments and validation-only
ridge selection on the frozen AR(1), AR(2), colored-noise, higher-order and phase-surrogate set.

`results/synth/gate2_report.md` applies the frozen SUPPORTED/PARTIAL/NOT_SUPPORTED rule and reports
both agreements and failures without claiming quantum advantage. Canonical command:
`bash scripts/run_rotaA_stage2.sh`. It stops at Gate 2; no shot, physical-resource or manuscript
artifact is produced.

## Rota A Stage 3

The Stage 2 post-gate addendum reads only frozen synthetic CSVs and does not rerun a reservoir.
The Gate 3 protocol and configuration were frozen, hashed, and recorded before finite-shot
results. Reservoir settings are loaded from the committed blob `HEAD:results/eeg/hp_selected.json`;
the execution aborts if its canonical six-model mapping or any frozen EEG split/reference changes.

The exact official-r=0.7 baseline must reproduce every available frozen segment row before shots.
Finite-shot features use independent binomial Pauli sampling in train, validation and test, with
ridge alpha selected only on complete validation segments. Resource counts are independently
derived from qubit count, kernel buffer length, dtype, observable count and trajectory length.

Canonical command: `bash scripts/run_rotaA_stage3.sh`. It writes the raw/summary/contrast shot
tables, resource table, figures, metadata and `results/eeg/gate3_report.md`, then runs the full
test suite and `scripts/verify_gate3.py`. It stops at Gate 3; no manuscript, release or Stage 4
artifact is generated.
