# Rota A Gate 3 — frozen physical-resource and finite-shot protocol

Status: **FROZEN BEFORE generation of finite-shot results** on 2026-07-14,
America/Sao_Paulo.

## Scope and preserved gates

Gate 1 remains `FAIL_SEPARABLE_FACTORIZATION`; Gate 2 remains mechanically `SUPPORTED` with
moderate within-scenario evidence. The Gate 2 post-gate addendum is analytical only and cannot
change that verdict. Gate 3 characterizes resources and finite-sampling sensitivity; it cannot
establish quantum advantage, hardware readiness or classical superiority.

## Configuration and provenance

Every reservoir HP is loaded from the committed blob `HEAD:results/eeg/hp_selected.json`. The
script validates a hash of the full six-model mapping rather than embedding K, r, past mass or
delay. A mismatch is `INVALID_CONFIG`. Commit, dirty-tree status, dependency versions, split
hashes and artifact hashes are recorded. Existing user changes and all frozen gate artifacts are
preserved.

The current working-tree EEG gate was later regenerated with r=0.9, while the authoritative blob
and `_prefix_snapshot_gate` contain r=0.7. Exact reproduction is therefore checked against the
r=0.7 snapshot for single-kernel and AB rows at h={1,2,4,8}, seeds 1–3, and against the current
frozen rows for parameter-free K=0 and unchanged AB where available. Extended r=0.7 single-kernel
horizons have no frozen reference and are explicitly labeled as such. Maximum absolute NRMSE
difference on every available matched test-segment row must be <=1e-9 before shot noise is run;
otherwise stop as `INVALID_BASELINE_REPRODUCTION`.

## Frozen finite-shot design

- Main models fixed before results: QRC K=0, AB-noaux and single distributed kernel. Triangular
  and uniform are excluded from the shot experiment solely for projected compute cost, but remain
  in the resource table. No model may be added/removed after results.
- Sets Z, F and S; channel seeds 1, 2 and 3; every frozen train, validation and test segment; eight
  horizons h={1,2,4,8,16,32,64,128}; training-only scaling; segment-disjoint partitions.
- Exact features are generated once per set/model/seed. Finite levels are 100, 300, 1000, 3000
  and 10000 shots with 10 deterministic independent noise replicates. Ten was frozen instead of
  20 because the projected run already contains 27 density-matrix evolutions, 11,016
  validation-selected horizon fits and 73,440 held-out segment rows; it is the minimum allowed,
  selected before results and independent of performance.
- The exact baseline reproduction uses every valid temporal row. The shot experiment retains all
  20 frozen test segments but, for feasibility, evaluates the deterministic temporal grid
  `washout, washout+16, ...` in train, validation and test. This stride was frozen before the first
  shot result after estimating that the unstrided design would require about 35 billion binomial
  draws. The strided design still requires roughly 2.2 billion draws. It is independent of model
  performance, set and shot level; strided exact rows form the paired reference for inflation.
- For every replicate, binomial shot noise is applied independently to all 66 Pauli expectations
  in train, validation and test. Ridge alpha is reselected using complete validation segments and
  the readout is refit on train+validation. Test segments are evaluated once.

For exact Pauli expectation mu, use `p=(1+clip(mu,-1,1))/2`, draw
`k~Binomial(N,p)`, and return `mu_hat=2k/N-1`. Thus
`Var(mu_hat)=(1-mu^2)/N`. The intercept is added only inside the readout and is never noised.
Independent observables represent a conservative separate-ensemble model. No Gaussian noise,
fake within-group correlation or backaction model is used. Qubit-wise commuting grouping is not
implemented.

## Frozen metrics and classification

Raw rows include set/model/horizon/seed/noise replicate/segment/shots and NRMSE, RMSE, R2, MAE,
exact difference and relative inflation. Summaries include mean/median, paired segment-bootstrap
95% CI, relative-inflation p90, contrast-sign preservation, interaction change and the existing
symmetric useful-horizon definition: largest h with mean NRMSE<1 and paired-bootstrap lower
CI(persistence-model)>0. Persistence is read from the frozen gate control rows.

Principal contrast signs are kernel-versus-K0 and kernel-versus-AB interaction signs between
h=2 and h=64, separately in F and Z; S remains reported as the frozen null causal case. A finite
N is globally robust only if median relative NRMSE inflation <=5%, p90 <=10%, and all four F/Z
principal signs are preserved. If a finite N meets criteria globally, classify
`ROBUST_AT_N_SHOTS` at the smallest N. If criteria hold only in some set/horizon strata, classify
`MIXED_SHOT_SENSITIVITY`; otherwise `SHOT_SENSITIVE_UP_TO_10000`. Thresholds cannot change.

Technical verdicts are `COMPLETE`, `INVALID_CONFIG`, `INVALID_BASELINE_REPRODUCTION`,
`INVALID_PROVENANCE` or `INCOMPLETE`.

## Required limits and stop

The experiment adds finite-sample estimation noise to exact observables. It does not simulate
gate decoherence, preparation error, drift, full measurement backaction, or physical storage of
past states. Finite shots are not hardware execution. Gate 3 stops after resource/shot artifacts,
tests, report and verifier. No Stage 4 or manuscript is permitted.
