# EEG causal-memory gate preregistration

Status: **FROZEN BEFORE any gate model was evaluated** (2026-07-13, America/Sao_Paulo).

## Scope and estimand

The primary estimand is not mean NRMSE. It is the model-by-horizon interaction: whether the
held-out NRMSE curve of `single_kernel` degrades more slowly with forecast horizon than the
curves of causal controls and classical baselines. All preprocessing, train/validation/test
segment IDs, causal train-only scaling, per-segment washout, confirmatory seeds 1--10, and
segment-blocked hyperparameter selection remain frozen from the corrected pipeline.

Horizons are `h = {1, 2, 4, 8, 16, 32, 64, 128}` samples. At `fs = 173.61 Hz`, milliseconds
are computed mechanically as `1000 h / fs`. Reservoir features are generated once per
model/set/seed and reused for every horizon.

## Models

- Target model: `single_kernel`, with its already-selected exponential state-memory kernel.
- Causal control: `QRC_K0`, the same quantum channel, initial state, observables and seed, but
  with present-state mass 1 and no delayed-state mixture (`K=0`).
- Classical controls: persistence; ridge AR(p), with `p` selected from
  `{1, 2, 4, 8, 15, 25, 50}`; degree-2 NVAR using the selected AR lag window; and
  `tapped_delay`, whose classical input history uses the exact selected `single_kernel`
  exponential weights, lag window and past mass. Ridge alpha is selected from the frozen grid
  on whole disjoint validation segments. Persistence has no fitted hyperparameter.

All models are evaluated on the same frozen test segments. Deterministic classical models are
replicated across seeds 1--10 solely to preserve the paired output schema; inferential pairing
first averages seeds within each test segment, so replication does not inflate sample size.

## Frozen degradation contrast

Primary endpoints are `h_short = 2` samples (11.520 ms) and `h_long = 64` samples
(368.642 ms). Horizon 1 and 128 remain in all curves and useful-horizon calculations but are
not substituted into the primary contrast.

For model `m`, set `s`, and test segment `i`, define

`D(m,s,i) = NRMSE(m,s,h_long,i) - NRMSE(m,s,h_short,i)`.

For comparator `c`, the paired interaction is

`I(kernel,c,s,i) = D(c,s,i) - D(single_kernel,s,i)`.

Positive values mean the kernel degrades more slowly. The reported estimate is the segment
mean of `I`; the 95% percentile bootstrap resamples the 20 paired test segments 10,000 times.
The two-sided paired Wilcoxon p-value is Holm-adjusted across the complete frozen `eeg_gate`
family: 5 comparators (`QRC_K0`, `AR`, `NVAR2`, `persistence`, `tapped_delay`) times 3 sets,
for 15 tests. A contrast is significant in the expected direction only when `I > 0`, the
bootstrap lower bound is above 0, and Holm-adjusted `p < 0.05`.

The strongest classical comparator in a set is defined conservatively after evaluation as the
member of `{AR, NVAR2, tapped_delay}` with the smallest observed degradation `D` (the hardest
classical curve for the kernel to beat). Persistence is tested in the family and reported, but
is not eligible to replace that strongest fitted classical comparator.

## Frozen decision rule

**PASS** requires both F and Z independently to satisfy both conditions:

1. `single_kernel` degrades significantly more slowly than `QRC_K0`, in the expected direction.
2. `single_kernel` degrades significantly more slowly than the strongest classical comparator
   (`AR`, `NVAR2`, or `tapped_delay`), in the expected direction.

**FAIL** otherwise. This explicitly includes a non-significant/wrong-direction K=0 interaction,
or any fitted classical baseline reproducing a degradation curve that the kernel does not beat
significantly. S is always reported but cannot rescue failure in F or Z. No alternative horizon,
one-sided test, family definition, comparator selection, or mean-NRMSE framing may replace this
rule after results are observed. Any implementation deviation is listed in the gate report.

## Useful horizon

For each model and set, the useful horizon is the largest evaluated `h` for which its mean NRMSE
is below 1 and below both persistence and AR, with a paired 95% bootstrap confidence interval for
each improvement wholly above 0. If no horizon qualifies, it is reported as missing. For AR
itself, the second comparison is equality by definition, so AR has no useful horizon under this
strict rule; this is reported rather than silently redefining the endpoint. Persistence likewise
cannot be strictly better than itself.

## Deviations

None at freeze time. Any later deviation must be appended here and repeated in
`results/eeg/gate_report.md`; the original rule above remains the basis of PASS/FAIL.
