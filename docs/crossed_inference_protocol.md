# Crossed segment x seed inference: frozen protocol

Status: **frozen before any bootstrap or mixed-model call was made**
(2026-07-22, America/Sao_Paulo). This is a sensitivity analysis, additive to
the canonical repository. Parameters are frozen in
`config/esn_distributed_memory_frozen.yaml:crossed_inference`; this
document's SHA-256 is recorded alongside the other two frozen artifacts in
`results/eeg/followup/PROTOCOL_HASHES.sha256` before execution. No endpoint,
resampling scheme, family definition, or interpretation rule may change
after that hash is recorded.

## 1. Relationship to the canonical gate analysis

`docs/eeg_gate_preregistration.md` and `scripts/make_gate_report.py` define
the **original, canonical, frozen** analysis: per construction/set/horizon,
NRMSE is first averaged **across the 10 confirmatory seeds within each test
segment** (`per_segment = raw.groupby([...,"segment_id"])["nrmse"].mean()`),
and *then* the 20 test segments are paired-bootstrapped. That analysis is
untouched by this document -- it remains the canonical basis for the
`eeg_gate` PASS/FAIL rule, `gate_interactions.csv`, and every claim in
`docs/claims_registry.md`.

This protocol adds a **sensitivity analysis** that does not collapse the
seed axis before resampling. It asks: does the sign/magnitude/significance
of the model-by-horizon interaction survive when segment-to-segment *and*
seed-to-seed variability are both preserved and jointly resampled, rather
than the seed axis being averaged away first? Seeds are not nested inside
segments -- every seed is evaluated on every segment -- so the design is
**crossed**, and the resampling scheme below treats it as such (independent
resampling of the segment axis and the seed axis, with the bootstrap replica
built from their Cartesian product).

## 2. Primary endpoint

For comparator `c`, set `s`, test segment `i`, confirmatory seed `k`:

```
I(kernel,c,s,i,k) =
    [ NRMSE_c(s, h=64, i, k)      - NRMSE_c(s, h=2, i, k) ]
  - [ NRMSE_kernel(s, h=64, i, k) - NRMSE_kernel(s, h=2, i, k) ]
```

Positive values mean the QRC-kernel arm degrades more slowly from `h=2` to
`h=64` than the comparator. `h_short=2`, `h_long=64` are the same frozen
endpoints as `docs/eeg_gate_preregistration.md`
(`config/eeg_frozen.yaml:eeg_gate.h_short/h_long`); they are not re-chosen
here.

For the deterministic classical comparators reused from the gate family
(none appear in this family -- see contrast list below), the gate
preregistration's convention (seed-replication solely to preserve the paired
schema) would apply verbatim if such a comparator were ever added to this
family; it does not need to be invoked for the five contrasts actually run
here, all of which are seed-varying reservoir constructions.

## 3. Two-factor crossed bootstrap

For each `(comparator, set)` cell:

1. Let `Seg` = the 20 frozen test segment IDs for that set, `Seed` = the 10
   confirmatory seeds (`config/eeg_frozen.yaml:channel.confirmatory_seeds`).
2. For `n_bootstrap_replicates = 10000` replicates (`config/esn_distributed_memory_frozen.yaml:crossed_inference.n_bootstrap_replicates`),
   using RNG seed `bootstrap_rng_seed = 20260722`
   (`numpy.random.default_rng(20260722)`, one stream shared sequentially
   across all cells in a fixed, logged iteration order so the run is exactly
   reproducible):
   a. draw `|Seg|` segment IDs from `Seg` **with replacement**;
   b. draw `|Seed|` seed values from `Seed` **with replacement**,
      independently of (a);
   c. form the replica's sample as the Cartesian product of the two draws
      (`|Seg| x |Seed|` paired `I(kernel,c,s,i,k)` values, with repeats
      counted with multiplicity from both axes);
   d. the replica statistic is the mean of `I` over that Cartesian-product
      sample.
3. Report: bootstrap mean, bootstrap median, 95% percentile CI (2.5/97.5),
   bootstrap standard error (`std` of the replica statistics), and the
   fraction of replicates with the expected sign (`I > 0`).
4. Two-sided bootstrap `p` (sensitivity statistic only, never substituted for
   the canonical Wilcoxon test): `2 * min(mean(replica <= 0), mean(replica >= 0))`,
   clipped to `[0, 1]`.

The pairing between constructions and horizons is preserved throughout
(every replica differences the *same* resampled `(segment, seed)` cell's
`h=64` and `h=2` values for both the kernel and the comparator before taking
`comp_d - kernel_d`).

Seeds are never treated as nested inside segments (i.e., never resampled
*conditional on* the segment draw) -- the two axes are resampled
independently, consistent with the crossed design where the full seed set is
observed on every segment.

## 4. Contrasts and family

Explicit family `eeg_followup_crossed_sensitivity`
(`config/esn_distributed_memory_frozen.yaml:crossed_inference.family_name`),
kept entirely separate from `eeg_primary` and `eeg_gate`; Holm correction is
computed fresh within this family only, never merged with the canonical
families' `p_holm` columns.

| # | Kernel | Comparator | Modes | Data source |
|---|---|---|---|---|
| 1 | `single_kernel` | `QRC_K0` | -- | `results/eeg/raw/eeg_holdout_by_segment_seed.csv` |
| 2 | `single_kernel` | `AB_noaux` | -- | same |
| 3 | `single_kernel` | `ESN_66` (existing) | -- | `results/eeg/raw/eeg_holdout_esn66_by_segment_seed.csv` |
| 4 | `ESN66_kernel` | `ESN66_K0` | `fixed_core`, `retuned_core` | `results/eeg/followup/raw/esn_distributed_memory_holdout_by_segment_seed.csv` |
| 5 | `ESN66_kernel` | `ESN66_AB` | `fixed_core`, `retuned_core` | same |

Run separately for Z, F, S: `(3 contrasts x 1) + (2 contrasts x 2 modes) = 7`
cells per set, `21` tests total in the family. `p_bootstrap` (two-sided) is
Holm-adjusted within this 21-test family. The Wilcoxon test is **not**
recomputed under the crossed design (the canonical Wilcoxon in `gate_interactions.csv`
already covers the seed-averaged design); the bootstrap `p` above is
reported as a sensitivity statistic, explicitly labeled as such in every
table.

## 5. Secondary verification: crossed mixed model

For `h in {2, 64}` only, per `(set, comparison)`, attempt

```
NRMSE ~ construction * horizon + (1 | segment_id) + (1 | seed)
```

a fully crossed random-intercept model (segment and seed as independent
crossed grouping factors, not nested). No R/`lme4` is available in this
environment; the fitting engine is `statsmodels.regression.mixed_linear_model.MixedLM`
using the documented crossed-random-effects variance-components
reformulation (single dummy `groups` column, `vc_formula={"segment": "0 +
C(segment_id)", "seed": "0 + C(seed)"}`, REML). This is an approximation to a
true crossed-random-intercept model (no cross term between the two variance
components beyond their independent diagonal contribution), logged as such,
not presented as an exact `lme4`-equivalent fit.

For each `(set, comparison)` cell the fit is attempted independently. The
interaction term of interest is the `construction:horizon` coefficient
(kernel vs. comparator, `h=64` vs. `h=2`).

**If the model fails to converge, returns a singular covariance matrix, or
any variance component is estimated at its boundary (~0)**, this is recorded
verbatim in `results/eeg/followup/crossed_inference/mixed_model_diagnostics.json`
(per-cell `converged`, `warnings`, `boundary_hit` flags) and the cell's point
estimate/CI from the model is reported alongside the diagnostic flag rather
than hidden. The crossed bootstrap (Section 3) remains the primary
sensitivity analysis regardless of mixed-model outcome; a non-converging or
boundary-hit mixed model is never silently swapped for an alternative,
more favorable model.

## 6. Scope and limitations (restated, not new)

The Bonn segment-to-subject mapping is randomized and unavailable
(`docs/eeg_protocol.md`); segments are never interpreted as independent
patients in this analysis either. This crossed design adds seed-level
resampling, which is a genuine second source of variability under this
protocol's own construction (the channel-seeded unitary), but it does not
create or imply subject-level independence.

## 7. Reported outputs

Under `results/eeg/followup/crossed_inference/`:

- `crossed_bootstrap.csv` -- one row per `(comparison, mode, set)`: mean,
  median, ci95_lo, ci95_hi, se, sign_fraction, p_bootstrap, p_holm.
- `original_style_replication.csv` -- the canonical seed-averaged-then-segment-
  bootstrap analysis (Section 1's original scheme), recomputed for the same
  21 cells so Section 4 of the technical report can show both side by side.
  This is a **replication for comparison**, not a modification of the
  canonical `gate_interactions.csv`, which is left untouched.
- `mixed_model_results.csv` -- per-cell fixed-effect interaction estimate,
  CI, p-value, where available.
- `mixed_model_diagnostics.json` -- per-cell convergence/singularity/boundary
  flags and raw solver warnings.

## 8. Reproduction

```bash
.venv/bin/python scripts/run_crossed_inference.py
.venv/bin/python scripts/make_crossed_inference_figure.py
.venv/bin/python -m pytest tests/test_crossed_inference.py -q
```

## 9. Interpretation rules (frozen, applied mechanically after data is seen)

- If the crossed bootstrap weakens (CI crosses zero, or Holm-adjusted
  `p_bootstrap >= 0.05`) an interaction that was significant under the
  canonical seed-averaged analysis, the corresponding claim's strength is
  reduced accordingly in `docs/claims_registry.md` and the technical report
  -- not defended by switching endpoints.
- If the result stays stable (same sign, CI excludes zero, comparable
  magnitude) under the crossed design, it is reported as robust to joint
  segment x seed variation.
- The S-set null result, if it remains null here, is preserved and
  highlighted, not dropped.

## Deviations

None at freeze time. Any deviation discovered during execution is appended
here and repeated in `results/eeg/followup/technical_report.md`.
