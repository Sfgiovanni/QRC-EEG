# Pre-registration: Exponential State-Memory Kernel vs. Memory-Mechanism Controls on Bonn EEG

Status: frozen before any model is run on real EEG data. Written after the
mechanism-check suite (`tests/test_mechanism_checks.py`,
`tests/test_constructions_differ.py`, `tests/test_batched_matches_reference.py`)
passed on synthetic input only. Any deviation from this document that occurs
after real data is touched will be reported as a deviation, not silently
absorbed.

## Scope decisions made with the maintainer (2026-07-12)

- **AB**: the only AB implementation that exists in either source repository
  (`QRC-Glicose`, `QRC-Kernel`) is a discrete-delay density-matrix
  approximation (`AB-noaux-residual` / `DiscreteDelayReservoir`), not a
  genuine auxiliary-qubit backflow construction. Checked across full git
  history of both repositories; no genuine version ever existed. Per the
  maintainer: **use the noaux AB as-is**, labeled honestly as such throughout.
- **ABC**: dropped. Not run, not reported.
- **Kernel variants**: in addition to the primary single-exponential and
  dual-exponential kernels, the triangular and uniform kernel-shape ablations
  (already implemented in vendored `state_kernels.py`) are run as secondary
  controls, per the maintainer's explicit request to exercise kernel-shape
  variants.
- **Input-encoding channel and ESN**: neither exists in either repository in
  a form usable on real signals (`QRC-Glicose`'s channels ignore the scalar
  input; there is no ESN class anywhere). Both were built new
  (`src/qrc_eeg/channels.py`, `src/qrc_eeg/esn.py`) as standard,
  well-precedented constructions, and are held **identical across every
  quantum arm** (only `KernelWeights` differs), so they cannot confound the
  memory-mechanism contrast. See `docs/eeg_protocol.md` for the justification
  and equations.

## Hypothesis

The exponential state-memory kernel (single- and dual-exponential) delivers
greater non-linear memory than the alternative QRC construction available
(AB-noaux, kernel-shape ablations) under an equalized compute budget, and
this converts into lower EEG forecasting error, with the size of the
advantage **increasing with the non-linear memory demand of the task**
(gradient Z -> D -> E). The classical ESN is a substrate control, not a
target to beat. This is a claim about the **memory-kernel mechanism within
QRC**, not a quantum-vs-classical claim.

## Data source (amended 2026-07-12, before any real-data run)

The canonical host (`epileptologie-bonn.de` / `meb.uni-bonn.de`) has been
retired and now redirects to the department's generic homepage; the UPF NTSA
mirror and archive.ics.uci.edu both return HTTP 403 to automated fetches. Per
the maintainer, an easier-to-download equivalent source was sought instead of
fighting bot protection. Found: `RYH2077/EEG-Epilepsy-Datasets` on GitHub
mirrors the **exact original** 2001 Andrzejak Bonn release (`A_Z`, `B_O`,
`C_N`, `D_F`, `E_S`, each 100 `.txt` files of 4097 samples), fetched via
`raw.githubusercontent.com` (no bot-blocking), pinned to commit
`185859ab48bc701c9a10f6bb2b5f76d8e28e4003`, SHA256 of the archive
`f4c2dc52fd5320d4404fcbc6ecb9db69a4a7e408df4e3d5456530343dbcb75ad`. This is a
change of *download source* only -- the data itself, its structure, and every
downstream section of this pre-registration are unaffected.

Bonn University EEG database (Andrzejak et al.), sets:

- **Z** (healthy, eyes open) -- low non-linear demand
- **D** (interictal, epileptogenic zone, seizure-free) -- intermediate
- **E** (ictal, active seizure) -- high non-linear demand

100 single-channel segments per set, 4097 samples, 173.6 Hz. Fetched and
hash-verified by `scripts/fetch_eeg.py`; frozen hashes in
`data/eeg/CHECKSUMS.txt` once acquired (Task 5).

## Primary task

h-step-ahead forecasting of the normalized EEG signal, `h in {1, 2, 4, 8}`
steps. Primary metric: **NRMSE** (RMSE normalized by the target's held-out
standard deviation). Secondary: RMSE, R^2, MAE.

## Nonlinear-demand score (frozen formula, x-axis of the main figure)

For each set, computed on the **raw EEG signal only** (not reservoir
output), independent of any reservoir construction, so it cannot leak model
performance into its own x-axis:

1. Delay-embed the signal with order `p` (frozen: `p = 10`) into vectors
   `x_t = [u(t-1), ..., u(t-p)]`.
2. **Linear predictability**: ridge regression of `u(t)` on `x_t`
   (5-fold CV within-set, alpha selected from `logspace(-6, 2, 9)` by inner
   CV), report held-out R^2_linear.
3. **Nonlinear predictability**: ridge regression of `u(t)` on `x_t` expanded
   with all pairwise and squared degree-2 terms
   (`x_i * x_j` for `i <= j`, `p(p+1)/2` extra features), same CV/alpha
   protocol, held-out R^2_quadratic.
4. **Nonlinear demand score**:
   `D = max(0, R^2_quadratic - R^2_linear) / max(R^2_quadratic, epsilon)`,
   `epsilon = 1e-6`. Bounded in `[0, 1]`; `D = 0` means the signal is fully
   explained linearly (quadratic terms add nothing), `D -> 1` means
   quadratic terms account for nearly all explainable variance.

Written to `results/eeg/nonlinear_demand.csv` with one row per set,
per-fold R^2 values, and the aggregate `D`. This score is fixed **a priori**
and is not touched by the choice of reservoir construction.

## Models (equalized budget: same ridge readout, alpha grid, seeds, washout/split)

1. Single-exponential state kernel (4 qubits)
2. Dual-exponential state kernel (4 qubits)
3. AB-noaux discrete-delay reservoir (4 qubits)
4. Triangular / uniform kernel-shape ablations (4 qubits, secondary)
5. ESN (classical substrate control)

All quantum arms share qubit count (4), feature extraction (weight <=2 Pauli
expectation values, 66 features), and the input-encoding channel. Reported
in `tab_quadratic_capacity` alongside feature counts so no reader can mistake
this for a dimension-inflation result.

## Quadratic-capacity linkage

Frozen synthetic quadratic-capacity protocol (vendored `memory_capacity.py`),
5 common seeds, run on every construction, **not used for model selection**.
Regression: EEG advantage (Delta-NRMSE, kernel vs. AB-noaux/ESN) on
quadratic capacity and on the nonlinear-demand score, reporting the
coefficient and its 95% CI.

## Splits, seeds, hyperparameters

- Segment-level split, stratified by set: train/validation/test, saved to
  `data/eeg/splits/`.
- Nested CV over segments: HP selection on inner validation folds only,
  evaluation on all outer folds.
- HP grids (frozen in `config/eeg_frozen.yaml`): kernel `r`/`K`/`past_mass`;
  dual kernel fast/slow weights and scales; AB-noaux `tau`/`delayed_mass`;
  ESN spectral radius / input scale / leak rate / size; ridge
  `alpha = logspace(-8, 2, 11)` for all.
- 10 common seeds for the quantum constructions in the final held-out
  evaluation; seeds are aggregated per segment before cohort summaries (not
  treated as independent samples). HP-search-stage seed count and segment
  subsampling may be reduced from the full held-out set purely for compute
  budget -- **never** the confirmatory held-out evaluation itself. Any such
  reduction is logged in `results/eeg/PROVENANCE.md` with the exact counts
  used.

## Statistics (family: `eeg_primary`)

Per set, per horizon, paired contrasts: kernel-single vs. AB-noaux, vs. ESN;
dual vs. single. 95% bootstrap CI (paired, 10000 resamples),
Wilcoxon signed-rank, Holm correction within family, Cohen's dz, win
fraction. No post-hoc set selection; all three sets and all four horizons
are always reported.

## Amendment: nonlinear-demand score computed on real data (2026-07-12)

Ran the frozen formula above on the real Z/F/S segments (results in
`results/eeg/nonlinear_demand.csv`). Reporting exactly what came out, before
any reservoir model touched this data:

| set | R2_linear | R2_quadratic | nonlinear_demand D |
|---|---|---|---|
| Z | 0.9277 | 0.9279 | 0.00024 |
| F | 0.9743 | 0.9745 | 0.00013 |
| S | 0.9791 | 0.9791 | 0.0 |

Two things, reported as observed, not adjusted:

1. **The quadratic residual is ~0 for every set.** At 173.6 Hz, one-step-ahead
   prediction from 10 lags is dominated by simple continuity; linear terms
   already saturate R^2, leaving essentially nothing for the quadratic terms
   to explain. The frozen `D` score is degenerate as a continuous x-axis --
   it does not discriminate between sets.
2. **R2_linear itself is ordered Z (0.928) < F (0.974) < S (0.979) -- i.e.
   seizure (S) is the *most* linearly predictable set, not the least.** This
   is the opposite of what the pre-registered hypothesis assumes (S = highest
   non-linear demand). Physiologically plausible (rhythmic, high-amplitude
   ictal spiking is smoother sample-to-sample than background activity), but
   it directly contradicts the demand-gradient premise if linear
   predictability is read as an inverse proxy for non-linear demand.

**Per the honesty commitment in this document, the frozen formula is not
being replaced or re-derived now that its output is known** -- doing so
would be exactly the post-hoc metric-picking the pre-registration exists to
prevent, since this score is the x-axis of the primary figure. `D` is
reported in `tab_eeg_endpoints`/`fig_eeg_demand_gradient` exactly as computed
above. Any additional nonlinearity characterization (e.g. sample entropy,
correlation dimension) explored after this point will be labeled
**exploratory / post-hoc** in every table, figure, and caption it appears in,
never substituted for or relabeled as this pre-registered score.

The confirmatory reservoir-vs-reservoir contrasts (kernel vs. AB-noaux/ESN,
per set and horizon) are computed independently of this score and are
unaffected by this finding; they proceed as planned. The continuous
demand-gradient framing of the main figure is now in question and is flagged
to the maintainer rather than silently kept or silently dropped.

## Honesty commitment

If the demand-gradient pattern (advantage near zero on Z, growing through D
to E) does not appear, or the kernel does not lead on set E, this is reported
exactly as observed in `RESULTS.md`, with no adjustment of the hypothesis,
the demand score, or the contrast set after the fact.
