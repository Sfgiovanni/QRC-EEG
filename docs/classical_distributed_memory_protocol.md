# Classical distributed-memory ESN control: frozen protocol

Status: **frozen before any construction was run** (2026-07-22, America/Sao_Paulo).
This is a confirmatory follow-up experiment, additive to the canonical repository.
It does not modify, rerun, or reinterpret the canonical Gate 1/1B/2/3 artifacts,
`docs/eeg_gate_preregistration.md`, or `config/eeg_frozen.yaml`. All parameters
below are frozen in `config/esn_distributed_memory_frozen.yaml`; its SHA-256 and
this document's SHA-256 are recorded in
`results/eeg/followup/PROTOCOL_HASHES.sha256` before Section 1's holdout runs.
No parameter, endpoint, statistical family, or interpretation rule may change
after that hash is recorded.

## 1. Why this control exists

The published QRC results attribute part of the kernel's advantage to
*distributed* (multi-lag, exponentially-weighted) state memory acting inside
the recurrence, as opposed to *concentrated* single-lag memory (AB-noaux) or
no memory (QRC K=0). The existing classical ESN control
(`src/qrc_eeg/esn.py`, `results/eeg/hp_selected_esn66.json`) has **no**
analogous distributed-memory mechanism -- it mixes nothing before the leaky
integrator. This follow-up asks a strictly mechanistic question: if the
*same* mixing structure used by the QRC arms is grafted onto a classical
leaky-integrator ESN, does the resulting curve-shape advantage (slower
degradation with horizon) also appear? If yes, the mechanism is generic to
distributed recurrent mixing, not specific to the quantum substrate. If it
appears only in the QRC arm, that is evidence of substrate-dependent
manifestation within this protocol -- not evidence of quantum advantage,
which is never claimed.

## 2. Construction

Let `x_t` be the 66-dimensional ESN state (dimension-matched to the 66-feature
quantum readout, as in the existing `ESN_66` control). Define the causal
mixed state

```
m_t = w0 * x_t + sum_{tau=1..K} w_tau * x_{t-tau}
```

using the same `KernelWeights` (`w0=present`, `w_tau=delayed`) construction as
the QRC arms (`src/qrc_eeg/state_kernels.py`, unmodified, reused verbatim).
The mixture enters the recurrence **before** the nonlinearity and before the
leak, exactly mirroring how `run_batched_reservoir` mixes density matrices
before the channel step:

```
x_{t+1} = (1 - a) * m_t + a * tanh(W_res @ m_t + W_in * u_t)
```

No filter is applied after the fact to features or predictions; the mixture
is the only new operation, and it is internal to the recurrence.

### Buffer discipline

- Causal: `m_t` uses only `x_t, x_{t-1}, ..., x_{t-K}`.
- Reinitialized independently at the start of every segment.
- Initial buffer value: 66 zero vectors (`K+1` copies), matching
  `EchoStateNetwork.reset()`'s zero initial state.
- Indexing convention identical to `qrc_eeg.batched.run_batched_reservoir`:
  the buffer holds `K+1` slots oldest-to-newest; at each step
  `mix = present*state + sum_i delayed[i-1]*buffer[-1-i]`, then
  `buffer = concat(buffer[1:], [new_state])`. Verified by direct comparison
  against the quantum buffer's index arithmetic (`tests/test_esn_distributed_memory.py`).
- Never shared across segments (each segment gets a fresh `EchoStateNetwork`-
  equivalent state and buffer, exactly as the existing ESN and QRC arms do).
- The buffer itself is bookkeeping, not a trainable feature; its size (`K+1`
  vectors of 66 floats) is reported in the resource table below, and it costs
  no extra fitted parameters.

## 3. Three constructions

| Construction | Memory | Kernel HP | Source |
|---|---|---|---|
| `ESN66_K0` | none (`w0=1`) | `no_memory_weights()` | must reproduce `ESN_66` numerically |
| `ESN66_AB` | concentrated single-lag | `matched_delay_weights(tau=5, delayed_mass=0.3)` | `hp_selected.json:AB_noaux.hp` (identical between the Gate-1 snapshot and current HEAD -- no divergence) |
| `ESN66_kernel` | distributed exponential | `single_exponential_weights(K=15, r=0.9, past_mass=0.3)` | `hp_selected.json:single_kernel.hp` -- **see divergence note below** |

### HP-provenance divergence (documented, not resolved)

`docs/eeg_gate_preregistration.md` and `config/effective_kernel_gate1_frozen.json`
freeze `single_kernel` at `r=0.7`. The HEAD-committed `results/eeg/hp_selected.json`
-- the file every downstream QRC holdout/gate/contrast script actually reads
(`scripts/run_holdout_eval.py`, `scripts/run_gate_baselines.py`,
`scripts/run_esn66_contrasts.py`) -- instead carries `single_kernel.hp.r = 0.9`.
This is a **pre-existing repository inconsistency**, not introduced by this
follow-up (the repository already carries an explicit
`results/eeg/_invalid_config_r09_snapshot/` marking it as such). Per the
maintainer's instruction for this follow-up specifically: use the officially
committed HEAD value (`r=0.9`) for `ESN66_kernel`, and document the
divergence rather than silently substituting the Gate-1 `r=0.7` value or
silently "fixing" the committed file. This divergence is repeated verbatim in
`results/eeg/followup/technical_report.md`.

## 4. Pairing conditions (identical across the three arms)

- Fixed dimension: `n_reservoir = 66`.
- Identical inputs and causal training-only normalization
  (`qrc_eeg.preprocessing.scale_set_from_training`, unmodified).
- Identical frozen splits (`data/eeg/splits/{set}_split.json`).
- Identical washout (`config/eeg_frozen.yaml:readout.washout = 50`).
- Identical horizons (`config/eeg_frozen.yaml:readout.horizons`).
- Identical ridge readout and alpha grid
  (`qrc_eeg.readout`, `config/eeg_frozen.yaml:readout.alpha_grid`), selected
  by `qrc_eeg.pipeline.fit_readouts_per_horizon` unmodified.
- Identical seeds (`config/eeg_frozen.yaml:channel.confirmatory_seeds` for
  holdout, `hp_search_seeds` for Analysis B).
- For a given seed, exactly one *base* draw (`W_raw` 66x66, `W_in_raw` 66,
  from a single `numpy.random.default_rng(seed)` stream, same order as
  `qrc_eeg.pipeline.batched_esn_features`) is shared, unmodified, across all
  three arms in every analysis mode. `W_res`/`W_in` are then rescaled from
  that shared base by each arm's `(spectral_radius, input_scale)`. In
  Analysis A (fixed core) all three arms use the same rescale, so the final
  `W_res`/`W_in` are bit-identical across arms. In Analysis B (retuned core)
  each arm may select a different `(spectral_radius, input_scale)`, so only
  the base draw -- not the final rescaled matrices -- is guaranteed shared.
  Only the mixing kernel (and, in Analysis B, the rescale) differs between
  arms.
- No additional feature, no additional nonlinearity beyond the shared `tanh`.
- Buffers do not count as trainable features (the ridge readout still sees
  exactly 66 features per timestep, identical to `ESN_66`); their storage
  cost is reported for transparency only.

## 5. Two analyses

### Analysis A -- fixed core (mechanistic isolation)

All three arms reuse the already-selected `ESN_66` core HP
(`results/eeg/hp_selected_esn66.json`: `spectral_radius=0.5, input_scale=1.0,
leak_rate=0.7`). This isolates the effect of substituting `x_t` with `m_t`
inside the recurrence, holding every other reservoir property fixed.

### Analysis B -- paired retuned-budget core

For each arm independently, `(spectral_radius, input_scale, leak_rate)` is
selected from the exact grid already frozen for the ESN in
`config/eeg_frozen.yaml:hp_grids.ESN` (`n_reservoir` overridden to 66, the
independent variable of this experiment -- not searched, mirroring the
already-logged deviation in `scripts/run_esn66_hp_search.py`), using:

- the same `hp_search_seeds` (`[101, 102]`);
- the same train/validation subsample
  (`hp_search_subsample: train_segments_per_set=20, val_segments_per_set=10`,
  pooled across Z/F/S);
- the same aggregated criterion: mean validation NRMSE across all horizons,
  scored with `refit_on_train_validation=False`;
- the same search budget for every arm: `3 spectral_radius x 1 input_scale x
  2 leak_rate = 6` combinations x 2 seeds = 12 fits per arm.

The memory-kernel HP (`ESN66_AB`'s `tau`/`delayed_mass`, `ESN66_kernel`'s
`K`/`r`/`past_mass`) is **never** part of this search -- it stays fixed at the
Section 3 values in both analyses. The held-out test partition is never read
during this search (enforced by `qrc_eeg.pipeline.assert_disjoint_segment_ids`
and by construction -- the HP-search code path only ever receives the
train/val subsample arrays).

Both analyses are run **once**, after HP selection is frozen, over the 10
confirmatory seeds on the frozen held-out test segments. No re-running after
inspecting test-segment results.

## 6. Resource accounting (reported, not optimized)

For each arm and mode:

- **Trainable parameters**: the ridge readout weights only (66 inputs -> 1
  output per horizon, `alpha` chosen from the frozen grid). `W_res`, `W_in`
  are fixed (untrained) after the seeded draw, identical to the existing ESN
  control's convention.
- **Fixed (non-trained) parameters**: `W_res` (66x66), `W_in` (66,), and the
  kernel weights (`K+1` scalars, `K` of them trainable-in-neither-sense --
  they are a frozen HP, not fit by gradient/ridge).
- **Memory**: `(K+1) x 66` floats for the causal buffer, held per active
  segment during evolution (never shared across segments; freed between
  segments).
- **Approximate operations per step**: one `66x66` mat-vec (`O(66^2)`
  multiply-adds) for `W_res @ m_t`, plus `O(K*66)` multiply-adds for the
  mixture, plus `O(66)` for the leak/tanh/`W_in` term -- dominated by the
  `W_res` mat-vec exactly as in the existing ESN control; the kernel mixture
  adds a small, `K`-linear, `K<=15` overhead.

This table is written to
`results/eeg/followup/classical_control/tab_resource_accounting.csv`.

## 7. Mandatory tests (`tests/test_esn_distributed_memory.py`)

- `ESN66_K0` numerically matches the existing `ESN_66` implementation
  (explicit tolerance, `atol=1e-10`).
- The batched implementation matches a non-batched sequential reference
  implementation.
- Kernel weights are non-negative and sum to 1.
- No future state is used (causality check via zeroed-tail perturbation).
- Buffers are independent across segments (perturbing one segment's history
  does not change another segment's features in a batched call).
- The base draw (`W_raw`, `W_in_raw`) is identical across arms for the same
  seed. Final `W_res`/`W_in` are additionally identical across arms in
  Analysis A (fixed core, shared rescale); they are not asserted identical
  in Analysis B (retuned core), where each arm may rescale independently.
- Different seeds produce different reservoirs.
- Features have dimension 66.
- No NaNs or infinities anywhere in features or metrics.
- Train, validation and test segment IDs never overlap.
- The Analysis-B HP search never reads the test partition.
- Every expected `(construction, analysis_mode, set, horizon, seed,
  segment_id)` holdout cell is present exactly once.

If `ESN66_K0` does not reproduce `ESN_66`, the experiment stops and the
divergence is diagnosed before proceeding -- not silently patched over.

## 8. Artifacts

Under `results/eeg/followup/`:

- `PROTOCOL_HASHES.sha256` -- SHA-256 of this document, the crossed-inference
  protocol, and `config/esn_distributed_memory_frozen.yaml`, recorded before
  any holdout run.
- `classical_control/hp_search_log.csv`, `hp_selected.json` (Analysis B).
- `classical_control/fixed_core_hp.json` (Analysis A, documents the reused
  `ESN_66` HP).
- `classical_control/selected_alphas.csv`.
- `classical_control/tab_resource_accounting.csv`.
- `classical_control/tab_classical_distributed_memory.csv` (aggregated
  mean/CI table per construction x mode x set x horizon).
- `raw/esn_distributed_memory_holdout_by_segment_seed.csv` (per
  construction, analysis_mode, set, horizon, seed, segment_id: nrmse, rmse,
  r2, mae).
- `metadata.json` (commit, Python/dependency versions, OS, thread count,
  per-stage wall time).
- `technical_report.md`.

Figures: `figures/eeg/fig_classical_distributed_memory.{pdf,png}`.

## 9. Reproduction

```bash
.venv/bin/python scripts/run_esn_distributed_memory_hp_search.py
.venv/bin/python scripts/run_esn_distributed_memory_holdout.py
.venv/bin/python scripts/make_classical_distributed_memory_figure.py
.venv/bin/python scripts/verify_esn_distributed_memory.py
.venv/bin/python -m pytest tests/test_esn_distributed_memory.py -q
```

## 10. Interpretation rules (frozen, applied mechanically after data is seen)

- If `ESN66_kernel` also shows slower degradation than `ESN66_K0`/`ESN66_AB`,
  the benefit is reported as compatible with a generic distributed-recurrence
  mechanism, not specific to the quantum substrate.
- If the effect appears in the QRC arm but not in the paired ESN, this is
  reported as evidence of substrate-dependent manifestation within this
  protocol, not as quantum advantage.
- A lower absolute NRMSE for the ESN does not invalidate the QRC mechanism
  claim -- absolute skill and mechanism (curve-shape) are reported
  separately, as in the canonical gate report.
- No claim of quantum advantage, clinical utility, patient independence, or
  hardware viability is made anywhere in this follow-up.
- The S-set null result, if it remains null here, is preserved and reported,
  not dropped.

## Deviations

Two corrections made during design review, **before any code was written and
before any HP-search or holdout call was made** (protocol hash recomputed
after both, still before execution):

1. Confirmed by direct recomputation of one held-out cell
   (`single_kernel`, `Z`, `h=1`, `seed=1`, segment `Z005`) that the
   HEAD-committed `results/eeg/raw/eeg_holdout_by_segment_seed.csv` embodies
   `r=0.9` exactly (`0.2640926161359957` reproduced to full float precision
   with `r=0.9`; `r=0.7` gives `0.2633728828823425`, not a match). This makes
   `r=0.9` for `ESN66_kernel` internally consistent with the `single_kernel`
   data used in crossed-inference contrasts 1-3, not merely the value found
   in `hp_selected.json`.
2. The original `shared_reservoir_weights` wording (keyed on
   `(seed, spectral_radius, input_scale)`) was unsatisfiable in Analysis B
   (retuned core), where arms select different `(spectral_radius,
   input_scale)`. Fixed by separating the seed-only base draw from the
   per-arm rescale (Section 4 above); the identity test is scoped to
   Analysis A, and a base-draw-sharing test covers Analysis B.

No parameter value, statistical family, or interpretation rule changed as a
result of either correction.
