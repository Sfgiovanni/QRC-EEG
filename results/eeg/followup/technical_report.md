# Follow-up technical report: classical distributed-memory ESN control and crossed segment x seed inference

Additive follow-up to the canonical QRC-EEG repository. Does not modify any canonical Gate/Gate1B/Gate2/Gate3 artifact, `docs/claims_registry.md`, or `results/eeg/gate_interactions.csv`. Frozen protocol: `docs/classical_distributed_memory_protocol.md`, `docs/crossed_inference_protocol.md`, `config/esn_distributed_memory_frozen.yaml` (hashes in `results/eeg/followup/PROTOCOL_HASHES.sha256`).

## 1. Does K0 reproduce the existing ESN-66?

**YES.** Compared 4800 matched (set, horizon, seed, segment) cells between `ESN66_K0`/`fixed_core` and the existing committed `ESN_66` arm (`results/eeg/raw/eeg_holdout_esn66_by_segment_seed.csv`). Max absolute NRMSE difference = `0.000e+00` (unit test tolerance `tests/test_esn_distributed_memory.py::test_k0_reproduces_existing_esn_implementation` uses `1e-10` on the underlying feature trajectories directly; this end-to-end comparison additionally passes through the ridge readout fit, hence the slightly looser but still effectively-exact tolerance here).

## 2-3. Does distributed memory change the degradation curve in the ESN, and does it beat concentrated delay?

`D(construction) = NRMSE(h=64) - NRMSE(h=2)`, per mode/set (lower magnitude = slower degradation = the direction expected of the mechanism if it is generic, not substrate-specific).

- fixed_core/F: D(K0)=+0.7904, D(AB)=+0.7459, D(kernel)=+0.7492 (kernel < K0: slower degradation; kernel >= AB: does not beat concentrated delay)
- fixed_core/S: D(K0)=+0.6425, D(AB)=+0.5569, D(kernel)=+0.6339 (kernel < K0: slower degradation; kernel >= AB: does not beat concentrated delay)
- fixed_core/Z: D(K0)=+0.6564, D(AB)=+0.6036, D(kernel)=+0.6337 (kernel < K0: slower degradation; kernel >= AB: does not beat concentrated delay)
- retuned_core/F: D(K0)=+0.7904, D(AB)=+0.7459, D(kernel)=+0.7492 (kernel < K0: slower degradation; kernel >= AB: does not beat concentrated delay)
- retuned_core/S: D(K0)=+0.6425, D(AB)=+0.5569, D(kernel)=+0.6339 (kernel < K0: slower degradation; kernel >= AB: does not beat concentrated delay)
- retuned_core/Z: D(K0)=+0.6564, D(AB)=+0.6036, D(kernel)=+0.6337 (kernel < K0: slower degradation; kernel >= AB: does not beat concentrated delay)

## 4. Does the result depend on fixed vs. retuned core HP?

All three arms' independently retuned core HP matched the fixed-core (existing ESN-66) HP exactly, so the fixed_core and retuned_core analyses coincide in this run -- the conclusion does not depend on which mode is used.

| construction | fixed_core_esn_hp | retuned_core_esn_hp | same_hp |
|---|---|---|---|
| ESN66_K0 | {"n_reservoir": 66, "spectral_radius": 0.5, "input_scale": 1.0, "leak_rate": 0.7} | {"n_reservoir": 66, "spectral_radius": 0.5, "input_scale": 1.0, "leak_rate": 0.7} | True |
| ESN66_AB | {"n_reservoir": 66, "spectral_radius": 0.5, "input_scale": 1.0, "leak_rate": 0.7} | {"n_reservoir": 66, "spectral_radius": 0.5, "input_scale": 1.0, "leak_rate": 0.7} | True |
| ESN66_kernel | {"n_reservoir": 66, "spectral_radius": 0.5, "input_scale": 1.0, "leak_rate": 0.7} | {"n_reservoir": 66, "spectral_radius": 0.5, "input_scale": 1.0, "leak_rate": 0.7} | True |

## 5. Do the original F/Z/S conclusions survive the segment x seed bootstrap?

Of 21 tests in the `eeg_followup_crossed_sensitivity` family, 10 were significant in the original-style (seed-averaged) replication at the raw 95% CI level; of those, 8 remained significant under the crossed bootstrap (Holm-adjusted) and 2 did not.

Weakened under the crossed design:
| set | kernel | comparator | analysis_mode | bootstrap_mean_crossed | ci95_lo_crossed | ci95_hi_crossed | p_holm |
|---|---|---|---|---|---|---|---|
| S | ESN66_kernel | ESN66_K0 | fixed_core | 0.0087 | 0.0007 | 0.0162 | 0.1944 |
| S | ESN66_kernel | ESN66_K0 | retuned_core | 0.0086 | 0.0007 | 0.0160 | 0.1944 |

## 6. Does the mixed model converge and agree with the bootstrap?

18/21 cells converged; 21/21 flagged a variance-component boundary (near-zero segment or seed variance, or a solver boundary warning); 17/21 had a non-finite/singular covariance matrix; 0/21 raised an exception during fitting (all diagnostics recorded verbatim, never hidden, in `mixed_model_diagnostics.json`). Where a finite point estimate was obtained, its sign agreed with the crossed bootstrap's sign in 21/21 cells. Per the frozen protocol, the crossed bootstrap (Section 3) remains the primary sensitivity analysis regardless of mixed-model convergence; statsmodels' crossed-random-effects variance-components approximation is used because no R/lme4 is available in this environment (documented in `docs/crossed_inference_protocol.md` Section 5).

## 7. Does any paper claim need to be reduced or reformulated?

13/21 tests did not meet the expected-direction + CI-excludes-zero + Holm-p<0.05 condition under the crossed bootstrap:

| set | kernel | comparator | analysis_mode | bootstrap_mean | ci95_lo | ci95_hi | p_holm |
|---|---|---|---|---|---|---|---|
| Z | single_kernel | AB_noaux | not_applicable | -0.0443 | -0.0527 | -0.0363 | 0.0000 |
| Z | ESN66_kernel | ESN66_AB | fixed_core | -0.0300 | -0.0378 | -0.0222 | 0.0000 |
| Z | ESN66_kernel | ESN66_AB | retuned_core | -0.0301 | -0.0375 | -0.0224 | 0.0000 |
| F | single_kernel | AB_noaux | not_applicable | -0.0148 | -0.0207 | -0.0092 | 0.0000 |
| F | ESN66_kernel | ESN66_AB | fixed_core | -0.0033 | -0.0113 | 0.0055 | 1.0000 |
| F | ESN66_kernel | ESN66_AB | retuned_core | -0.0034 | -0.0115 | 0.0056 | 1.0000 |
| S | single_kernel | QRC_K0 | not_applicable | -0.0031 | -0.0108 | 0.0044 | 1.0000 |
| S | single_kernel | AB_noaux | not_applicable | -0.0881 | -0.1007 | -0.0753 | 0.0000 |
| S | single_kernel | ESN_66 | not_applicable | -0.0075 | -0.0199 | 0.0035 | 0.7648 |
| S | ESN66_kernel | ESN66_K0 | fixed_core | 0.0087 | 0.0007 | 0.0162 | 0.1944 |
| S | ESN66_kernel | ESN66_K0 | retuned_core | 0.0086 | 0.0007 | 0.0160 | 0.1944 |
| S | ESN66_kernel | ESN66_AB | fixed_core | -0.0769 | -0.0894 | -0.0635 | 0.0000 |
| S | ESN66_kernel | ESN66_AB | retuned_core | -0.0770 | -0.0898 | -0.0634 | 0.0000 |

Per `docs/crossed_inference_protocol.md` Section 9, any canonical claim resting on these specific cells should be reported with reduced strength; claims C1-C7 in `docs/claims_registry.md` are otherwise based on the canonical seed-averaged gate analysis, which is untouched by this sensitivity check.

## 8. Files feeding the new paper tables/figures

- `results/eeg/followup/classical_control/tab_classical_distributed_memory.csv` -- aggregated NRMSE by construction x mode x set x horizon.
- `results/eeg/followup/classical_control/tab_resource_accounting.csv` -- parameter/memory/op accounting.
- `results/eeg/followup/crossed_inference/crossed_bootstrap.csv` -- primary sensitivity endpoint, 21-test family, Holm-corrected.
- `results/eeg/followup/crossed_inference/original_style_replication.csv` -- side-by-side canonical-style replication (comparison only).
- `results/eeg/followup/crossed_inference/mixed_model_results.csv` / `mixed_model_diagnostics.json` -- secondary verification + diagnostics.
- `figures/eeg/fig_classical_distributed_memory.{pdf,png}`.
- `figures/eeg/fig_crossed_inference.{pdf,png}`.
- `results/eeg/followup/metadata.json` -- commit/dependency/OS/timing/hash provenance.

## Reproduction

```bash
.venv/bin/python scripts/run_esn_distributed_memory_hp_search.py
.venv/bin/python scripts/run_esn_distributed_memory_holdout.py
.venv/bin/python scripts/make_classical_distributed_memory_figure.py
.venv/bin/python scripts/run_crossed_inference.py
.venv/bin/python scripts/make_crossed_inference_figure.py
.venv/bin/python scripts/verify_esn_distributed_memory.py
.venv/bin/python scripts/make_followup_metadata.py
.venv/bin/python -m pytest tests/test_esn_distributed_memory.py tests/test_crossed_inference.py -q
```