# Canonical results summary

Generated from the canonical CSV/JSON sources; values are not manually transcribed.

- Gate 1: `FAIL_SEPARABLE_FACTORIZATION`; companion radius `0.958724`.
- Gate 2: `SUPPORTED`; aggregate Spearman `0.990876`, median within-scenario `0.60`, best-model match `60%`.
- Gate 3: `MIXED_SHOT_SENSITIVITY`; `66/120` set×horizon strata pass, with no globally passing shot level.
- EEG: the primary interaction is supported in F and Z; S is null. All models exceed mean NRMSE 1 at h=64.
- EEG follow-up (additive, not part of the canonical gate): a classical distributed-memory ESN
  control (`ESN66_kernel` vs `ESN66_K0`) reproduces the F/Z slower-degradation pattern found in QRC
  (`single_kernel` vs `QRC_K0`) and is null in S under Holm correction, consistent with a mechanism
  generic to distributed recurrent mixing rather than substrate-specific (C8). A new comparison
  against concentrated single-lag delay (`AB_noaux`/`ESN66_AB`), not part of the frozen `eeg_gate`
  family, does **not** favor the distributed exponential kernel in either substrate under the same
  h=2→h=64 metric (C9). A segment×seed crossed-bootstrap sensitivity check of the F/Z findings holds
  up except for one marginal classical-ESN effect in S (C10). See
  `results/eeg/followup/technical_report.md` for full numbers and
  `docs/classical_distributed_memory_protocol.md` / `docs/crossed_inference_protocol.md` for the
  frozen protocols.

Machine-readable values: `results/final/key_results.csv` and `results/final/key_results.json`.
