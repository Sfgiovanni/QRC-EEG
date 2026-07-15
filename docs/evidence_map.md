# Evidence map

| Claim | Primary evidence | Independent/diagnostic evidence | Status |
|---|---|---|---|
| C1–C2 local mechanism | `theory_vs_sim_check.csv`, response NPZ | symbolic record, companion spectrum, amplitude sweep | confirmatory local |
| C3 synthetic prediction | frozen prediction/measured CSV, Gate 2 metadata | centered post-gate analysis and LOSO | `SUPPORTED` with moderate within-scenario ordering |
| C4 EEG horizon dependence | `gate_interactions.csv` | NRMSE curves and useful-horizon v2 | F/Z supported |
| C5 S-null | S kernel-vs-K0 row in `gate_interactions.csv` | shot contrast instability in S | null |
| C6 kernel shape | distributed-shape curves | useful horizons | tied/partial; not exponential superiority |
| C7 finite shots | raw, summary, strata and contrast CSVs | tail decomposition | `MIXED_SHOT_SENSITIVITY` |

Exact permitted and prohibited wording is normative in `docs/claims_registry.md` and
`results/final/claims_registry.csv`. Numeric values belong in `results/final/key_results.csv`.
