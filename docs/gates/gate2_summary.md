# Gate 2 — synthetic validation

## Scientific question

Do predictions frozen from `H_actual` anticipate degradation differences across and within synthetic processes?

## Inputs, scripts and artifacts

Ten frozen scenarios and five models; `scripts/run_synthetic_stage2.py` and analytical-only `scripts/make_gate2_postgate_addendum.py`. Sources: `results/synth/theory_predictions_vs_measured.csv` (SHA256 `a67c77f0bca210eabbf627759af3047c2f657b5cb065506423c28b8b0bc4008d`) and `results/synth/gate2_postgate_sensitivity.csv` (SHA256 `e3e9a5c8842ab1ea8b86918fd5a90734eb9ab9c26247d8948505ea10adc893fc`).

## Metrics and verdicts

Aggregate Spearman=0.990876 (95% CI [0.887295, 0.995096]); median within-scenario Spearman=0.60; predicted best matches 6/10. Technical verdict: **PASS**. Scientific/mechanical verdict: **SUPPORTED**.

## Limitations

Aggregate association partly reflects between-process scale. Within-process ordering is moderate; ar1_phi030 and nonlinear_ar1_phi085 are explicit negative-ordering failures. No universal T_eff-to-slope law. The post-gate addendum does not alter the freeze. Commit: `6b4b4ea68fd040d29729d5a8405476e14e15fd69`. Status: **canonical**.
