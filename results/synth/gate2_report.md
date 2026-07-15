# Rota A Gate 2 — synthetic theory validation

**Mechanical verdict: SUPPORTED.**

The predictions were frozen from `H_actual` before nonlinear simulation. The external
`W_K(z)R(z)` factorization was not used.

## Frozen aggregate checks

- Aggregate Spearman: 0.990876 (bootstrap 95% CI [0.887295, 0.995096]).
- Median within-scenario Spearman: 0.600000.
- Predicted/measured best-model match fraction: 0.600.
- Caveat: the aggregate coefficient includes between-process slope-scale differences and
  therefore is not an isolated measure of within-process model ordering; the frozen rule
  also requires the within-scenario median and best-model match reported above.

## Per-scenario ordering

| Scenario | Spearman | Predicted best | Measured best |
|---|---:|---|---|
| ar1_phi030 | -0.900 | uniform | QRC_K0 |
| ar1_phi060 | 0.900 | QRC_K0 | QRC_K0 |
| ar1_phi085 | 0.300 | QRC_K0 | QRC_K0 |
| ar1_phi095 | 0.200 | uniform | AB_noaux |
| ar2_rho075_f012 | 0.900 | AB_noaux | AB_noaux |
| ar2_rho090_f012 | 0.900 | AB_noaux | AB_noaux |
| colored_beta05 | 1.000 | AB_noaux | AB_noaux |
| colored_beta15 | 1.000 | uniform | uniform |
| nonlinear_ar1_phi085 | -0.800 | single_kernel | AB_noaux |
| phase_surrogate_nonlinear_ar1_phi085 | 0.300 | QRC_K0 | AB_noaux |

## Phase-surrogate diagnostic

Measured slope change (surrogate minus higher-order source):
- `AB_noaux`: +0.000453.
- `QRC_K0`: +0.000203.
- `single_kernel`: +0.000142.
- `triangular`: +0.000275.
- `uniform`: +0.000108.

## Interpretation and limits

SUPPORTED is the mechanical frozen classification, not uniform agreement. Negative
within-scenario correlations are explicit failures of the local ordering prediction.
The verdict applies only to the frozen processes, amplitudes, channel seeds and linear readout.
Agreement supports local spectral/dynamical prediction; disagreement marks nonlinear,
observability or finite-sample effects not captured by the tangent theory. This is not a
claim of quantum advantage or universal superiority.

Stage 2 stops here. No EEG rerun,
shots, physical-resource analysis or manuscript work was performed.
