# Corrected EEG results

The empirical claim is mechanistic: distributed state-history changes the slope of error degradation with horizon. There is no claim of quantum advantage or absolute forecasting superiority; classical models lead at short horizons, and all evaluated models have mean NRMSE above 1 at h=64.

## Useful horizon (symmetric v2)

Useful horizon is the largest h with mean NRMSE below 1 and a paired-bootstrap lower confidence bound above zero for the improvement over persistence. The same criterion is applied to every model; persistence is NA because its self-difference is zero.

| construction | set | useful_horizon | useful_horizon_ms | nrmse_at_useful_horizon | persistence_improvement_ci95_lo |
|---|---|---|---|---|---|
| AR | F | 8.000 | 46.080 | 0.941 | 0.135 |
| NVAR2 | F | 8.000 | 46.080 | 0.944 | 0.121 |
| QRC_K0 | F | 8.000 | 46.080 | 0.928 | 0.147 |
| persistence | F | NA | NA | NA | NA |
| single_kernel | F | 8.000 | 46.080 | 0.910 | 0.165 |
| tapped_delay | F | 8.000 | 46.080 | 0.943 | 0.137 |
| AR | S | 16.000 | 92.161 | 0.976 | 0.495 |
| NVAR2 | S | 32.000 | 184.321 | 1.000 | 0.241 |
| QRC_K0 | S | 32.000 | 184.321 | 0.998 | 0.241 |
| persistence | S | NA | NA | NA | NA |
| single_kernel | S | 32.000 | 184.321 | 0.995 | 0.245 |
| tapped_delay | S | 32.000 | 184.321 | 1.000 | 0.240 |
| AR | Z | 8.000 | 46.080 | 0.972 | 0.236 |
| NVAR2 | Z | 8.000 | 46.080 | 0.972 | 0.236 |
| QRC_K0 | Z | 16.000 | 92.161 | 0.997 | 0.181 |
| persistence | Z | NA | NA | NA | NA |
| single_kernel | Z | 16.000 | 92.161 | 0.987 | 0.190 |
| tapped_delay | Z | 8.000 | 46.080 | 0.988 | 0.215 |

In F, the kernel and fitted classical baselines all stop at h=8 (46.080 ms). In Z, the kernel reaches h=16 (92.161 ms), while AR, NVAR2 and tapped-delay stop at h=8; QRC K=0 also reaches h=16. In S, the kernel, K=0, NVAR2 and tapped-delay reach h=32 (184.321 ms). Thus useful horizon does not establish a uniquely quantum or uniquely exponential advantage.

## Preregistered causal-memory interaction

The frozen gate verdict is **PASS**. Its primary endpoint is the model-by-horizon interaction from h=2 to h=64, not the h=64 endpoint itself.

- Set F: kernel-vs-K=0 interaction=+0.036805 (95% CI [+0.019668, +0.056495], Holm p=0.000574112; condition=PASS). Strongest fitted classical comparator=NVAR2, interaction=+0.146532 (95% CI [+0.080097, +0.223691], Holm p=7.43866e-05; condition=PASS).
- Set Z: kernel-vs-K=0 interaction=+0.031875 (95% CI [+0.015703, +0.051662], Holm p=0.000114441; condition=PASS). Strongest fitted classical comparator=AR, interaction=+0.085064 (95% CI [+0.046161, +0.133147], Holm p=0.000967026; condition=PASS).

S is null in the causal test: kernel-vs-K=0 interaction=-0.003142 (95% CI [-0.009171, +0.002987], Holm p=0.245487). The effect is confined to F and Z.

Single, dual, triangular and uniform distributed kernels have closely overlapping degradation curves and the same useful horizons in F/Z/S. The supported interpretation concerns the distributed-memory class, not the exponential shape specifically.

## Long-horizon endpoint

At h=64 the kernel mean NRMSE is 1.035460 in F, 1.080465 in Z and 1.007275 in S. All exceed 1, so there is no absolute skill claim at this endpoint; h=64 is used only as the frozen long endpoint of the interaction.

## Quadratic capacity

The iid quadratic-capacity gap regression has slope +0.006034 with n=3; no valid interval is claimed from three capacity-gap values. The iid quadratic measure used here showed no detectable positive association with EEG gains in the evaluated configurations.

## Scope

Inference is limited to benchmark segments from one EEG database. Bonn segment-to-subject mapping is randomized/unavailable, there is no subject-disjoint split, and no between-patient or cross-dataset generalization is claimed. Z is scalp EEG; F and S are intracranial recordings.

## Reproduction

The complete corrected empirical pipeline is:

```bash
bash scripts/run_eeg.sh
```

The Stage 0 integrity refresh is:

```bash
bash scripts/run_rotaA_stage0.sh
```

The original empirical gate is frozen in `results/eeg/gate_report.md`; the corrected symmetric headline table is `results/eeg/useful_horizon_v2.csv`.
