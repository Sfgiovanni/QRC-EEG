# EEG Results

These results use causal training-only normalization and segment-blocked ridge/HP validation. All numeric statements below are regenerated from the result CSVs.

## Bottom line

Performance depends on forecast horizon. In the matched 66-feature comparison on F and Z, the expected ESN-short/kernel-long direction occurs in 6/8 cells; Holm-significant cells in that direction are 3 at short horizons and 3 at long horizons. This is a horizon interaction, not a claim of overall superiority or generic competitiveness.

Against AB-noaux, the kernel has lower NRMSE in 11/12 set-by-horizon cells, with 8/12 Holm-significant kernel-favoring cells.

The quadratic iid measure used here showed no detectable positive association with EEG gains in the evaluated configurations. The descriptive comparison-level slope is -0.005995 with n=3 independent capacity gaps; no inferential confidence interval is reported.

Corrected quadratic-capacity means are: AB_noaux=1.9296, single_kernel=2.6151, dual_kernel=2.5051, triangular=2.6154, uniform=2.5878, ESN=0.0087.

## Scope limitation

The statistical unit is the held-out segment, not the patient. The Bonn distribution randomized and withheld the segment-to-subject mapping, so there is no subject-separated evaluation and no claim of generalization between patients. Z is surface EEG; F and S are intracranial EEG. Inference is limited to benchmark segments.

## Outputs

- `results/eeg/tab_eeg_endpoints.csv`
- `results/eeg/tab_eeg_contrasts.csv`
- `results/eeg/tab_esn_matched.csv`
- `results/eeg/quadratic_capacity.csv`
- `results/eeg/fase1_diff_report.md`

## Reproduction

```bash
bash scripts/run_eeg.sh
```
