# Canonical CSV table inventory

No LaTeX table is generated in this stage. All files below are rebuilt by
`scripts/build_repository_release.py data` from frozen sources.

| CSV | Purpose | Canonical source |
|---|---|---|
| `results/final/table_eeg_interactions.csv` | Frozen model×horizon contrasts, CIs and Holm values | `results/eeg/gate_interactions.csv` |
| `results/final/table_useful_horizon.csv` | Symmetric absolute-skill/useful-horizon criterion | `results/eeg/useful_horizon_v2.csv` |
| `results/final/table_synthetic_validation.csv` | Predicted/measured synthetic slopes and uncertainty | `results/synth/theory_predictions_vs_measured.csv` |
| `results/final/table_physical_resources.csv` | Density-matrix simulator and measurement-resource counts | `results/resources/qrc_resource_table.csv` |
| `results/final/table_shot_sensitivity.csv` | Shot inflation by set, model, horizon and shot count | `results/eeg/shot_sensitivity_by_stratum.csv` |
| `results/final/table_negative_null_results.csv` | Required falsifications, nulls and failure cases | Gate 1–3 canonical results |

The future author should select columns from these CSVs rather than transcribing values from
Markdown. The CSVs are evidence inputs, not manuscript tables.
