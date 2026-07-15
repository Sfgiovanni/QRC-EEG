# EEG result directory

Canonical current evidence is indexed in `results/ARTIFACT_INDEX.csv`. Principal files include
`gate_interactions.csv`, `gate_nrmse_curves.csv`, `useful_horizon_v2.csv`, Gate 1 theory checks,
and Gate 3 shot results.

- `_prefix_snapshot_*`: preserved before-state snapshots; not current headlines.
- `_invalid_config_r09_snapshot/`: valid exploratory computation for r=0.9 but formally
  `INVALID_CONFIG`; never use as confirmatory evidence.
- `raw/`: segment-level generated rows used by summaries.
- run logs and earlier overnight/fase1 outputs: legacy audit material.

The statistical unit is a segment, not a patient.
