# Gate 0 — corrected EEG pipeline

## Scientific question

Does distributed state history change the model-by-horizon degradation curve under causal preprocessing, segment-blocked HP selection and held-out segment evaluation?

## Configuration and inputs

Frozen Bonn Z/F/S splits; h={1,2,4,8,16,32,64,128}; 10 channel seeds; segment is the statistical unit. Inputs: `config/eeg_frozen.yaml` (SHA256 `3df99491cc57c661b3702d2000711ae3863e957a7c061e41b4f7e838b2f86842`), `results/eeg/gate_interactions.csv` (SHA256 `83cfc142cce77cd4e77c35f6630ab23e1bf3352e1d3821c3567d41cd3dd70366`).

## Scripts and artifacts

`scripts/run_eeg.sh`, `scripts/make_gate_report.py`, `scripts/make_useful_horizon_v2.py`; canonical outputs are `gate_interactions.csv`, `gate_nrmse_curves.csv`, `useful_horizon_v2.csv`, `gate_report.md` and `RESULTS.md`.

## Metrics and verdicts

Primary metric: comparator-minus-kernel change in NRMSE degradation from h=2 to h=64, paired over 20 test segments with bootstrap CI and Holm correction. F/Z kernel-vs-K0 conditions pass (Z=0.031875, F=0.036805); S is null (`-0.003142`, CI [-0.009171, 0.002987]). Technical verdict: **PASS**. Scientific verdict: horizon dependence supported in F/Z, null in S.

## Limitations

One Bonn database; randomized/unavailable subject mapping; no subject-disjoint or clinical generalization; h=64 is an interaction endpoint with mean NRMSE>1, not an absolute-skill headline. Commit: `6b4b4ea68fd040d29729d5a8405476e14e15fd69`. Status: **canonical**.
