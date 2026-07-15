# Gate 3 — resources and finite-shot sensitivity

## Scientific question

What are the simulator/measurement resources and how heterogeneous is finite-shot readout sensitivity?

## Configuration, inputs and scripts

QRC K0, AB-noaux and single-kernel; exact plus N={100,300,1000,3000,10000}, 10 noise replicates, seeds 1–3, Z/F/S, eight horizons. `scripts/run_shot_sensitivity.py`, `scripts/make_physical_resource_table.py`. Sources: `results/eeg/shot_sensitivity_raw.csv` (SHA256 `211ced0d4809347b4cebd6f663569d78600956a5fcd5b81e08a7d4c9de11ca28`) and `results/resources/qrc_resource_table.csv` (SHA256 `5421daa1897482a4cfd4c60b67c87af7b13abc8ed7688e1f9f457f40e4d14ea4`).

## Metrics and resources

Median/P90 relative NRMSE inflation, absolute inflation, set×horizon strata, interaction sign and magnitude. No shot level passes globally; 66/120 strata pass. K0/AB/K15 buffers use 4096/24576/65536 bytes. There are 66 conservative measurement groups.

Technical verdict: **COMPLETE**. Scientific verdict: **MIXED_SHOT_SENSITIVITY**.

## Limitations

Principal sign fraction covers only preregistered F/Z contrasts; S is reported as null. Sign preservation is not magnitude preservation. h=64 is a sensitivity endpoint, not a skill headline. Shot noise omits decoherence, preparation error, drift, full backaction and physical history storage. Commit: `6b4b4ea68fd040d29729d5a8405476e14e15fd69`. Status: **canonical**.
