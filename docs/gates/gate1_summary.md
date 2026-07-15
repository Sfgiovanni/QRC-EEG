# Gate 1 — effective-kernel mechanism

## Scientific question

Does the implementation-faithful tangent recurrence reproduce the local nonlinear response, and is the external factorization `H_sep=W_K R` valid?

## Configuration and inputs

Committed `single_kernel`: K=15, r=0.7, past mass=0.3, seed=1, epsilon=0.0001. Inputs: `results/eeg/theory_vs_sim_check.csv` (SHA256 `9ffc76d4ed8c911224696314ea2aaa4235e9ba65ae73da38d20d36dc5b8413d0`) and `results/eeg/theory_vs_sim_responses.npz` (SHA256 `074eecd3d7006c80e4537c502bd7e0c3ec0c297fb4c2f6a0103da8e6b2f20e55`).

## Scripts, metrics and artifacts

`scripts/run_effective_kernel_check.py`; impulse, step and FFT relative Frobenius errors plus memory-energy L1. Tangent: 4/6 metrics pass. Separable ansatz: 0/6 pass. Companion spectral radius: 0.958724 (locally stable).

Technical verdict: **PASS**. Scientific verdict: **FAIL_SEPARABLE_FACTORIZATION**. The correct local transfer is `H_actual(z)=C[zI-AW_K(z)]^-1B`.

## Limitations and status

Local small-signal theory only; no universal T_eff law or complete EEG explanation. The r=0.9 snapshot is `INVALID_CONFIG`, exploratory only. Commit: `6b4b4ea68fd040d29729d5a8405476e14e15fd69`. Status: **canonical**.
