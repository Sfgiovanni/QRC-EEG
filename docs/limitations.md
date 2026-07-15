# Limitations

- The EEG evidence uses one Bonn benchmark and held-out segments, not identified patients. The
  unavailable/randomized segment-to-subject mapping prevents subject-disjoint inference,
  between-patient generalization and clinical claims.
- EEG time rows are autocorrelated. Causal normalization, whole-segment partitions and blocked HP
  validation prevent identified leakage modes but do not make segments independent patients.
- Classical AR/NVAR/tapped-delay and ESN comparisons are empirical controls, not an exhaustive
  model search. Classical methods lead at short horizons.
- Mean NRMSE exceeds 1 at h=64 for all evaluated models. That horizon is retained as a frozen
  interaction/sensitivity endpoint, not an absolute forecasting-skill headline.
- The exponential, triangular and uniform distributed kernels have overlapping results. No
  universal superiority of the exponential shape is supported.
- The transfer theory is a local small-signal linearization. `T_eff` alone does not determine
  forecasting slopes; PSD, observability, nonlinearity and readout matter.
- Gate 2 is mechanically `SUPPORTED`, but aggregate correlation includes between-process scale and
  within-scenario ordering is moderate with explicit failures.
- The implementation stores and mixes density matrices classically. Unknown states cannot be
  freely copied, and a hardware protocol would require ensembles, reexecution, repreparation,
  memory/ancillas and a gate decomposition that are not supplied here.
- Shot sensitivity adds finite Pauli-estimation noise only. It omits gate decoherence, preparation
  error, drift, full recurrent backaction and physical state-history storage.
- `MIXED_SHOT_SENSITIVITY` is not global robustness and is not evidence of quantum advantage.
