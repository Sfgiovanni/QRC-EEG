# Frozen protocol — effective-kernel theory versus simulation

Status: **FROZEN BEFORE Stage 1 simulation** (2026-07-14, America/Sao_Paulo).

## Fixed system

- Construction: the gate-selected `single_kernel` (`K`, `r`, and past mass read from the frozen
  `results/eeg/hp_selected.json`), compared with the identical `QRC_K0` base reservoir.
- Channel/observables: the existing four-qubit channel and 66 Pauli features, seed 1.
- Expansion point: constant standardized input `u0=0`, after convergence to its fixed state.
- Perturbation amplitude: `epsilon=1e-4` standardized input units.
- Response length: 256 samples. Initial state and every history-buffer entry equal the fixed state.
- Signals: unit impulse at sample 0 and unit step from sample 0, each multiplied by epsilon in the
  nonlinear simulation and divided by epsilon after baseline subtraction.

No Bonn segment is read by this test. It probes the already-implemented reservoir mechanism.

## Models tested

1. **Tangent recurrence (implementation-faithful):**
   `delta x[t+1] = A sum_(tau=0)^K w_tau delta x[t-tau] + B delta u[t]`, with `A` and `B`
   evaluated numerically at the fixed state and feature map `C` unchanged.
2. **Separable product proposed in the prompt:** `H_sep(z)=W(z)R(z)`, implemented by convolving
   the measured/tangent K=0 impulse response `R` with `[w0,w1,...,wK]`.

The second expression is not assumed exact. Algebra must decide whether it represents the actual
state-history recurrence. It is tested separately so a good tangent linearization cannot conceal a
bad factorization.

## Frozen metrics and tolerances

All norms combine time and all 66 features. Denominators are protected only by machine epsilon.

- impulse relative RMSE: `||g_pred-g_meas||_F / ||g_meas||_F <= 0.01`;
- step relative RMSE: `||s_pred-s_meas||_F / ||s_meas||_F <= 0.01`;
- frequency-response relative error: Frobenius error of the 256-point complex FFT of the impulse,
  divided by measured FFT norm, `<= 0.01`;
- linear-memory-function L1 error: impulse-energy distributions
  `m(t)=||g(t)||_2^2/sum_t ||g(t)||_2^2`, `sum_t |m_pred(t)-m_meas(t)| <= 0.02`.

## Frozen Gate 1 rule

- `TANGENT_PASS` requires all four tangent-recurrence metrics to meet tolerance for the
  state-memory kernel.
- `SEPARABLE_PASS` requires all four `W(z)R(z)` metrics to meet the same tolerances.
- Overall **PASS** requires both. If the tangent recurrence passes but the separable product fails,
  the result is **FAIL_SEPARABLE_FACTORIZATION**: local linearization is adequate, but the requested
  product formula is not the transfer function of the implemented recurrence. If the tangent model
  fails, the result is **FAIL_LINEARIZATION**.

No tolerance, seed, amplitude, response length, norm, or verdict rule may be changed after the
simulation output is observed. The stage stops at Gate 1 regardless of verdict.
