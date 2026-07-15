# Effective state-history kernel: linear response and falsification result

## Scope and conclusion first

This derivation concerns the simulated/hybrid state-history reservoir implemented in this
repository. It does not establish quantum advantage. The implementation-faithful tangent model
reproduces the nonlinear simulator locally, but the requested separable formula
`H(z)=W(z)R(z)` does **not** represent this recurrence and fails the frozen simulation check.
Accordingly, Stage 1 ends with **FAIL_SEPARABLE_FACTORIZATION** and no Stage 2 prediction is
authorized without human revision of the theoretical premise.

## Fixed-input CPTP map and linearization

For a fixed scalar input `u`, the implemented channel is

\[
\Phi_u(\rho)=U\left[\rho_{\mathrm{in}}(u)\otimes
\operatorname{Tr}_0(\rho)\right]U^\dagger,
\]

where `rho_in(u)` encodes the sigmoid-transformed input in qubit 0 and `U` is the frozen seeded
unitary. For fixed `u`, this map is linear, completely positive and trace preserving. Dependence
on `u` is smooth but nonlinear. Let `rho_*` be the fixed state at constant `u0=0`, and vectorize
trace-zero Hermitian perturbations as `x_t=vec(rho_t-rho_*)`. With

\[
A=D_\rho\Phi_{u_0}|_{\rho_*},\qquad
B=D_u\Phi_u(\rho_*)|_{u_0},\qquad y_t=Cx_t,
\]

the first-order recurrence actually implemented is

\[
x_{t+1}=A\left(\sum_{\tau=0}^{K}w_\tau x_{t-\tau}\right)+B\,\delta u_t,
\qquad \sum_{\tau=0}^{K}w_\tau=1.
\]

The assumptions are: perturbations remain small; the constant-input fixed state is locally
attractive; higher derivatives of the input encoding are negligible at amplitude `1e-4`; and the
same 66 linear Pauli expectations form `C`. The finite-difference derivative is central and the
state derivative is exact on the trace-zero tangent space because `Phi_u0` is linear in `rho`.

## Transfer function: exact recurrence versus separable ansatz

Define

\[
W_K(z)=\sum_{\tau=0}^{K}w_\tau z^{-\tau}.
\]

Up to a convention-dependent monomial delay, the z-transform of the recurrence gives

\[
\boxed{H_{\mathrm{actual}}(z)=C\,[zI-AW_K(z)]^{-1}B.}
\]

For K=0, `W_0=1` and the base response is

\[
R(z)=C(zI-A)^{-1}B.
\]

Multiplying the base response externally by the history polynomial would instead give

\[
H_{\mathrm{sep}}(z)=W_K(z)R(z).
\]

These expressions are not generally equal: `W_K` occurs inside the closed-loop resolvent in the
implemented reservoir, not as an output convolution. For one scalar eigenmode `a` of `A`,

\[
H_{\mathrm{actual},a}(z)\propto [z-aW_K(z)]^{-1},\qquad
H_{\mathrm{sep},a}(z)\propto \frac{W_K(z)}{z-a}.
\]

SymPy verified symbolically that their generic difference is nonzero. Equality requires
degenerate/special cases (for example no delayed mass), not the selected gate model.

## Geometric weights, effective time and poles

For delayed mass `m`, `w0=1-m`, and finite weights proportional to `r^tau`,

\[
S_K(r)=\sum_{\tau=1}^{K}r^\tau=\frac{r(1-r^K)}{1-r},
\]

\[
W_K(z)=(1-m)+\frac{m}{S_K(r)}
\frac{rz^{-1}[1-(rz^{-1})^K]}{1-rz^{-1}}.
\]

Although written rationally, this finite-K expression is an FIR polynomial: the apparent pole at
`z=r` cancels. Only the infinite-K idealization has an uncancelled geometric pole associated with
`r`. Thus it would be incorrect to claim a literal extra pole at `r` for the implemented K=15
kernel. Instead, the closed-loop poles for scalar mode `a` are the roots of

\[
z^{K+1}-a\sum_{\tau=0}^{K}w_\tau z^{K-\tau}=0,
\]

and the full system uses the eigenvalues of the corresponding block companion operator. The exact
stability condition is spectral radius below one for that companion operator. `|r|<1` guarantees
summability of the infinite geometric weights; it is not by itself sufficient for closed-loop
reservoir stability.

The normalized delayed-state mean lag is

\[
T_{\mathrm{eff}}=
\frac{\sum_{\tau=1}^{K}\tau r^\tau}{\sum_{\tau=1}^{K}r^\tau}
=\frac{1-(K+1)r^K+Kr^{K+1}}{(1-r)(1-r^K)}.
\]

For the selected `K=15`, `r=0.9`, delayed mass 0.3 model, SymPy and direct evaluation give
`T_eff=6.11090228779` samples. Delayed mass controls strength, while this normalized `T_eff`
describes location of the delayed weights.

## PSD and forecast-degradation hypothesis

For a stationary small-signal input with spectral density `S_u(omega)`, the linearized readout has

\[
S_y(\omega)=H(e^{i\omega})S_u(\omega)H(e^{i\omega})^*.
\]

The state-history polynomial therefore changes modal poles and the frequency weighting of the
observed features. A broader impulse-energy distribution can retain predictive covariance across
more horizons when its time scales overlap those of the input PSD. K=0 lacks the delayed-state
terms. A single AB delay introduces one isolated lag, whereas distributed weights introduce a
range of lags. This motivates, but does not prove, slower error degradation for distributed
memory. NRMSE slope also depends on the input PSD, nonlinear terms, observability and the fitted
readout; `T_eff` alone cannot determine an ordering.

That qualification is required by the empirical results: S is null in the K=0 causal interaction,
and exponential, dual, triangular and uniform kernels have essentially tied useful horizons. The
data support neither a universal ordering nor specificity of the exponential shape.

## Frozen theory-versus-simulation check

The protocol and tolerances were frozen in `docs/effective_kernel_check_protocol.md` before the
simulator ran. The fixed state converged in 296 iterations with final Frobenius change
`9.4544e-14`. Results are:

| Theory | Check | Error | Tolerance | Pass |
|---|---|---:|---:|---|
| Tangent recurrence | Impulse relative RMSE | 0.00002789 | 0.01 | yes |
| Tangent recurrence | Step relative RMSE | 0.00003945 | 0.01 | yes |
| Tangent recurrence | Frequency relative error | 0.00002789 | 0.01 | yes |
| Tangent recurrence | Memory-function L1 | 0.00000790 | 0.02 | yes |
| `W(z)R(z)` | Impulse relative RMSE | 0.482491 | 0.01 | **no** |
| `W(z)R(z)` | Step relative RMSE | 0.114885 | 0.01 | **no** |
| `W(z)R(z)` | Frequency relative error | 0.482491 | 0.01 | **no** |
| `W(z)R(z)` | Memory-function L1 | 0.735277 | 0.02 | **no** |

The implementation-faithful tangent recurrence is an excellent local description. The separable
filter fails every frozen tolerance by large margins. Therefore the issue is not failure of local
linearization; it is the placement of the history kernel inside the recurrence. Treating it as an
external convolution changes the transfer function.

## Gate 1 verdict and stop

**FAIL_SEPARABLE_FACTORIZATION.** The proposed `H(z)=W(z)R(z)` mechanism is falsified for the
implemented reservoir. The corrected resolvent expression is supported locally, but it does not
yet supply the simple `T_eff -> NRMSE-slope` prediction requested for Stage 2. Per protocol, work
stops here for human review; no synthetic battery, physical-resource analysis or manuscript work
is started.
