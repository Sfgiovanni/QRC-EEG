# Physical resources and implementability of the simulated state-history QRC

## What is implemented

The present code is a **hybrid density-matrix simulation of QRC**. At each time step it stores a
buffer of density matrices, forms their weighted convex mixture classically, applies a dense
input-dependent channel, and computes 66 exact Pauli expectations. It is not an immediate hardware
implementation, and exact expectations are not physical single-shot measurements.

For n qubits, d=2^n and maximum state delay K, the buffer contains K+1 normalized Hermitian
matrices. Their independent real dimension is `(K+1)(d^2-1)`; the conservative algorithmic count
is `(K+1)d^2` real scalars. The actual NumPy implementation stores complex128 arrays and therefore
uses `16(K+1)d^2` bytes for an unbatched dense buffer. The CSV computes these quantities from the
committed HPs and verifies dtype rather than embedding the expected numbers. For the current
four-qubit K=15 arms this yields 16 matrices, 4080 independent real parameters, 4096 conservative
real scalars and 65,536 bytes (64 KiB). K=0 uses one matrix and 4096 bytes (4 KiB).

The classical mixing proxy is O((K+1)d^2), a dense two-sided channel application is O(d^3), and
66 trace expectations cost O(66d^2) per step. These are simulator costs, not physical gate counts.
The unitary has not been decomposed into a hardware gate set, so no physical gate count, depth,
connectivity mapping or NISQ realizability is claimed.

## Measurement and trajectory cost

The conservative measurement model assigns one group to each of 66 observables. Qubit-wise
commuting grouping is not implemented. At N shots this means `66N` state preparations/repetitions
per time step and `66NT` over a length-T trajectory, before accounting for reproducing history.
The resource CSV reports values at N=10000 and T=4097, but the expressions apply to every tested
shot level. Parallel ensembles can exchange wall-clock time for hardware count; they do not remove
the preparation cost.

## Physical implementability

Unknown quantum states cannot be copied freely. Measuring 66 generally incompatible observables
destroys or perturbs a system, so expectations at many times and bases require parallel ensembles,
reexecution of the full input history, repreparation, quantum memory/ancillas, or an equivalent
probabilistic protocol. A convex mixture can in principle be realized by classical randomization
between preparations, provided the past states can be reproduced. This still requires classical
records and reexecution, or additional quantum storage; distributed state memory is not free.

Consequently, the current explicit mixture of past density matrices should be described as a
hybrid/simulated state-history mechanism, or a QRC/quantum-inspired simulation where appropriate.
It must not be described as a ready NISQ circuit without an explicit state-reproduction, memory,
measurement and gate-decomposition protocol.

## What the finite-shot experiment does not model

The Gate 3 experiment adds binomial estimation noise to Pauli readout features in training,
validation and test. It does not simulate gate decoherence, state-preparation error, drift,
complete measurement backaction in the recurrence, or solve physical storage of past states.
Independent per-observable ensembles do not reproduce all correlations or backaction of hardware.
Finite-shot robustness, if observed, is only robustness of the fitted readout to sampling noise;
it is neither hardware validation nor evidence of quantum advantage.

## Gate 3 decomposition and interpretation guardrails

Canonical decompositions are `results/eeg/shot_sensitivity_by_stratum.csv` and
`results/eeg/shot_sensitivity_tail_analysis.csv`. They report absolute and relative inflation by
shot level, model, set and horizon, identify upper-tail cells, and retain both sign and magnitude
changes of the frozen interactions.

“Principal sign fraction” uses only the preregistered F/Z kernel-vs-K0 and kernel-vs-AB
interactions. S remains visible but is not counted as a positive effect because its primary causal
contrast was null. Preserving a sign does not imply preserving its magnitude. The h=64 contrast is
retained as a frozen sensitivity test, but cannot lead a forecasting claim because mean NRMSE is
above one there. The scientific classification remains `MIXED_SHOT_SENSITIVITY`.
