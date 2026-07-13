# EEG Protocol: Methods and Modeling-Choice Justification

## Provenance of reused vs. built code

| Component | Origin |
|---|---|
| `state_kernels.py` (kernel weight construction) | Vendored from `QRC-Glicose` `src/qrc_memory/state_kernels.py`, unchanged except attribution header |
| `models.py` (mix-then-channel reservoir, AB-noaux/kernel-shape classes) | Vendored from `QRC-Glicose` `src/qrc_memory/models.py`, unchanged |
| `metrics.py`, `statistics.py`, `readout.py`, `memory_capacity.py`, `provenance.py`, `stability.py`, `observables.py` | Vendored from `QRC-Glicose`, unchanged |
| `channels.py` (input-encoding channel) | **New.** Neither repository ships an input-*dependent* channel usable on real signals; `QRC-Glicose`'s `identity_channel`/`depolarizing_channel` ignore the scalar input and exist only to exercise `StateMemoryReservoir` in smoke tests. |
| `esn.py` (classical ESN) | **New.** No ESN implementation exists in either repository (`QRC-Kernel`'s ESN numbers are frozen result CSVs with no source; `baselines.py` in `QRC-Glicose` provides only EMA/lagged-window feature maps, not a recurrent reservoir). |
| `batched.py` (vectorized evolution) | **New.** Neither repository batches reservoir evolution; both drive one segment at a time. Built to make the full nested-CV protocol computationally tractable; verified numerically identical to the sequential reference (`tests/test_batched_matches_reference.py`, max abs error < 1e-8). |
| `eeg_data.py`, `tasks.py` (fetch, forecasting targets, demand score, seizure task) | **New.** EEG-specific; no analogue in either repository (which target glucose forecasting). |

## Why AB is "noaux" and ABC is dropped

Both repositories were searched, including full git history (not just HEAD;
`QRC-Glicose` is a single squashed commit, `QRC-Kernel` has four commits, none
of which touch a genuine auxiliary-qubit implementation). Every "AB"/"ABC"
variant that exists (`AB-noaux-residual`, `AB-optimized`, `ABC-noaux-tied`,
`legacy-AB`) is a discrete-delay or hierarchical-delay approximation that
mixes past states of the *same* register -- there is no coupling Hamiltonian,
no `J_AB`/`J_BC`, no `gamma_B`/`gamma_C`, and no second/third qubit register
anywhere. Building a genuine multi-qubit backflow construction from scratch
would mean inventing the physical model (Hamiltonian, coupling topology,
dissipation schedule) that the study is supposed to be testing -- exactly the
kind of unreviewed strawman the anti-inflation guardrail (spec section 3)
exists to prevent. Per the maintainer (2026-07-12): use the noaux AB,
labeled honestly; drop ABC entirely.

One consequence: the qubit-inflation concern from the original spec section 3
("if the kernel beats AB/ABC with 5-6 qubits...") does not arise, because
every quantum arm in this study -- kernel, AB-noaux, kernel-shape ablations --
is a 4-qubit / 16x16 construction. The qubit-count and feature-count table is
still reported in `tab_quadratic_capacity` for transparency, but there is no
dimension asymmetry to defend among the quantum arms (the classical ESN
control is a separate, explicitly larger register; see the feature-dimension
caveat in `RESULTS.md`).

## Input-encoding channel: construction and why it doesn't confound the comparison

Standard Fujii-Nakajima-style QRC channel (`src/qrc_eeg/channels.py`):

1. Qubit 0 of the 4-qubit register is the *input qubit*. Its reduced state is
   discarded (partial trace) and replaced with an amplitude encoding of the
   scalar input `u in (0,1)` (obtained by squashing the z-scored EEG sample
   through a logistic function): `|psi(u)> = sqrt(1-u)|0> + sqrt(u)|1>`.
2. The resulting product state is evolved by a **fixed, seeded** entangling
   unitary `U = expm(-i H t)`, `H` a random transverse-field Ising
   Hamiltonian (`H = sum_i h_i Z_i + sum_{i<j} J_ij X_i X_j`, couplings drawn
   once from `Uniform(0.5, 1.5)` with a frozen seed, `t = 1.0`).

This channel is applied identically, with the identical frozen `U`, in every
reservoir arm (AB-noaux, single/dual kernel, triangular, uniform). Only
the upstream history-mixing step (`KernelWeights`, i.e. how much and which
past density matrices get blended into the state *before* this channel is
applied) differs between arms. Because the channel is held constant, any
difference in downstream forecasting performance across arms is attributable
to the memory mechanism, not to differences in how the signal is encoded or
how strongly the register is scrambled -- which is the controlled comparison
the study is designed around.

## Classical ESN control

Standard leaky-integrator ESN (`src/qrc_eeg/esn.py`):
`x_t = (1-a) x_{t-1} + a * tanh(W_res x_{t-1} + W_in u_t)`, `W_res` scaled to
a target spectral radius, ridge readout on `x_t`. This is a well-precedented
classical-substrate control, not the model the kernel construction is trying
to beat (per the pre-registered hypothesis).

## Feature extraction

All quantum arms use the same feature map: real expectation values of every
weight-<=2 Pauli string on the 4-qubit register (`observables.py`,
vendored), 66 features total, fed to the same ridge readout
(`readout.py`, vendored, `alpha` selected by nested CV).

## What "seed" means for the quantum arms

The reservoir evolution itself is otherwise deterministic (fixed channel,
fixed kernel weights, linear ridge readout) -- there is no intrinsic
randomness to vary across "10 common seeds" unless the channel itself is
seeded. Resolution: `channels.build_input_channel(seed=k)` draws a distinct
random transverse-field Ising Hamiltonian (hence a distinct frozen unitary
`U_k`) per seed `k`. For a given trial `k`, `U_k` is held identical **across
every construction** (AB-noaux, kernels, and the analogous ESN weight
draw for that same `k`) -- so the arm-vs-arm contrast within a seed is still
uncontaminated -- while the 10 trials give genuine seed-to-seed variability
for the paired bootstrap/Wilcoxon statistics.

## HP-selection protocol (mirrors QRC-Kernel's own documented design)

Hyperparameter and ridge-alpha selection: train -> validation only. Final
readout: refit on train+validation combined. Final evaluation: held-out test
segments only, 10 common seeds. This is the same split discipline
`QRC-Kernel`'s README documents for its T2DM benchmark, reused here for EEG.
One simplification made for compute-budget reasons: HP/alpha selection is
done on segments and horizons **pooled** across all three sets (Z, F, S) and
all four horizons, yielding a single frozen HP configuration per
construction rather than per-set/per-horizon tuning. This is logged here
rather than silently applied: it makes the comparison *more* conservative
(nobody gets set-specific tuning), not less, since every construction is
still tuned as generously as every other.

## Compute-budget notes

Sequential per-segment evolution costs ~0.28 ms/step (measured); a full
protocol (hundreds of segments x seeds x HP grid) at that rate would take on
the order of 10 hours. `batched.py` vectorizes evolution across a batch axis
(segments and/or seeds), giving a measured throughput improvement (see
`results/eeg/PROVENANCE.md` for the actual measured numbers used in this run)
without changing any equation -- verified bit-for-bit equivalent (up to
floating-point associativity, <1e-8 max error) against the sequential
reference. Where the HP-search stage still needed a reduced grid or seed
count to fit the compute budget, the reduction applies **only** to HP search,
never to the confirmatory held-out evaluation; exact counts are logged in
`results/eeg/PROVENANCE.md`.
