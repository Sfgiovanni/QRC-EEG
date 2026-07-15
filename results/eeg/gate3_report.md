# Rota A Gate 3 — resources and finite-shot sensitivity

**Technical verdict: COMPLETE.**
**Scientific classification: MIXED_SHOT_SENSITIVITY.**

The exact official-r=0.7 baseline passed every available frozen-row comparison before shots.
Extended single-kernel r=0.7 horizons had no frozen reference and are not presented as a reproduction.

## Shot-level classification

| Shots | Median relative inflation | P90 | Principal sign fraction | Pass |
|---:|---:|---:|---:|:---:|
| 100 | 4.9469% | 144.1757% | 1.000 | False |
| 300 | 3.8626% | 101.3897% | 1.000 | False |
| 1000 | 2.9070% | 66.4244% | 1.000 | False |
| 3000 | 2.1822% | 46.0080% | 1.000 | False |
| 10000 | 1.5768% | 31.0484% | 1.000 | False |

No finite shot level passes globally, but 66 of 120 set×horizon strata pass the frozen 5%/10% criteria. The classification is therefore MIXED rather than globally robust or uniformly sensitive.

## Contrasts

Kernel-vs-K0 and kernel-vs-AB interactions use h=2 and h=64. S is reported even when null.

| Shots | Set | Comparator | Interaction | Change | Sign preserved |
|---:|---|---|---:|---:|:---:|
| 0 | Z | QRC_K0 | 0.023333 | 0.000000 | True |
| 0 | Z | AB_noaux | -0.057726 | 0.000000 | True |
| 0 | F | QRC_K0 | 0.025441 | 0.000000 | True |
| 0 | F | AB_noaux | -0.029130 | 0.000000 | True |
| 0 | S | QRC_K0 | -0.004187 | 0.000000 | True |
| 0 | S | AB_noaux | -0.096381 | 0.000000 | True |
| 100 | Z | QRC_K0 | 0.014062 | -0.009271 | True |
| 100 | Z | AB_noaux | -0.007215 | 0.050511 | True |
| 100 | F | QRC_K0 | 0.026253 | 0.000812 | True |
| 100 | F | AB_noaux | -0.006851 | 0.022279 | True |
| 100 | S | QRC_K0 | 0.005656 | 0.009843 | False |
| 100 | S | AB_noaux | -0.001669 | 0.094712 | True |
| 300 | Z | QRC_K0 | 0.021218 | -0.002115 | True |
| 300 | Z | AB_noaux | -0.008889 | 0.048837 | True |
| 300 | F | QRC_K0 | 0.025390 | -0.000051 | True |
| 300 | F | AB_noaux | -0.010684 | 0.018446 | True |
| 300 | S | QRC_K0 | 0.006528 | 0.010715 | False |
| 300 | S | AB_noaux | -0.010698 | 0.085683 | True |
| 1000 | Z | QRC_K0 | 0.029634 | 0.006300 | True |
| 1000 | Z | AB_noaux | -0.017233 | 0.040493 | True |
| 1000 | F | QRC_K0 | 0.031457 | 0.006016 | True |
| 1000 | F | AB_noaux | -0.010798 | 0.018332 | True |
| 1000 | S | QRC_K0 | 0.010438 | 0.014626 | False |
| 1000 | S | AB_noaux | -0.017249 | 0.079132 | True |
| 3000 | Z | QRC_K0 | 0.032478 | 0.009145 | True |
| 3000 | Z | AB_noaux | -0.026342 | 0.031384 | True |
| 3000 | F | QRC_K0 | 0.035116 | 0.009675 | True |
| 3000 | F | AB_noaux | -0.012486 | 0.016644 | True |
| 3000 | S | QRC_K0 | 0.018797 | 0.022985 | False |
| 3000 | S | AB_noaux | -0.033351 | 0.063029 | True |
| 10000 | Z | QRC_K0 | 0.032387 | 0.009053 | True |
| 10000 | Z | AB_noaux | -0.036677 | 0.021049 | True |
| 10000 | F | QRC_K0 | 0.037835 | 0.012394 | True |
| 10000 | F | AB_noaux | -0.010637 | 0.018493 | True |
| 10000 | S | QRC_K0 | 0.023099 | 0.027286 | False |
| 10000 | S | AB_noaux | -0.052642 | 0.043739 | True |

## Required limitations

This experiment adds binomial estimation noise to Pauli observables in train, validation and test.
It does not model gate decoherence, state-preparation error, drift, complete measurement backaction,
or physical storage of past states. Independent observable ensembles do not emulate every hardware
correlation. Finite shots are not a real hardware execution, and robustness is not quantum advantage.

Gate 3 stops here; no Stage 4 or manuscript work was performed.
