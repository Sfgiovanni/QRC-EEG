# EEG Results

Pre-registration: `docs/eeg_preregistration.md`. Protocol and modeling-choice
justification: `docs/eeg_protocol.md`. Every number below traces to a script
and source file in `results/eeg/PROVENANCE.md`.

## Bottom line (reported as observed, not adjusted to fit the story)

Held-out results (kernel variants, AB-noaux, and ESN; 3 sets x 4 horizons x
10 seeds, `n=20` held-out test segments per set):

The statistical unit is the segment, not the patient. The Bonn release
randomized and withheld the segment-to-subject mapping, so there is no
subject-separated evaluation and no evidence here for generalization between
patients. Z is surface EEG; F and S are intracranial EEG. All inference is
limited to benchmark segments.

**The kernel achieves strong absolute forecasting performance.** At 1-step
lookahead, single-kernel reaches R^2 = 0.92 (Z), 0.97 (F), 0.97 (S); even at
2-step lookahead R^2 stays at 0.74-0.89. This degrades at longer horizons
(4, 8 steps), as expected for EEG forecasting generally.

**The one clean comparative positive: kernel beats AB-noaux.** Significant in
8/12 set x horizon cells, mean win fraction 0.854 favoring the kernel. Within
the QRC family, the exponential-kernel history mixing is clearly better than
concentrating all delayed mass at one discrete lag.

**Dual kernel does not beat single kernel**: significant in 4/12 cells,
mean win fraction 0.312 -- when it differs, dual tends to be *worse*, not
better. The added multiscale complexity does not pay off here.

**Secondary, caveated observation -- classical ESN vs. kernel (dimension-
confounded, do not over-read):** the held-out run showed ESN with the
lowest mean NRMSE in every set (`tab_eeg_endpoints.csv`: ESN
0.482/0.522/0.631 vs. single-kernel 0.484/0.539/0.635 on F/S/Z), significant
in 9/12 cells, mean win fraction 0.279 for the kernel. But the ESN readout
used **200** features against **66** for every quantum arm -- a 3x
expressivity difference, not an equalized budget. A follow-up check with
`n_reservoir=66` (matching the quantum feature count exactly, set Z, seed 1)
gives ESN-66 NRMSE 0.2627 at `h=1` vs. single-kernel 0.2657 at the same
horizon/set -- the gap **collapses to noise level** once the readout
dimension is equalized. The pre-registration always framed ESN as a
substrate control, "not a target to beat"; at matched budget there is no
evidence it beats the kernel either. The original 200-feature ESN numbers
are kept in `tab_eeg_endpoints.csv` for transparency but should not be read
as a substrate-level result.

**Quadratic capacity** (frozen synthetic protocol, independent of EEG):
single-kernel (2.614), triangular (2.607), and uniform (2.582) lead among
quantum arms; AB-noaux is lowest quantum arm (1.914); ESN is far lower
(0.013) than every quantum construction despite its raw NRMSE numbers above
-- another sign the raw ESN comparison was dimension/task-fit driven, not a
genuine memory-capacity advantage. The quadratic iid measure used here showed
no detectable positive association with EEG gains in the evaluated
configurations. The corrected descriptive analysis has only three independent
comparison-level capacity gaps (`n=3`), not 36 independent set x horizon x
comparison rows, and therefore reports no regression confidence interval.
See `results/eeg/capacity_demand_regression_summary.csv`.

**Interpretation, stated plainly**: as a case-study demonstration, the
single-exponential kernel construction achieves strong absolute EEG
forecasting accuracy at short horizons, clearly outperforms the weaker QRC
alternative (AB-noaux), and, at matched readout dimension, is competitive
with a classical ESN control. It has the highest quadratic memory capacity
among the tested QRC constructions on the frozen synthetic protocol. The
demand-gradient framing of the main figure is undermined by the
nonlinear-demand-score finding below (main figure uses a categorical Z/F/S
axis instead). This is reported as the honest result, per the
pre-registration's honesty commitment -- not reframed post hoc.

## Known finding logged before the confirmatory run

The pre-registered nonlinear-demand score `D` (frozen formula, computed on
real Z/F/S segments before any reservoir model touched the data) came out
near-zero for all three sets, and its linear-predictability component is
ordered **opposite** to the hypothesis (S, ictal, is the *most* linearly
predictable set, not the least). This is reported, not adjusted --
see the amendment in `docs/eeg_preregistration.md`. It affects only the
demand-gradient framing of the main figure; the reservoir-construction
contrasts below are independent of it.

## Tables

- `results/eeg/tab_eeg_endpoints.csv` / `paper/tab_eeg_endpoints.tex`
- `results/eeg/tab_eeg_contrasts.csv` / `paper/tab_eeg_contrasts.tex`
- `results/eeg/quadratic_capacity.csv` / `paper/tab_quadratic_capacity.tex`
- `results/eeg/linear_capacity.csv`

## Figures

- `figures/eeg/fig_eeg_demand_gradient.{pdf,png}`
- `figures/eeg/fig_eeg_capacity_vs_gain.{pdf,png}`

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/run_eeg.sh
```
