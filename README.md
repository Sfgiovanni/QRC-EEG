# Exponential State-Memory Kernel QRC on EEG: A Case Study

Quantum reservoir computing (QRC) forecasting of real EEG signals, comparing
an exponentially weighted state-memory kernel construction against a
discrete-delay QRC alternative (AB-noaux) and a classical echo state network
(ESN) control, on the Bonn University EEG database.

The reservoir mixes the current density matrix with an exponentially
weighted history of past density matrices before applying an input-dependent
quantum channel:

```text
rho_tilde(t) = w0 rho(t) + sum_{tau=1}^K w_tau rho(t - tau)
rho(t+1)    = E_{u_t}(rho_tilde(t))
```

`w_tau` is proportional to `r^tau` for the primary single-exponential kernel;
a dual-exponential (two-timescale) variant and triangular/uniform
kernel-shape ablations are also evaluated. The channel `E_u` is a standard
Fujii-Nakajima-style input-encoding channel (fixed, seeded entangling
unitary), held identical across every construction so that only the
history-mixing mechanism differs between arms.

## Main results

On held-out Bonn EEG segments (sets Z/F/S, 4 forecast horizons, 10 seeds),
the corrected analysis is reported as a horizon-dependent interaction:
ESN-66 tends to lead at short horizons and the kernel at longer horizons.
Exact regenerated counts, effect sizes, confidence intervals and Holm values
are written to `RESULTS.md` and the result CSVs by the pipeline; this README
does not duplicate numeric claims that can become stale.

Full results, caveats, and honest null/negative findings are in
[`RESULTS.md`](RESULTS.md). Methodology and every modeling choice are
justified in [`docs/eeg_protocol.md`](docs/eeg_protocol.md); the frozen
hypothesis and any amendments made along the way are in
[`docs/eeg_preregistration.md`](docs/eeg_preregistration.md).

## Repository structure

```text
src/qrc_eeg/        Installable Python package (reservoir models, channel,
                     ESN, statistics, pipeline)
scripts/             Data acquisition, HP search, evaluation, figures
config/              Frozen hyperparameter grids and study configuration
tests/               Mechanism sanity checks and guardrail tests
data/eeg/            Dataset instructions and checksums (no raw data)
results/eeg/         Result tables and provenance
figures/eeg/         Main figures (PDF + PNG, APS style)
paper/               REVTeX table sources
docs/                Protocol, pre-registration, provenance justification
provenance/          Checksums for every generated artifact
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick start

```bash
pytest                        # mechanism sanity checks + guardrails
bash scripts/run_eeg.sh       # complete deterministic pipeline and final fail-high gate
```

`bash scripts/run_eeg.sh` is the canonical single command. It performs
fetch+SHA256 verification, segment-grouped sanity/leak tests, HP searches,
held-out runs (including ESN-66), capacity, paired statistics, tables,
figures, regenerated narrative/diff, the full pytest suite, provenance
checksums and `verify_fase1.py`. Ridge `alpha` and model HP selection use
whole disjoint train/validation segments; temporal rows are never randomly
mixed across those partitions.

Or step by step:

```bash
python scripts/fetch_eeg.py             # download + verify Bonn EEG data
python scripts/run_sanity_checks.py     # mechanism checks (must pass)
python scripts/run_hp_search.py         # train -> validation HP selection
python scripts/run_esn66_hp_search.py   # matched ESN HP selection
python scripts/run_holdout_eval.py      # held-out evaluation, 10 seeds
python scripts/run_esn66_holdout.py     # matched ESN held-out evaluation
python scripts/run_quadratic_capacity.py
python scripts/run_statistics.py        # Holm-corrected paired contrasts
python scripts/run_esn66_contrasts.py
python scripts/run_tables_figures.py    # REVTeX tables + APS figures
python scripts/verify_fase1.py           # final gate + checksums
```

## Dataset

Bonn University EEG database (Andrzejak et al., 2001): sets **Z** (healthy,
eyes open), **F** (interictal, epileptogenic zone), **S** (ictal). 100
segments per set, 4097 samples, 173.61 Hz. Not redistributed in this
repository; `scripts/fetch_eeg.py` downloads and SHA256-verifies it. See
[`data/eeg/README.md`](data/eeg/README.md) for the exact source and license.

## Reproducibility

Every number in the tables and figures traces to a script and source file
in [`results/eeg/PROVENANCE.md`](results/eeg/PROVENANCE.md), with SHA256
checksums for every generated file in
[`provenance/eeg_checksums.txt`](provenance/eeg_checksums.txt). Splits,
seeds, and hyperparameter grids are frozen in
[`config/eeg_frozen.yaml`](config/eeg_frozen.yaml).

## Provenance of the code

The kernel-weight construction, mixing reservoir, metrics, statistics,
ridge readout, and memory-capacity protocol are vendored from the author's
own `QRC-Glicose` repository (MIT license). The input-encoding channel,
classical ESN, batched/vectorized reservoir evolution, and all EEG-specific
code (data acquisition, forecasting tasks, nested HP search) were built new
for this study. See `docs/eeg_protocol.md` for the full breakdown.

## License

Code: MIT (see `LICENSE`). Results and documentation: see individual file
headers. The EEG dataset itself is not redistributed here; see
`data/eeg/README.md` for its license terms.
