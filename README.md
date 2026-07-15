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

## Scientific question and evidence

The primary question is whether distributed state history changes how forecasting error degrades
with horizon, and whether that effect is consistent with a local recurrent-feedback mechanism.
This is a mechanistic study of a density-matrix QRC simulation, not a demonstration of quantum
advantage or immediate hardware implementation.

## Main results

On held-out Bonn EEG segments (sets Z/F/S, 8 forecast horizons, 10 seeds),
the primary analysis is the preregistered model-by-horizon interaction. It
tests whether the state-memory kernel degrades more slowly than QRC K=0 and
blocked-validation persistence/AR/NVAR/tapped-delay controls. The pipeline
writes the mechanical PASS/FAIL result; this README does not anticipate it.
Exact regenerated counts, effect sizes, confidence intervals and Holm values
are written to `RESULTS.md` and the result CSVs by the pipeline; this README
does not duplicate numeric claims that can become stale.

The intended framing is mechanistic, not a claim of quantum advantage:
classical models lead at short horizons, all models lack absolute skill at
h=64, S is null in the causal K=0 contrast, and the distributed kernel shapes
are effectively tied. Rota A is staged with mandatory human-review stops in
[`docs/rotaA_plan.md`](docs/rotaA_plan.md).

Rota A Stage 1, repeated with the committed frozen configuration K=15, r=0.7
and past mass 0.3, falsified the separable transfer proposal `H(z)=W(z)R(z)`
for the implemented state-history recurrence. The implementation-faithful tangent
resolvent reproduced impulse, step, frequency and memory responses, but the
separable product failed every frozen tolerance. The earlier r=0.9 run is
preserved as `INVALID_CONFIG`. The derivation and stop
decision are in [`docs/effective_kernel_theory.md`](docs/effective_kernel_theory.md).

Rota A Stage 2 tests frozen `H_actual` predictions on synthetic AR(1), AR(2), colored-noise and
phase-surrogate processes. Its mechanical verdict and the scenario-level agreements and failures
are reported in [`results/synth/gate2_report.md`](results/synth/gate2_report.md); this does not
constitute a quantum-advantage claim.

Rota A Stage 3 audits the simulator/resource cost and finite-shot Pauli readout sensitivity. Its
technical and scientific classifications, including the visible S-null and hardware limitations,
are reported in [`results/eeg/gate3_report.md`](results/eeg/gate3_report.md). It does not model a
complete hardware execution or establish quantum advantage.

Honest positive and negative claims are frozen in the [claims registry](docs/claims_registry.md).
Required negatives include the failed external factorization, moderate within-scenario synthetic
ordering, the S-null, tied distributed kernel shapes, lack of absolute skill at h=64 and mixed—not
global—shot sensitivity. Canonical numbers are generated in
[`results/final/key_results.csv`](results/final/key_results.csv), with limitations collected in
[`docs/limitations.md`](docs/limitations.md).

Full results, caveats, and honest null/negative findings are in
[`RESULTS.md`](RESULTS.md). Methodology and every modeling choice are
justified in [`docs/eeg_protocol.md`](docs/eeg_protocol.md); the frozen
hypothesis and any amendments made along the way are in
[`docs/eeg_preregistration.md`](docs/eeg_preregistration.md).
The frozen causal-memory gate is in
[`docs/eeg_gate_preregistration.md`](docs/eeg_gate_preregistration.md).

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
bash scripts/run_rotaA_stage0.sh  # symmetric useful-horizon refresh; stops at Gate 0
bash scripts/run_rotaA_stage1.sh  # theory-vs-simulation falsification; stops at Gate 1
bash scripts/run_rotaA_stage2.sh  # frozen synthetic prediction validation; stops at Gate 2
bash scripts/run_rotaA_stage3.sh  # resources + finite-shot characterization; stops at Gate 3
bash scripts/verify_repository_quick.sh  # fast check; no data download or simulation
bash scripts/run_repository_release.sh  # rebuild derived CSVs/figures + final verifier
```

`bash scripts/run_eeg.sh` is the canonical single command. It performs
fetch+SHA256 verification, segment-grouped sanity/leak tests, HP searches,
eight-horizon held-out runs (including QRC K=0 and ESN-66), persistence,
AR/NVAR/tapped-delay controls, capacity, the 15-contrast `eeg_gate` family,
tables, figures, regenerated narrative, the full pytest suite, provenance
checksums and `verify_gate.py`. Ridge `alpha`, AR order and model HP selection use
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
python scripts/run_gate_baselines.py    # persistence/AR/NVAR/tapped controls
python scripts/run_quadratic_capacity.py
python scripts/run_statistics.py        # Holm-corrected paired contrasts
python scripts/run_esn66_contrasts.py
python scripts/run_tables_figures.py    # REVTeX tables + APS figures
python scripts/make_gate_report.py       # frozen interaction + PASS/FAIL
python scripts/update_results_gate.py    # CSV-derived RESULTS.md
python scripts/verify_gate.py            # final gate + checksums
```

The gate commands can be computationally expensive. Repository-only regeneration uses existing
frozen artifacts and is documented in [reproducibility.md](docs/reproducibility.md).

## Scientific repository navigation

- [Audit summaries for Gates 0–3](docs/gates/README.md)
- [Artifact index with status and SHA256](results/ARTIFACT_INDEX.csv)
- [Final figure inventory](docs/figure_inventory.md)
- [Final CSV-table inventory](docs/table_inventory.md)
- [Evidence map](docs/evidence_map.md) and [reviewer-risk register](docs/reviewer_risk_register.md)
- [Future paper blueprint](docs/paper_blueprint.md) — organization only, not a manuscript

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

## Limitations

Inference is restricted to held-out segments from one Bonn EEG database; segments cannot be mapped
to subjects, so there is no patient-level or clinical generalization. The theory is local, the
implementation mixes density matrices in a classical simulator, and finite-shot analysis is a
readout-sampling model rather than complete hardware noise. See [limitations.md](docs/limitations.md).

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

## Citation

No DOI or immutable software release exists yet. Author, affiliation, ORCID and archival metadata
require human confirmation before creating `CITATION.cff` or `.zenodo.json`; do not cite an
invented identifier. Release preparation is tracked in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
