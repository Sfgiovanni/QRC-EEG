# Repository structure

- `config/`: frozen empirical/theory/synthetic/shot configurations.
- `data/eeg/`: data provenance, fetch verification and frozen segment splits; raw data ignored.
- `src/qrc_eeg/`: reusable simulation, preprocessing, readout, baseline and resource code.
- `scripts/`: gate execution, derived-artifact generation and fail-high verification.
- `tests/`: leakage, mechanism, synthetic, shot and release guardrails.
- `results/eeg/`: canonical EEG/Gate 1/Gate 3 outputs plus labeled historical snapshots.
- `results/synth/`: frozen Gate 2 predictions, measurements and post-gate diagnostics.
- `results/resources/`: physical/simulator resource audit.
- `results/final/`: machine-readable writing inputs generated in Stage 4.
- `figures/final/`: four canonical repository figures in PDF and 600-dpi PNG.
- `docs/gates/`: one audit summary per completed gate.
- `paper/`: pre-existing LaTeX tables; Stage 4 does not read, create or modify `.tex` files.
- `provenance/`: SHA256 inventory.

`results/ARTIFACT_INDEX.csv` classifies canonical, derived, legacy, snapshot and invalid artifacts.
