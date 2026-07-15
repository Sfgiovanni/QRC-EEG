# Reproducibility commands

## Fast, no data download or simulation

```bash
bash scripts/verify_repository_quick.sh
```

This checks frozen hashes, final CSV consistency, figures, documentation guards and `.tex`
integrity. It does not regenerate EEG or synthetic reservoirs.

## Tests and derived release artifacts

```bash
.venv/bin/python -m pytest -q
bash scripts/run_repository_release.sh
```

The release wrapper rebuilds only tables/figures/indexes from existing frozen artifacts and runs
the final verifier.

## Reproduction by gate

```bash
bash scripts/run_eeg.sh             # corrected EEG gate; expensive
bash scripts/run_rotaA_stage1.sh    # local theory check
bash scripts/run_rotaA_stage2.sh    # synthetic reservoirs; expensive
bash scripts/run_rotaA_stage3.sh    # finite shots; very expensive
```

Gate wrappers stop at their own gate. Full recomputation requires the Bonn data, significant CPU
time and disk space. Derived repository preparation never substitutes r=0.9 for the committed
r=0.7 confirmatory configuration.
