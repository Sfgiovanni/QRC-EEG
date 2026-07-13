# Bonn EEG Dataset (Z, F, S sets)

No raw EEG data is committed to this repository. Fetch and verify with:

```bash
python scripts/fetch_eeg.py
```

## Source

Original release: Andrzejak, R. G., Lehnertz, K., Mormann, F., Rieke, C.,
David, P., & Elger, C. E. (2001). *Indications of nonlinear deterministic and
finite-dimensional structures in time series of brain electrical activity:
Dependence on recording region and brain state*, Physical Review E, 64,
061907.

The canonical host (`epileptologie-bonn.de` / `meb.uni-bonn.de`) has been
retired (redirects to a generic departmental homepage); the UPF NTSA mirror
and `archive.ics.uci.edu` both return HTTP 403 to automated fetches. This
project fetches the identical, unmodified original files (100 `.txt` files of
4097 samples per set) from a GitHub mirror instead, pinned to a specific
commit for reproducibility:

- Mirror: `RYH2077/EEG-Epilepsy-Datasets`, commit
  `185859ab48bc701c9a10f6bb2b5f76d8e28e4003`
- Archive SHA256: `f4c2dc52fd5320d4404fcbc6ecb9db69a4a7e408df4e3d5456530343dbcb75ad`
- See `docs/eeg_preregistration.md` for the amendment note explaining this
  substitution (made before any model touched real data).

## Sets used

- **Z** (`A_Z/`): healthy subjects, eyes open (surface EEG)
- **F** (`D_F/`): interictal, epileptogenic zone (intracranial EEG)
- **S** (`E_S/`): ictal / seizure activity (intracranial EEG)

100 segments per set, 4097 samples each, 173.61 Hz sampling rate, 12-bit
resolution. `O` and `N` sets are present in the archive but not used.

## License / usage

Non-commercial research and educational use, per the original database's
terms. Not redistributed here in raw form; `scripts/fetch_eeg.py` downloads
and verifies it on demand.

## Verification

`scripts/fetch_eeg.py` computes the SHA256 of the downloaded archive and
aborts if it does not match the frozen value in `CHECKSUMS.txt` (written on
first successful fetch). It also asserts each set contains exactly 100 files
of exactly 4097 samples before considering the fetch successful.
