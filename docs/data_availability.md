# Data availability preparation

The Bonn EEG data are not redistributed by this repository. `scripts/fetch_eeg.py` retrieves the
configured source and `data/eeg/CHECKSUMS.txt` verifies the downloaded files. Exact origin and the
4096/4097-sample discrepancy are documented in `data/eeg/README.md`.

Generated result CSVs, frozen configurations, scripts and hashes are local repository artifacts.
No DOI or immutable public archive exists yet. Before publication, the human author must confirm
the upstream data terms, choose the archival payload and create an immutable release. Do not cite
a DOI until one actually exists.
