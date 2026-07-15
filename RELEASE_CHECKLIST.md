# Scientific repository release checklist

## Automated

- [ ] `bash scripts/verify_repository_quick.sh`
- [ ] `.venv/bin/python -m pytest -q`
- [ ] Gate 0–3 verifiers pass or emit only documented cross-stage warnings.
- [ ] `bash scripts/run_repository_release.sh` rebuilds derived CSV/figures deterministically.
- [ ] All final PDFs exist and PNGs report at least 600 dpi.
- [ ] SHA256 inventory is current.
- [ ] Every `.tex` hash matches the Stage 4 preflight.

## Human review required

- [ ] Confirm author names, order, affiliations and ORCIDs.
- [ ] Confirm funding, conflicts and acknowledgments.
- [ ] Verify literature citations and Bonn data terms.
- [ ] Review every permitted/prohibited claim.
- [ ] Decide archival contents and create a DOI only after immutable deposition.
- [ ] Review manuscript separately; no manuscript is created by this stage.
- [ ] Commit/tag/push only after explicit authorization.
