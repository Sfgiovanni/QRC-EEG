.PHONY: reproduce-eeg test

reproduce-eeg:
	bash scripts/run_eeg.sh

test:
	python -m pytest -q
