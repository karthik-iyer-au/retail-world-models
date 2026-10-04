.PHONY: install test smoke run data-register data-verify

install:
	pip install -e ".[dev,trees]"

test:
	pytest -q

# End-to-end check on generated data. No downloads needed.
smoke:
	python -m rwm.experiments.run --config configs/experiments/smoke.yaml

# Reportable run. Refuses to start on uncommitted code or unregistered data.
# Usage: make run CONFIG=configs/experiments/m5_test_seasonal_naive.yaml
run:
	python -m rwm.experiments.run --config $(CONFIG) --strict

# Usage: make data-register DATASET=m5
data-register:
	python -m rwm.data.manifest register $(DATASET)

data-verify:
	python -m rwm.data.manifest verify $(DATASET)
