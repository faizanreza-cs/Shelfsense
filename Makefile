PY ?= .venv/bin/python
UV ?= uv

.PHONY: setup build test lint eval eval-llm api charts powerbi monitor docker clean

setup:            ## create venv and install everything
	$(UV) venv --allow-existing .venv
	$(UV) pip install --python $(PY) -e ".[llm,dev]"

build:            ## generate the store, train every model, write data/results.json
	$(PY) -m shelfsense.cli build

test:             ## unit + integration tests (offline, mock LLM)
	$(PY) -m pytest

lint:             ## ruff + format check + mypy
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

eval:             ## agent eval on golden questions with the offline mock provider
	$(PY) -m shelfsense.evals.run_evals --provider mock --min-pass-rate 0.95

eval-llm:         ## agent eval with a real model (set provider credentials and SHELFSENSE_MODEL)
	$(PY) -m shelfsense.evals.run_evals --provider $${SHELFSENSE_PROVIDER:-anthropic}

api:              ## REST API on :8000
	$(PY) -m uvicorn shelfsense.api:app --reload --port 8000

charts:           ## README charts
	$(PY) -m shelfsense.cli charts

powerbi:          ## Power BI star schema CSVs in powerbi/data
	$(PY) -m shelfsense.cli export-powerbi

monitor:          ## drift and alerts for the latest month (exit 1 when something fired)
	$(PY) -m shelfsense.cli monitor

docker:
	docker build -t shelfsense .

clean:
	rm -rf data .pytest_cache .mypy_cache .ruff_cache powerbi/data
