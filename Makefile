# cs-intel — developer convenience targets.
# On Windows use Git Bash / WSL, or run the underlying commands directly.

PY ?= ./.venv/Scripts/python.exe   # use ./.venv/bin/python on macOS/Linux

.PHONY: help setup data intents churn causal pipeline llmops api ui test lint typecheck docker up down clean

help:
	@echo "setup      create venv + install dev deps"
	@echo "pipeline   run the full offline pipeline (datagen->intents->churn->causal->llmops)"
	@echo "api        run FastAPI on :8000"
	@echo "ui         run the Vite dev server on :5173"
	@echo "test       run pytest"
	@echo "lint       ruff check"
	@echo "typecheck  mypy"
	@echo "up         docker compose up --build   (dashboard at http://localhost:5173)"

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) -m datagen

intents:
	$(PY) -m intents

churn:
	$(PY) -m churn.train

causal:
	$(PY) -m causal

llmops:
	$(PY) -m common.llmops

pipeline:
	$(PY) scripts/build_pipeline.py --force

api:
	$(PY) -m uvicorn api.main:app --reload --port 8000

ui:
	cd ui && npm install && npm run dev

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

typecheck:
	$(PY) -m mypy common datagen churn intents causal agent api

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -f data/*.parquet data/*.json artifacts/*.png artifacts/*.json artifacts/*.joblib artifacts/*.jsonl
	rm -f agent/memory_store/*.jsonl
