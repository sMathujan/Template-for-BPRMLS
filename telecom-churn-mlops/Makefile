# Telecom Customer Churn — production pipeline automation
# bash is required: the template's `source` commands fail under the default /bin/sh
SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV        := .venv
PYTHON      := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
MLFLOW_PORT ?= 5001
API_PORT    ?= 8000
TRIALS      ?= 5
IMAGE       ?= telecom-churn-api

.PHONY: help install data-pipeline data-pipeline-rebuild train-pipeline train-pipeline-fast \
        train-baseline streaming-inference stream-demo batch-inference run-all serve test \
        mlflow-ui stop-all docker-build docker-run clean clean-all

help:
	@echo "Setup"
	@echo "  make install               Create .venv and install dependencies"
	@echo "Pipelines"
	@echo "  make data-pipeline         Clean, split 60/20/20, fit features on train (reuses artifacts)"
	@echo "  make data-pipeline-rebuild Force a full rebuild of the data artifacts"
	@echo "  make train-pipeline        Full Optuna tuning (50/100/100 trials - slow, research-grade)"
	@echo "  make train-pipeline-fast   Quick tuning (TRIALS=$(TRIALS) per model)"
	@echo "  make train-baseline        No tuning - research baseline hyperparameters"
	@echo "  make streaming-inference   Score one sample customer"
	@echo "  make stream-demo           Replay 25 customers as a stream (MLflow monitoring)"
	@echo "  make batch-inference       Score the whole raw CSV -> artifacts/predictions/"
	@echo "  make run-all               Data -> train (fast) -> inference"
	@echo "Serving & ops"
	@echo "  make serve                 Start the FastAPI service on :$(API_PORT) (docs at /docs)"
	@echo "  make test                  Run the pytest suite"
	@echo "  make mlflow-ui             MLflow UI on :$(MLFLOW_PORT)"
	@echo "  make docker-build / docker-run"
	@echo "  make clean                 Remove generated artifacts (keeps raw data)"

install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "✅ Done. Activate with: source $(VENV)/bin/activate"

data-pipeline:
	$(PYTHON) pipelines/data_pipeline.py

data-pipeline-rebuild:
	$(PYTHON) pipelines/data_pipeline.py --force

train-pipeline:
	$(PYTHON) pipelines/training_pipeline.py

train-pipeline-fast:
	$(PYTHON) pipelines/training_pipeline.py --n-trials $(TRIALS)

train-baseline:
	$(PYTHON) pipelines/training_pipeline.py --no-tune

streaming-inference:
	$(PYTHON) pipelines/streaming_inference_pipeline.py

stream-demo:
	$(PYTHON) pipelines/streaming_inference_pipeline.py --stream 25

batch-inference:
	$(PYTHON) pipelines/streaming_inference_pipeline.py --batch-file data/raw/Dataset-Telco-Customer-Churn.csv

run-all:
	@echo "========== Step 1: Data pipeline =========="
	$(PYTHON) pipelines/data_pipeline.py --force
	@echo "========== Step 2: Training pipeline =========="
	$(PYTHON) pipelines/training_pipeline.py --n-trials $(TRIALS)
	@echo "========== Step 3: Inference =========="
	$(PYTHON) pipelines/streaming_inference_pipeline.py
	@echo "✅ All pipelines completed"

serve:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

test:
	$(PYTHON) -m pytest tests -v

mlflow-ui:
	@echo "MLflow UI: http://localhost:$(MLFLOW_PORT)"
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port $(MLFLOW_PORT)

stop-all:
	-@lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true
	-@lsof -ti:$(API_PORT) | xargs kill -9 2>/dev/null || true
	@echo "✅ MLflow and API servers stopped"

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(API_PORT):8000 $(IMAGE)

clean:
	rm -rf artifacts/* data/processed/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "✅ Artifacts removed (MLflow history kept; use 'make clean-all' to remove it too)"

clean-all: clean
	rm -rf mlruns mlflow.db
