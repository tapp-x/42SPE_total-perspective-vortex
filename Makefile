SHELL := /bin/bash

PYTHON = python3
VENV = venv_tpv
VENV_PY = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

SUBJECT ?= 1
RUNS ?= 4
DATA_PATH ?= data/files

CVS ?= 5
TEST_SIZE ?= 0.2
VAL_SIZE ?= 0.2
SEED ?= 42
MODEL_OUT ?=
DIM_RED ?= csp
N_COMPONENTS ?= 5

RUNTIME_ROOT = /tmp/tpv-$(USER)
RUNTIME_HOME = $(RUNTIME_ROOT)/home
MNE_HOME = $(RUNTIME_ROOT)/mne
RUN_ENV = HOME="$(RUNTIME_HOME)" MNE_HOME="$(MNE_HOME)" "$(VENV_PY)"

MAIN_CMD = $(RUN_ENV) src/inspect_preprocessing.py
TRAIN_CMD = $(RUN_ENV) src/train.py
PREDICT_CMD = $(RUN_ENV) src/predict.py
BENCH_CMD = $(RUN_ENV) src/benchmark.py
MYBCI_CMD = $(RUN_ENV) src/mybci.py
IMPORT_CMD = $(RUN_ENV) src/import_data.py

BCI_ARGS = --path "$(DATA_PATH)" --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)"
TRAIN_KW = --cvs "$(CVS)" --test-size "$(TEST_SIZE)" --val-size "$(VAL_SIZE)" --seed "$(SEED)"
EVAL_KW = --cvs "$(CVS)" --test-size "$(TEST_SIZE)" --val-size "$(VAL_SIZE)" --seed "$(SEED)"

.DEFAULT_GOAL := help

.PHONY: help venv install inspect train predict benchmark mybci import-data check-env clean clean-local-runtime

runtime:
	mkdir -p "$(RUNTIME_HOME)"
	mkdir -p "$(MNE_HOME)"

help:
	@echo "Commands:"
	@echo "  make train SUBJECT=1 RUNS='4'"
	@echo "  make predict SUBJECT=1 RUNS='4'"
	@echo "  make inspect SUBJECT=1 RUNS='4'"
	@echo "  make train SUBJECT=1 RUNS='4' DIM_RED=csp N_COMPONENTS=5"
	@echo "  make benchmark SUBJECT=1 RUNS='4'"
	@echo "  make mybci"
	@echo "  make mybci DIM_RED=csp"
	@echo "  make import-data SUBJECT=1 RUNS='4'"
	@echo "  make install"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

inspect: runtime
	$(MAIN_CMD) "$(SUBJECT)" $(RUNS) --plot $(BCI_ARGS)

train: runtime
	$(TRAIN_CMD) "$(SUBJECT)" $(RUNS) $(BCI_ARGS) $(TRAIN_KW) $(if $(MODEL_OUT),--model-out "$(MODEL_OUT)")

predict: runtime
	$(PREDICT_CMD) "$(SUBJECT)" $(RUNS) $(BCI_ARGS) $(if $(MODEL_OUT),--model "$(MODEL_OUT)")

benchmark: runtime
	$(BENCH_CMD) --subjects $(SUBJECT) --runs $(RUNS) $(EVAL_KW) --quiet

mybci: runtime
	$(MYBCI_CMD) $(BCI_ARGS) $(EVAL_KW)
import-data: runtime
	$(IMPORT_CMD) --subjects $(SUBJECT) --runs $(RUNS) --path "$(DATA_PATH)"

check-env:
	@echo "SUBJECT=$(SUBJECT)"
	@echo "RUNS=$(RUNS)"
	@echo "DATA_PATH=$(DATA_PATH)"
	@echo "CVS=$(CVS)"
	@echo "TEST_SIZE=$(TEST_SIZE)"
	@echo "VAL_SIZE=$(VAL_SIZE)"
	@echo "SEED=$(SEED)"
	@echo "MODEL_OUT=$(MODEL_OUT)"
	@echo "DIM_RED=$(DIM_RED)"
	@echo "N_COMPONENTS=$(N_COMPONENTS)"
	@echo "VENV=$(VENV)"

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-local-runtime:
	rm -rf .home .mne
