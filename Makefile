SHELL := /bin/bash

PYTHON = python3
VENV = venv_tpv
VENV_PY = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

SUBJECT ?= 1
RUNS ?= 4
DATA_PATH ?= data/files
DIM_RED ?= csp
N_COMPONENTS ?= 5
CVS ?= 5
MODEL_OUT ?=

RUNTIME_ROOT = /tmp/tpv-$(USER)
RUNTIME_HOME = $(RUNTIME_ROOT)/home
MNE_HOME = $(RUNTIME_ROOT)/mne
RUN_ENV = HOME="$(RUNTIME_HOME)" MNE_HOME="$(MNE_HOME)" "$(VENV_PY)"

MAIN_CMD = $(RUN_ENV) src/preprocessing_preview.py
TRAIN_CMD = $(RUN_ENV) src/train.py
PREDICT_CMD = $(RUN_ENV) src/predict.py
BCI_ARGS = --path "$(DATA_PATH)" --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)"
INSPECT_ARGS = --path "$(DATA_PATH)"
TRAIN_KW = --cvs "$(CVS)"

.DEFAULT_GOAL := help

.PHONY: help venv install inspect train predict clean

runtime:
	mkdir -p "$(RUNTIME_HOME)"
	mkdir -p "$(MNE_HOME)"

help:
	@echo "Commands:"
	@echo "  make train SUBJECT=1 RUNS='4'"
	@echo "  make predict SUBJECT=1 RUNS='4'"
	@echo "  make inspect SUBJECT=1 RUNS='4'"
	@echo "  make train SUBJECT=1 RUNS='4' DIM_RED=csp N_COMPONENTS=5"
	@echo "  make install"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

inspect: runtime
	$(MAIN_CMD) "$(SUBJECT)" $(RUNS) --plot $(INSPECT_ARGS)

train: runtime
	$(TRAIN_CMD) "$(SUBJECT)" $(RUNS) $(BCI_ARGS) $(TRAIN_KW) $(if $(MODEL_OUT),--model-out "$(MODEL_OUT)")

predict: runtime
	$(PREDICT_CMD) "$(SUBJECT)" $(RUNS) $(BCI_ARGS) $(if $(MODEL_OUT),--model "$(MODEL_OUT)")

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
