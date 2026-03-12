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
DIM_RED ?= none
N_COMPONENTS ?= 10

RUNTIME_ROOT = /tmp/tpv-$(USER)
RUNTIME_HOME = $(RUNTIME_ROOT)/home
MNE_HOME = $(RUNTIME_ROOT)/mne

.DEFAULT_GOAL := help

.PHONY: help venv install main train predict benchmark check-env clean clean-local-runtime

help:
	@echo "Commands:"
	@echo "  make train SUBJECT=1 RUNS='4'"
	@echo "  make predict SUBJECT=1 RUNS='4'"
	@echo "  make main SUBJECT=1 RUNS='4'"
	@echo "  make plot SUBJECT=1 RUNS='4'"
	@echo "  make train SUBJECT=1 RUNS='4' DIM_RED=pca N_COMPONENTS=10"
	@echo "  make train SUBJECT=1 RUNS='4' DIM_RED=csp N_COMPONENTS=4"
	@echo "  make benchmark SUBJECT=1 RUNS='4'"
	@echo "  make install"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

main:
	mkdir -p "$(RUNTIME_HOME)"
	mkdir -p "$(MNE_HOME)"
	HOME="$(RUNTIME_HOME)" MNE_HOME="$(MNE_HOME)" "$(VENV_PY)" src/main.py "$(SUBJECT)" $(RUNS) --path "$(DATA_PATH)" --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)"

plot:
	mkdir -p "$(RUNTIME_HOME)"
	mkdir -p "$(MNE_HOME)"
	HOME="$(RUNTIME_HOME)" MNE_HOME="$(MNE_HOME)" "$(VENV_PY)" src/main.py "$(SUBJECT)" $(RUNS) --path "$(DATA_PATH)" --plot --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)"

train:
	mkdir -p "$(RUNTIME_HOME)"
	mkdir -p "$(MNE_HOME)"
	HOME="$(RUNTIME_HOME)" MNE_HOME="$(MNE_HOME)" "$(VENV_PY)" src/train.py "$(SUBJECT)" $(RUNS) --path "$(DATA_PATH)" --cvs "$(CVS)" --test-size "$(TEST_SIZE)" --val-size "$(VAL_SIZE)" --seed "$(SEED)" --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)" $(if $(MODEL_OUT),--model-out "$(MODEL_OUT)")

predict:
	mkdir -p "$(RUNTIME_HOME)"
	mkdir -p "$(MNE_HOME)"
	HOME="$(RUNTIME_HOME)" MNE_HOME="$(MNE_HOME)" "$(VENV_PY)" src/predict.py "$(SUBJECT)" $(RUNS) --path "$(DATA_PATH)" --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)" $(if $(MODEL_OUT),--model "$(MODEL_OUT)")

benchmark:
	mkdir -p "$(RUNTIME_HOME)"
	mkdir -p "$(MNE_HOME)"
	HOME="$(RUNTIME_HOME)" MNE_HOME="$(MNE_HOME)" "$(VENV_PY)" src/benchmark.py "$(SUBJECT)" $(RUNS) --path "$(DATA_PATH)" --cvs "$(CVS)" --test-size "$(TEST_SIZE)" --val-size "$(VAL_SIZE)" --seed "$(SEED)"

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
