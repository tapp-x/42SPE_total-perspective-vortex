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
ACTION ?= train
VARIANTS ?= none pca:5 csp:4
ARGS ?=

RUNTIME_ROOT = /tmp/tpv-$(USER)
RUNTIME_HOME = $(RUNTIME_ROOT)/home
MNE_HOME = $(RUNTIME_ROOT)/mne
RUN_ENV = HOME="$(RUNTIME_HOME)" MNE_HOME="$(MNE_HOME)" "$(VENV_PY)"

.DEFAULT_GOAL := help

.PHONY: help venv install main train predict benchmark mybci benchmark-variants import-data check-env clean clean-local-runtime

runtime:
	mkdir -p "$(RUNTIME_HOME)"
	mkdir -p "$(MNE_HOME)"

help:
	@echo "Commands:"
	@echo "  make train SUBJECT=1 RUNS='4'"
	@echo "  make predict SUBJECT=1 RUNS='4'"
	@echo "  make main SUBJECT=1 RUNS='4'"
	@echo "  make plot SUBJECT=1 RUNS='4'"
	@echo "  make train SUBJECT=1 RUNS='4' DIM_RED=pca N_COMPONENTS=10"
	@echo "  make train SUBJECT=1 RUNS='4' DIM_RED=csp N_COMPONENTS=4"
	@echo "  make benchmark SUBJECT=1 RUNS='4'"
	@echo "  make mybci SUBJECT=1 RUNS='4' ACTION=train ARGS='--cvs 3'"
	@echo "  make mybci SUBJECT=1 RUNS='4' ACTION=predict"
	@echo "  make benchmark-variants SUBJECT='1 2 3' RUNS='6 10 14' VARIANTS='none csp:2 csp:4 csp:6'"
	@echo "  make import-data SUBJECT=1 RUNS='4'"
	@echo "  make install"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

main: runtime
	$(RUN_ENV) src/main.py "$(SUBJECT)" $(RUNS) --path "$(DATA_PATH)" --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)"

plot: runtime
	$(RUN_ENV) src/main.py "$(SUBJECT)" $(RUNS) --path "$(DATA_PATH)" --plot --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)"

train: runtime
	$(RUN_ENV) src/train.py "$(SUBJECT)" $(RUNS) --path "$(DATA_PATH)" --cvs "$(CVS)" --test-size "$(TEST_SIZE)" --val-size "$(VAL_SIZE)" --seed "$(SEED)" --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)" $(if $(MODEL_OUT),--model-out "$(MODEL_OUT)")

predict: runtime
	$(RUN_ENV) src/predict.py "$(SUBJECT)" $(RUNS) --path "$(DATA_PATH)" --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)" $(if $(MODEL_OUT),--model "$(MODEL_OUT)")

benchmark: runtime
	$(RUN_ENV) src/benchmark.py --subjects $(SUBJECT) --runs $(RUNS) --path "$(DATA_PATH)" --cvs "$(CVS)" --test-size "$(TEST_SIZE)" --val-size "$(VAL_SIZE)" --seed "$(SEED)" --quiet

mybci: runtime
	$(RUN_ENV) mybci.py "$(SUBJECT)" $(RUNS) "$(ACTION)" --path "$(DATA_PATH)" --dim-red "$(DIM_RED)" --n-components "$(N_COMPONENTS)" $(ARGS)

benchmark-variants: runtime
	$(RUN_ENV) mybci.py benchmark --subjects $(SUBJECT) --runs $(RUNS) --path "$(DATA_PATH)" --variants $(VARIANTS) --cvs "$(CVS)" --test-size "$(TEST_SIZE)" --val-size "$(VAL_SIZE)" --seed "$(SEED)" --quiet

import-data: runtime
	$(RUN_ENV) src/import_data.py --subjects $(SUBJECT) --runs $(RUNS) --path "$(DATA_PATH)"

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
	@echo "ACTION=$(ACTION)"
	@echo "VARIANTS=$(VARIANTS)"
	@echo "ARGS=$(ARGS)"
	@echo "VENV=$(VENV)"

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-local-runtime:
	rm -rf .home .mne
