# 42SPE_total-perspective-vortex

Brain-computer interface project based on EEG data from PhysioNet.

The mandatory part of the project is implemented around a full sklearn pipeline:

- EDF loading and EEG preprocessing with MNE
- band-pass filtering
- epoch extraction for the target classes
- CSP dimensionality reduction
- linear SVM classification
- train / validation / test workflow
- playback-style prediction with per-chunk latency measurement

## Project layout

- `src/preprocessing.py`: load EDF files, filter signals, extract epochs
- `src/csp.py`: custom CSP transformer used in the main pipeline
- `src/features.py`: spectral feature extractor used by the non-CSP variants
- `src/pipeline_config.py`: build the sklearn pipeline and parse runs
- `src/train.py`: training, cross-validation, validation, test, model saving
- `src/predict.py`: model loading and playback-style prediction
- `src/inspect_preprocessing.py`: preprocessing and pipeline inspection helper
- `src/mybci.py`: unified entry point
- `src/import_data.py`: download PhysioNet EEGBCI files

## Installation

```bash
make install
```

## Dataset

The code expects the EEGBCI dataset in `data/files` by default.

You can also pass another path with `--path` or set `EEG_DATA_PATH`.

## Common commands

Train a model:

```bash
make train SUBJECT=1 RUNS='4'
```

Run playback prediction:

```bash
make predict SUBJECT=1 RUNS='4'
```

Inspect preprocessing and feature shapes, with raw and filtered plots:

```bash
make inspect SUBJECT=1 RUNS='4'
```

Import data:

```bash
make import-data SUBJECT=1 RUNS='4'
```

Benchmark a subject:

```bash
make benchmark SUBJECT=1 RUNS='4'
```

## Notes

- `csp` is the main mandatory path.
- `pca` and `none` are extra variants that remain available, but the mandatory work is centered on CSP.
- Trained models are saved in `models/`.
- Benchmark outputs are saved in `results/`.
