import argparse
import os
from dataclasses import dataclass

import numpy as np
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from pipeline_config import build_pipeline, parse_runs, pipeline_suffix
from preprocessing import load_subject_epochs


@dataclass
class TrainingResult:
    pipeline: object
    subject: int
    runs: list
    dim_red: str
    n_components: int
    cv_scores: np.ndarray
    val_accuracy: float
    test_accuracy: float
    n_epochs: int
    classes: list


def get_valid_stratified_cv_folds(y, requested_cv):
    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(np.min(counts))
    return max(2, min(requested_cv, min_class_count))


def default_model_path(subject, runs, dim_red, n_components):
    runs_slug = "all" if runs == list(range(1, 15)) else "-".join(f"{r:02d}" for r in runs)
    variant_slug = pipeline_suffix(dim_red, n_components)
    return f"models/s{subject:03d}_runs_{runs_slug}_{variant_slug}.joblib"


def train_and_evaluate(
    subject,
    runs,
    base_path=None,
    test_size=0.2,
    val_size=0.2,
    cvs=5,
    seed=42,
    dim_red="csp",
    n_components=5,
    verbose=True,
):
    X, y = load_subject_epochs(subject_id=subject, runs=runs, base_path=base_path, plot=False)

    if X is None or y is None:
        raise ValueError("No data loaded. Check subject/runs/path.")

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 classes to train, found: {classes.tolist()}")

    if verbose:
        print("\n--- DATA SUMMARY ---")
        print(f"X shape (epochs, channels, time): {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Classes: {classes.tolist()}")
        print(f"Dimensionality reduction: {dim_red}")

    cv_folds = get_valid_stratified_cv_folds(y, cvs)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    pipeline = build_pipeline(dim_red=dim_red, n_components=n_components)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")

    if verbose:
        print("\n--- CROSS VALIDATION ---")
        print(f"Scores: {np.round(cv_scores, 4)}")
        print(f"Mean accuracy: {cv_scores.mean():.4f}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    val_ratio_in_train_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_in_train_val,
        random_state=seed,
        stratify=y_train_val,
    )

    if verbose:
        print("\n--- SPLIT ---")
        print(f"Train: {X_train.shape[0]} epochs")
        print(f"Val:   {X_val.shape[0]} epochs")
        print(f"Test:  {X_test.shape[0]} epochs")

    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)

    val_acc = accuracy_score(y_val, y_val_pred)

    final_pipeline = clone(pipeline)
    final_pipeline.fit(X_train_val, y_train_val)
    y_test_pred = final_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    if verbose:
        print("\n--- HOLDOUT METRICS ---")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Test accuracy:       {test_acc:.4f}")

    return TrainingResult(
        pipeline=final_pipeline,
        subject=subject,
        runs=runs,
        dim_red=dim_red,
        n_components=n_components,
        cv_scores=cv_scores,
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        n_epochs=X.shape[0],
        classes=classes.tolist(),
    )


def save_training_result(result, model_out):
    model_dir = os.path.dirname(model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    dump(
        {
            "pipeline": result.pipeline,
            "subject": result.subject,
            "runs": result.runs,
            "dim_red": result.dim_red,
            "n_components": result.n_components,
            "cv_scores": result.cv_scores,
            "val_accuracy": result.val_accuracy,
            "test_accuracy": result.test_accuracy,
        },
        model_out,
    )


def main():
    parser = argparse.ArgumentParser(description="Train EEG BCI pipeline.")
    parser.add_argument("subject", type=int, help="Subject ID (e.g. 1 for S001)")
    parser.add_argument(
        "runs",
        type=str,
        nargs="+",
        help="Runs to use (e.g. 4 8 12) or 'all'",
    )
    parser.add_argument("--path", type=str, default=None, help="Dataset base path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation split ratio on full dataset",
    )
    parser.add_argument("--cvs", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dim-red", choices=["none", "pca", "csp"], default="csp", help="Dimensionality reduction method")
    parser.add_argument("--n-components", type=int, default=5, help="Number of PCA or CSP components")
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="Output path for trained model (.joblib)",
    )
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    result = train_and_evaluate(
        subject=args.subject,
        runs=runs,
        base_path=args.path,
        test_size=args.test_size,
        val_size=args.val_size,
        cvs=args.cvs,
        seed=args.seed,
        dim_red=args.dim_red,
        n_components=args.n_components,
        verbose=True,
    )

    if args.model_out is None:
        model_out = default_model_path(args.subject, runs, args.dim_red, args.n_components)
    else:
        model_out = args.model_out

    save_training_result(result, model_out)
    print(f"\nModel saved to: {model_out}")


if __name__ == "__main__":
    main()
