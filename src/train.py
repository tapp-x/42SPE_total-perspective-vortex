import argparse
import os

import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from pipeline_config import build_pipeline, parse_runs
from preprocessing import preprocessing


def infer_cv_folds(y, requested_cv):
    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(np.min(counts))
    return max(2, min(requested_cv, min_class_count))


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
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="Output path for trained model (.joblib)",
    )
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    X, y = preprocessing(subject_id=args.subject, runs=runs, base_path=args.path, plot=False)

    if X is None or y is None:
        print("No data loaded. Check subject/runs/path.")
        return

    classes = np.unique(y)
    if len(classes) < 2:
        print(f"Need at least 2 classes to train, found: {classes.tolist()}")
        return

    print("\n--- DATA SUMMARY ---")
    print(f"X shape (epochs, channels, time): {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {classes.tolist()}")

    cv_folds = infer_cv_folds(y, args.cvs)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
    pipeline = build_pipeline()
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")

    print("\n--- CROSS VALIDATION ---")
    print(f"Scores: {np.round(cv_scores, 4)}")
    print(f"Mean accuracy: {cv_scores.mean():.4f}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    val_ratio_in_train_val = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_in_train_val,
        random_state=args.seed,
        stratify=y_train_val,
    )

    print("\n--- SPLIT ---")
    print(f"Train: {X_train.shape[0]} epochs")
    print(f"Val:   {X_val.shape[0]} epochs")
    print(f"Test:  {X_test.shape[0]} epochs")

    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n--- HOLDOUT METRICS ---")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")

    if args.model_out is None:
        runs_slug = "all" if runs == list(range(1, 15)) else "-".join(f"{r:02d}" for r in runs)
        model_out = f"models/s{args.subject:03d}_runs_{runs_slug}.joblib"
    else:
        model_out = args.model_out

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    dump(
        {
            "pipeline": pipeline,
            "subject": args.subject,
            "runs": runs,
            "cv_scores": cv_scores,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
        },
        model_out,
    )
    print(f"\nModel saved to: {model_out}")


if __name__ == "__main__":
    main()
