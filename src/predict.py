import argparse
import time

import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score

from pipeline_config import parse_runs, pipeline_suffix
from preprocessing import load_subject_epochs


def default_model_path(subject, runs, dim_red, n_components):
    runs_slug = "all" if runs == list(range(1, 15)) else "-".join(f"{r:02d}" for r in runs)
    variant_slug = pipeline_suffix(dim_red, n_components)
    return f"models/s{subject:03d}_runs_{runs_slug}_{variant_slug}.joblib"


def run_playback_prediction(
    subject,
    runs,
    base_path=None,
    model_path=None,
    dim_red="none",
    n_components=10,
    max_latency=2.0,
    verbose=True,
):
    resolved_model_path = model_path or default_model_path(subject, runs, dim_red, n_components)
    bundle = load(resolved_model_path)
    pipeline = bundle["pipeline"]

    X, y = load_subject_epochs(subject_id=subject, runs=runs, base_path=base_path, plot=False)
    if X is None or y is None:
        raise ValueError("No data loaded. Check subject/runs/path.")

    predictions = []
    latencies = []

    if verbose:
        print("\n--- PLAYBACK PREDICTION ---")
        print(f"Model: {resolved_model_path}")
        print(f"Epochs: {X.shape[0]}")

    for idx in range(X.shape[0]):
        epoch = X[idx : idx + 1]
        truth = int(y[idx])

        start = time.perf_counter()
        pred = int(pipeline.predict(epoch)[0])
        latency = time.perf_counter() - start

        predictions.append(pred)
        latencies.append(latency)

        if verbose:
            status = "True" if pred == truth else "False"
            print(
                f"Epoch {idx:02d}: prediction={pred} truth={truth} "
                f"correct={status} latency={latency:.4f}s"
            )

    predictions = np.array(predictions)
    accuracy = accuracy_score(y, predictions)
    mean_latency = float(np.mean(latencies))
    observed_max_latency = float(np.max(latencies))
    latency_target_ok = observed_max_latency < max_latency

    if verbose:
        print("\n--- SUMMARY ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Mean latency: {mean_latency:.4f}s")
        print(f"Max latency:  {observed_max_latency:.4f}s")
        print(f"Latency target ({max_latency:.2f}s): {'OK' if latency_target_ok else 'FAILED'}")

def main():
    parser = argparse.ArgumentParser(description="Run EEG BCI predictions.")
    parser.add_argument("subject", type=int, help="Subject ID (e.g. 1 for S001)")
    parser.add_argument(
        "runs",
        type=str,
        nargs="+",
        help="Runs to use for prediction (e.g. 4 8 12) or 'all'",
    )
    parser.add_argument("--path", type=str, default=None, help="Dataset base path")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (.joblib)")
    parser.add_argument("--dim-red", choices=["none", "pca", "csp"], default="none", help="Dimensionality reduction method")
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCA or CSP components")
    parser.add_argument(
        "--max-latency",
        type=float,
        default=2.0,
        help="Maximum target latency per epoch in seconds",
    )
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    run_playback_prediction(
        subject=args.subject,
        runs=runs,
        base_path=args.path,
        model_path=args.model,
        dim_red=args.dim_red,
        n_components=args.n_components,
        max_latency=args.max_latency,
        verbose=True,
    )


if __name__ == "__main__":
    main()
