import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score

from pipeline_config import parse_runs, pipeline_suffix
from preprocessing import load_subject_epochs


@dataclass
class PlaybackChunk:
    index: int
    epoch: np.ndarray
    truth: int


def default_model_path(subject, runs, dim_red, n_components):
    runs_slug = "all" if runs == list(range(1, 15)) else "-".join(f"{r:02d}" for r in runs)
    variant_slug = pipeline_suffix(dim_red, n_components)
    return f"models/s{subject:03d}_runs_{runs_slug}_{variant_slug}.joblib"


def iter_playback_chunks(X, y):
    for idx in range(X.shape[0]):
        yield PlaybackChunk(index=idx, epoch=X[idx : idx + 1], truth=int(y[idx]))


def run_playback_prediction(
    subject,
    runs,
    base_path=None,
    model_path=None,
    dim_red="csp",
    n_components=5,
    max_latency=2.0,
    verbose=True,
):
    resolved_model_path = model_path or default_model_path(subject, runs, dim_red, n_components)
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            f"Model file not found: {resolved_model_path}. "
            "Train the model first or pass --model with a valid path."
        )
    bundle = load(resolved_model_path)
    pipeline = bundle["pipeline"]

    X, y = load_subject_epochs(subject_id=subject, runs=runs, base_path=base_path, plot=False)
    if X is None or y is None:
        raise ValueError("No data loaded. Check subject/runs/path.")

    predictions = []
    latencies = []
    deadline_misses = 0

    if verbose:
        print("\n--- PLAYBACK PREDICTION ---")
        print(f"Model: {resolved_model_path}")
        print(f"Epochs: {X.shape[0]}")
        print(f"Max latency per chunk: {max_latency:.2f}s")

    for chunk in iter_playback_chunks(X, y):
        start = time.perf_counter()
        pred = int(pipeline.predict(chunk.epoch)[0])
        latency = time.perf_counter() - start

        predictions.append(pred)
        latencies.append(latency)
        deadline_ok = latency < max_latency
        if not deadline_ok:
            deadline_misses += 1

        if verbose:
            status = "True" if pred == chunk.truth else "False"
            deadline_status = "OK" if deadline_ok else "FAILED"
            print(
                f"Chunk {chunk.index:02d}: prediction={pred} truth={chunk.truth} "
                f"correct={status} latency={latency:.4f}s deadline={deadline_status}"
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
        print(f"Deadline misses: {deadline_misses}/{len(latencies)}")
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
    parser.add_argument("--dim-red", choices=["none", "pca", "csp"], default="csp", help="Dimensionality reduction method")
    parser.add_argument("--n-components", type=int, default=5, help="Number of PCA or CSP components")
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
