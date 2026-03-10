import argparse
import time

import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score

from pipeline_config import parse_runs
from preprocessing import preprocessing


def default_model_path(subject, runs):
    runs_slug = "all" if runs == list(range(1, 15)) else "-".join(f"{r:02d}" for r in runs)
    return f"models/s{subject:03d}_runs_{runs_slug}.joblib"


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
    parser.add_argument(
        "--max-latency",
        type=float,
        default=2.0,
        help="Maximum target latency per epoch in seconds",
    )
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    model_path = args.model or default_model_path(args.subject, runs)
    bundle = load(model_path)
    pipeline = bundle["pipeline"]

    X, y = preprocessing(subject_id=args.subject, runs=runs, base_path=args.path, plot=False)
    if X is None or y is None:
        print("No data loaded. Check subject/runs/path.")
        return

    predictions = []
    latencies = []

    print("\n--- PLAYBACK PREDICTION ---")
    print(f"Model: {model_path}")
    print(f"Epochs: {X.shape[0]}")

    for idx in range(X.shape[0]):
        epoch = X[idx : idx + 1]
        truth = int(y[idx])

        start = time.perf_counter()
        pred = int(pipeline.predict(epoch)[0])
        latency = time.perf_counter() - start

        predictions.append(pred)
        latencies.append(latency)

        status = "True" if pred == truth else "False"
        print(f"Epoch {idx:02d}: prediction={pred} truth={truth} correct={status} latency={latency:.4f}s")

    predictions = np.array(predictions)
    accuracy = accuracy_score(y, predictions)
    mean_latency = float(np.mean(latencies))
    max_latency = float(np.max(latencies))

    print("\n--- SUMMARY ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean latency: {mean_latency:.4f}s")
    print(f"Max latency:  {max_latency:.4f}s")
    print(f"Latency target ({args.max_latency:.2f}s): {'OK' if max_latency < args.max_latency else 'FAILED'}")


if __name__ == "__main__":
    main()
