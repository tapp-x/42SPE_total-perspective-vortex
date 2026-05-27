import argparse

import matplotlib

matplotlib.use("qtagg")

from pipeline_config import parse_runs
from preprocessing import load_subject_epochs


def main():
    """Preview preprocessing, plots, and data shapes."""

    parser = argparse.ArgumentParser(description="Preview EEG preprocessing.")
    parser.add_argument("subject", type=int, help="Subject number")
    parser.add_argument("runs", type=str, nargs="+", help="Runs to load, or all")
    parser.add_argument("--path", type=str, default=None, help="Dataset path")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show raw and filtered plots",
    )
    args = parser.parse_args()

    try:
        target_runs = parse_runs(args.runs)
    except ValueError:
        print("Invalid run list.")
        return

    X, y = load_subject_epochs(subject_id=args.subject, runs=target_runs, base_path=args.path, plot=args.plot)

    if X is None:
        print("No data loaded.")
        return

    print(f"\nSubject S{args.subject:03d}")
    print(f"Runs: {target_runs}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    if args.plot:
        print("Raw and filtered plots shown for the first available run.")


if __name__ == "__main__":
    main()
