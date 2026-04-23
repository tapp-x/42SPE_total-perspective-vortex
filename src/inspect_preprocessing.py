import argparse

import matplotlib

matplotlib.use("qtagg")

from pipeline_config import build_pipeline, parse_runs
from preprocessing import load_subject_epochs


def main():
    parser = argparse.ArgumentParser(description="Inspect EEG preprocessing and pipeline shapes.")
    parser.add_argument("subject", type=int, help="Subject ID (e.g., 1 for S001)")
    parser.add_argument("runs", type=str, nargs="+", help="List of runs to process (e.g., 4 8 12) or all")
    parser.add_argument("--path", type=str, default=None, help="Base path to the dataset")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show raw and filtered plots for the first available run",
    )
    parser.add_argument("--dim-red", choices=["none", "pca", "csp"], default="csp", help="Dimensionality reduction method")
    parser.add_argument("--n-components", type=int, default=5, help="Number of components for PCA or CSP")
    args = parser.parse_args()

    try:
        target_runs = parse_runs(args.runs)
    except ValueError:
        print("Error: invalid run numbers.")
        return

    X, y = load_subject_epochs(subject_id=args.subject, runs=target_runs, base_path=args.path, plot=args.plot)

    if X is None:
        return

    print("\n--- Preprocessing complete ---")
    print(f"X shape (3D): {X.shape}")
    print(f"y shape    : {y.shape}")
    if args.plot:
        print("Displayed raw and filtered plots for the first available run.")

    print("\n--- Pipeline check ---")
    pipeline = build_pipeline(dim_red=args.dim_red, n_components=args.n_components)

    if args.dim_red == "csp":
        reducer = pipeline.named_steps["dimensionality_reduction"]
        X_reduced = reducer.fit_transform(X, y)
        print(f"X after CSP ({args.n_components} components): {X_reduced.shape}")
    else:
        extractor = pipeline.named_steps["feature_extraction"]
        X_2D = extractor.fit_transform(X)
        print(f"X after spectral extraction: {X_2D.shape}")

        if args.dim_red == "pca":
            reducer = pipeline.named_steps["dimensionality_reduction"]
            X_reduced = reducer.fit_transform(X_2D)
            print(f"X after PCA ({args.n_components} components): {X_reduced.shape}")


if __name__ == "__main__":
    main()
