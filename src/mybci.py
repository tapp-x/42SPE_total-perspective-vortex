import argparse
import sys

from global_eval import run_full_evaluation


def print_usage():
    print("Usage:")
    print("  python src/mybci.py [options]")
    print("  python src/mybci.py <subject> <run> [runs ...] train [options]")
    print("  python src/mybci.py <subject> <run> [runs ...] predict [options]")
    print("  python src/mybci.py <subject> <run> [runs ...] benchmark [options]")
    print("  python src/mybci.py benchmark --subjects 1 2 3 --runs 4 8 12 [options]")


def run_global(argv):
    parser = argparse.ArgumentParser(description="Run the global BCI evaluation.")
    parser.add_argument("--path", type=str, default=None, help="Dataset base path")
    parser.add_argument("--dim-red", choices=["none", "pca", "csp"], default="csp", help="Dimensionality reduction method")
    parser.add_argument("--n-components", type=int, default=5, help="Number of PCA or CSP components")
    parser.add_argument("--cvs", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    run_full_evaluation(
        base_path=args.path,
        test_size=args.test_size,
        val_size=args.val_size,
        cvs=args.cvs,
        seed=args.seed,
        dim_red=args.dim_red,
        n_components=args.n_components,
    )


def find_subject_style_command(argv):
    for index, token in enumerate(argv):
        if token in {"train", "predict", "benchmark"}:
            return index, token
    return None, None


def dispatch_subject_style(argv):
    command_index, command = find_subject_style_command(argv)
    if command_index is None or command_index < 2:
        print_usage()
        raise SystemExit(2)

    subject = argv[0]
    runs = argv[1:command_index]
    options = argv[command_index + 1 :]

    if command == "train":
        from train import main as train_main

        sys.argv = ["train.py", subject, *runs, *options]
        train_main()
        return

    if command == "predict":
        from predict import main as predict_main

        sys.argv = ["predict.py", subject, *runs, *options]
        predict_main()
        return

    from benchmark import main as benchmark_main

    sys.argv = ["benchmark.py", "--subjects", subject, "--runs", *runs, *options]
    benchmark_main()


def main():
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("--"):
        run_global(argv)
        return

    if argv[0] == "benchmark":
        from benchmark import main as benchmark_main

        sys.argv = ["benchmark.py", *argv[1:]]
        benchmark_main()
        return

    dispatch_subject_style(argv)


if __name__ == "__main__":
    main()
