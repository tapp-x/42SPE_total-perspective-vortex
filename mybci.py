import argparse
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def print_usage():
    print("Usage:")
    print("  python mybci.py <subject> <run> [runs ...] train [options]")
    print("  python mybci.py <subject> <run> [runs ...] predict [options]")
    print("  python mybci.py <subject> <run> [runs ...] benchmark [options]")
    print("  python mybci.py benchmark --subjects 1 2 3 --runs 4 8 12 [options]")


def add_common_args(parser):
    parser.add_argument("--path", type=str, default=None, help="Dataset base path")
    parser.add_argument(
        "--dim-red",
        choices=["none", "pca", "csp"],
        default="none",
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of PCA or CSP components",
    )


def parse_subject_style(argv):
    from pipeline_config import parse_runs

    for index, token in enumerate(argv):
        if token not in {"train", "predict", "benchmark"}:
            continue

        if index < 2:
            return None

        subject = int(argv[0])
        runs = parse_runs(argv[1:index])
        command = token
        options = argv[index + 1 :]
        return subject, runs, command, options

    return None


def run_train(subject, runs, option_argv):
    from train import default_model_path, save_training_result, train_and_evaluate

    parser = argparse.ArgumentParser(description="Train the EEG BCI pipeline.")
    add_common_args(parser)
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation split ratio on full dataset",
    )
    parser.add_argument("--cvs", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-out", type=str, default=None, help="Output path for trained model")
    args = parser.parse_args(option_argv)

    result = train_and_evaluate(
        subject=subject,
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

    model_out = args.model_out or default_model_path(
        subject,
        runs,
        args.dim_red,
        args.n_components,
    )
    save_training_result(result, model_out)
    print(f"\nModel saved to: {model_out}")


def run_predict(subject, runs, option_argv):
    from predict import run_playback_prediction

    parser = argparse.ArgumentParser(description="Run playback predictions on EEG data.")
    add_common_args(parser)
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument(
        "--max-latency",
        type=float,
        default=2.0,
        help="Maximum target latency per epoch in seconds",
    )
    args = parser.parse_args(option_argv)

    run_playback_prediction(
        subject=subject,
        runs=runs,
        base_path=args.path,
        model_path=args.model,
        dim_red=args.dim_red,
        n_components=args.n_components,
        max_latency=args.max_latency,
        verbose=True,
    )


def save_and_print_benchmark(rows, subjects, runs, output_path):
    from benchmark import (
        aggregate_by_subject,
        aggregate_rows,
        default_output_path,
        print_best_overall,
        print_subject_summary,
        print_summary,
        write_rows_to_csv,
    )

    if not rows:
        print("No benchmark results produced.")
        return

    resolved_output = output_path or default_output_path(subjects, runs)
    write_rows_to_csv(resolved_output, rows)

    summary_rows = aggregate_rows(rows)
    subject_rows = aggregate_by_subject(rows)

    print_summary(summary_rows)
    print_subject_summary(subject_rows)
    print_best_overall(summary_rows)
    print(f"\nSaved benchmark results to: {resolved_output}")


def run_benchmark_from_args(subjects, runs, option_argv):
    from benchmark import parse_variant_specs, run_benchmark

    parser = argparse.ArgumentParser(description="Benchmark EEG BCI pipeline variants.")
    parser.add_argument("--path", type=str, default=None, help="Dataset base path")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["none", "pca:5", "csp:4"],
        help="Variants to benchmark, e.g. none pca:5 csp:4",
    )
    parser.add_argument("--cvs", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--quiet", action="store_true", help="Suppress inner training logs")
    args = parser.parse_args(option_argv)

    rows = run_benchmark(
        subjects=subjects,
        runs=runs,
        base_path=args.path,
        variants=parse_variant_specs(args.variants),
        cvs=args.cvs,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        quiet=args.quiet,
    )

    save_and_print_benchmark(rows, subjects, runs, args.output)


def run_global_benchmark(option_argv):
    from benchmark import parse_subjects, parse_variant_specs, run_benchmark
    from pipeline_config import parse_runs

    parser = argparse.ArgumentParser(description="Benchmark EEG BCI pipeline variants.")
    parser.add_argument("--subjects", type=str, nargs="+", required=True, help="Subjects to benchmark")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="Runs to benchmark")
    parser.add_argument("--path", type=str, default=None, help="Dataset base path")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["none", "pca:5", "csp:4"],
        help="Variants to benchmark, e.g. none pca:5 csp:4",
    )
    parser.add_argument("--cvs", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--quiet", action="store_true", help="Suppress inner training logs")
    args = parser.parse_args(option_argv)

    subjects = parse_subjects(args.subjects)
    runs = parse_runs(args.runs)

    rows = run_benchmark(
        subjects=subjects,
        runs=runs,
        base_path=args.path,
        variants=parse_variant_specs(args.variants),
        cvs=args.cvs,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        quiet=args.quiet,
    )

    save_and_print_benchmark(rows, subjects, runs, args.output)


def main():
    argv = sys.argv[1:]
    if not argv:
        print_usage()
        return

    if argv[0] == "benchmark":
        run_global_benchmark(argv[1:])
        return

    parsed = parse_subject_style(argv)
    if parsed is None:
        print_usage()
        raise SystemExit(2)

    subject, runs, command, option_argv = parsed

    if command == "train":
        run_train(subject, runs, option_argv)
        return

    if command == "predict":
        run_predict(subject, runs, option_argv)
        return

    run_benchmark_from_args([subject], runs, option_argv)


if __name__ == "__main__":
    main()
