import argparse
import csv
import os
from contextlib import redirect_stdout
from io import StringIO

from pipeline_config import parse_runs, parse_subjects
from train import train_and_evaluate


def parse_variant_specs(specs):
    variants = []
    for spec in specs:
        if ":" in spec:
            dim_red, component_value = spec.split(":", 1)
            n_components = int(component_value)
        else:
            dim_red = spec
            n_components = 10
        variants.append((dim_red, n_components))
    return variants


def default_output_path(subjects, runs):
    subject_slug = "all" if len(subjects) > 3 else "-".join(f"{subject:03d}" for subject in subjects)
    runs_slug = "all" if runs == list(range(1, 15)) else "-".join(f"{run:02d}" for run in runs)
    return f"results/benchmark_subjects_{subject_slug}_runs_{runs_slug}.csv"


def write_rows_to_csv(output_path, rows):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "subject",
                "runs",
                "dim_red",
                "n_components",
                "n_epochs",
                "classes",
                "cv_mean",
                "val_accuracy",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    subjects,
    runs,
    base_path=None,
    variants=None,
    cvs=5,
    test_size=0.2,
    val_size=0.2,
    seed=42,
    quiet=False,
):
    rows = []
    resolved_variants = variants or parse_variant_specs(["csp:5"])

    print("\n--- Benchmark ---")
    for subject in subjects:
        for dim_red, n_components in resolved_variants:
            print(f"Running subject={subject:03d} dim_red={dim_red} n_components={n_components}")
            try:
                if quiet:
                    with redirect_stdout(StringIO()):
                        result = train_and_evaluate(
                            subject=subject,
                            runs=runs,
                            base_path=base_path,
                            test_size=test_size,
                            val_size=val_size,
                            cvs=cvs,
                            seed=seed,
                            dim_red=dim_red,
                            n_components=n_components,
                            verbose=False,
                        )
                else:
                    result = train_and_evaluate(
                        subject=subject,
                        runs=runs,
                        base_path=base_path,
                        test_size=test_size,
                        val_size=val_size,
                        cvs=cvs,
                        seed=seed,
                        dim_red=dim_red,
                        n_components=n_components,
                        verbose=False,
                    )
            except Exception as exc:
                print(f"Skipped subject={subject:03d} dim_red={dim_red}: {exc}")
                continue

            rows.append(
                {
                    "subject": f"{subject:03d}",
                    "runs": "all" if runs == list(range(1, 15)) else "-".join(f"{run:02d}" for run in runs),
                    "dim_red": dim_red,
                    "n_components": n_components,
                    "n_epochs": result.n_epochs,
                    "classes": ",".join(str(value) for value in result.classes),
                    "cv_mean": f"{result.cv_scores.mean():.4f}",
                    "val_accuracy": f"{result.val_accuracy:.4f}",
                    "test_accuracy": f"{result.test_accuracy:.4f}",
                }
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark EEG BCI pipeline variants.")
    parser.add_argument("--subjects", type=str, nargs="+", required=True, help="Subjects to benchmark (e.g. 1 2 3 or all)")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="Runs to use (e.g. 4 8 12 or all)")
    parser.add_argument("--path", type=str, default=None, help="Dataset base path")
    parser.add_argument("--variants", type=str, nargs="+", default=["csp:5"], help="Variants to benchmark, e.g. csp:5 none pca:8")
    parser.add_argument("--cvs", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--quiet", action="store_true", help="Suppress inner training logs during benchmarking")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    runs = parse_runs(args.runs)
    variants = parse_variant_specs(args.variants)

    rows = run_benchmark(
        subjects=subjects,
        runs=runs,
        base_path=args.path,
        variants=variants,
        cvs=args.cvs,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        quiet=args.quiet,
    )

    if not rows:
        print("No benchmark results produced.")
        return

    output_path = args.output or default_output_path(subjects, runs)
    write_rows_to_csv(output_path, rows)
    print(f"Saved {len(rows)} benchmark rows to: {output_path}")


if __name__ == "__main__":
    main()
