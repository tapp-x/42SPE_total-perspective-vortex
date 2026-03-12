import argparse
import csv
import os

from pipeline_config import parse_runs
from train import train_and_evaluate


def parse_subjects(subject_args):
    if "all" in [value.lower() for value in subject_args]:
        return list(range(1, 110))
    return [int(value) for value in subject_args]


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
    return f"resuts/benchmark_subjects_{subject_slug}_runs_{runs_slug}.csv"


def write_rows_to_csv(output_path, rows):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark EEG BCI pipeline variants.")
    parser.add_argument(
        "subjects",
        type=str,
        nargs="+",
        help="Subjects to benchmark (e.g. 1 2 3) or 'all'",
    )
    parser.add_argument(
        "runs",
        type=str,
        nargs="+",
        help="Runs to use (e.g. 4 8 12) or 'all'",
    )
    parser.add_argument("--path", type=str, default=None, help="Dataset base path")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["none", "pca:5", "csp:4"],
        help="Variants to benchmark, e.g. none pca:10 csp:4",
    )
    parser.add_argument("--cvs", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    runs = parse_runs(args.runs)
    variants = parse_variant_specs(args.variants)

    rows = []
    print("\n--- BENCHMARK ---")
    for subject in subjects:
        for dim_red, n_components in variants:
            print(
                f"Running subject={subject:03d} runs={runs} dim_red={dim_red} n_components={n_components}"
            )
            try:
                result = train_and_evaluate(
                    subject=subject,
                    runs=runs,
                    base_path=args.path,
                    test_size=args.test_size,
                    val_size=args.val_size,
                    cvs=args.cvs,
                    seed=args.seed,
                    dim_red=dim_red,
                    n_components=n_components,
                    verbose=False,
                )
            except Exception as exc:
                print(f"Skipped subject={subject:03d} dim_red={dim_red}: {exc}")
                continue

            row = {
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
            rows.append(row)
            print(
                f"cv_mean={row['cv_mean']} val={row['val_accuracy']} test={row['test_accuracy']}"
            )

    if not rows:
        print("No benchmark results produced.")
        return

    output_path = args.output or default_output_path(subjects, runs)
    write_rows_to_csv(output_path, rows)
    print(f"\nSaved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
