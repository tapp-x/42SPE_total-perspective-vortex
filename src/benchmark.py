import argparse
import contextlib
import csv
import io
import os

from pipeline_config import parse_runs
from train import train_and_evaluate


def parse_subjects(subject_args):
    normalized_subjects = []
    for value in subject_args:
        normalized_subjects.extend(value.split())

    if "all" in [value.lower() for value in normalized_subjects]:
        return list(range(1, 110))
    return [int(value) for value in normalized_subjects]


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


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["dim_red"], row["n_components"])
        grouped.setdefault(
            key,
            {
                "dim_red": row["dim_red"],
                "n_components": row["n_components"],
                "runs": row["runs"],
                "subjects": [],
                "cv_mean_values": [],
                "val_accuracy_values": [],
                "test_accuracy_values": [],
            },
        )
        grouped[key]["subjects"].append(row["subject"])
        grouped[key]["cv_mean_values"].append(float(row["cv_mean"]))
        grouped[key]["val_accuracy_values"].append(float(row["val_accuracy"]))
        grouped[key]["test_accuracy_values"].append(float(row["test_accuracy"]))

    summary_rows = []
    for grouped_row in grouped.values():
        subject_count = len(set(grouped_row["subjects"]))
        cv_mean_values = grouped_row["cv_mean_values"]
        val_accuracy_values = grouped_row["val_accuracy_values"]
        test_accuracy_values = grouped_row["test_accuracy_values"]
        summary_rows.append(
            {
                "dim_red": grouped_row["dim_red"],
                "n_components": grouped_row["n_components"],
                "runs": grouped_row["runs"],
                "subject_count": subject_count,
                "cv_mean": f"{sum(cv_mean_values) / len(cv_mean_values):.4f}",
                "val_accuracy": f"{sum(val_accuracy_values) / len(val_accuracy_values):.4f}",
                "test_accuracy": f"{sum(test_accuracy_values) / len(test_accuracy_values):.4f}",
            }
        )

    summary_rows.sort(key=lambda row: row["test_accuracy"], reverse=True)
    return summary_rows


def aggregate_by_subject(rows):
    grouped = {}
    for row in rows:
        key = row["subject"]
        grouped.setdefault(key, []).append(row)

    subject_rows = []
    for subject, subject_entries in grouped.items():
        best_entry = max(subject_entries, key=lambda row: float(row["test_accuracy"]))
        best_dim_red = best_entry["dim_red"]
        best_n_components = best_entry["n_components"]
        subject_rows.append(
            {
                "subject": subject,
                "runs": best_entry["runs"],
                "best_variant": f"{best_dim_red}:{best_n_components}",
                "cv_mean": best_entry["cv_mean"],
                "val_accuracy": best_entry["val_accuracy"],
                "test_accuracy": best_entry["test_accuracy"],
            }
        )

    subject_rows.sort(key=lambda row: row["subject"])
    return subject_rows


def print_summary(summary_rows):
    print("\n--- SUMMARY BY VARIANT ---")
    for row in summary_rows:
        dim_red = row["dim_red"]
        n_components = row["n_components"]
        subject_count = row["subject_count"]
        cv_mean = row["cv_mean"]
        val_accuracy = row["val_accuracy"]
        test_accuracy = row["test_accuracy"]
        print(
            f"{dim_red}:{n_components} "
            f"subjects={subject_count} "
            f"cv_mean={cv_mean} "
            f"val={val_accuracy} "
            f"test={test_accuracy}"
        )


def print_subject_summary(subject_rows):
    print("\n--- BEST VARIANT BY SUBJECT ---")
    for row in subject_rows:
        subject = row["subject"]
        best_variant = row["best_variant"]
        cv_mean = row["cv_mean"]
        val_accuracy = row["val_accuracy"]
        test_accuracy = row["test_accuracy"]
        print(
            f"subject={subject} "
            f"best={best_variant} "
            f"cv_mean={cv_mean} "
            f"val={val_accuracy} "
            f"test={test_accuracy}"
        )


def print_best_overall(summary_rows):
    best_row = summary_rows[0]
    dim_red = best_row["dim_red"]
    n_components = best_row["n_components"]
    subject_count = best_row["subject_count"]
    cv_mean = best_row["cv_mean"]
    val_accuracy = best_row["val_accuracy"]
    test_accuracy = best_row["test_accuracy"]
    print("\n--- BEST OVERALL ---")
    print(
        f"{dim_red}:{n_components} "
        f"subjects={subject_count} "
        f"cv_mean={cv_mean} "
        f"val={val_accuracy} "
        f"test={test_accuracy}"
    )


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
    resolved_variants = variants or parse_variant_specs(["csp:5"])
    rows = []

    print("\n--- BENCHMARK ---")
    for subject in subjects:
        for dim_red, n_components in resolved_variants:
            print(
                f"Running subject={subject:03d} runs={runs} dim_red={dim_red} n_components={n_components}"
            )
            try:
                if quiet:
                    with contextlib.redirect_stdout(io.StringIO()):
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
            cv_mean = row["cv_mean"]
            val_accuracy = row["val_accuracy"]
            test_accuracy = row["test_accuracy"]
            print(f"cv_mean={cv_mean} val={val_accuracy} test={test_accuracy}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark EEG BCI pipeline variants.")
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        required=True,
        help="Subjects to benchmark (e.g. 1 2 3) or all",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="Runs to use (e.g. 4 8 12) or all",
    )
    parser.add_argument("--path", type=str, default=None, help="Dataset base path")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["csp:5"],
        help="Variants to benchmark, e.g. csp:5 none pca:8",
    )
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
    summary_rows = aggregate_rows(rows)
    subject_rows = aggregate_by_subject(rows)
    print_summary(summary_rows)
    print_subject_summary(subject_rows)
    print_best_overall(summary_rows)
    print(f"\nSaved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
