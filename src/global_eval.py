import contextlib
import io
import os

from train import train_and_evaluate


EXPERIMENT_RUNS = [4, 6, 8, 10, 12, 14]


def get_available_subjects(base_path):
    if base_path is None:
        base_path = os.getenv("EEG_DATA_PATH", "data/files")

    if not os.path.isdir(base_path):
        return []

    subjects = []
    for entry in sorted(os.listdir(base_path)):
        if not entry.startswith("S"):
            continue
        if not os.path.isdir(os.path.join(base_path, entry)):
            continue
        try:
            subjects.append(int(entry[1:]))
        except ValueError:
            continue

    return subjects


def run_full_evaluation(base_path=None, test_size=0.2, val_size=0.2, cvs=5, seed=42, dim_red="csp", n_components=5):
    subjects = get_available_subjects(base_path)
    if not subjects:
        print("No subjects found in data/files.")
        return

    experiment_means = []

    for experiment_index, run in enumerate(EXPERIMENT_RUNS):
        subject_scores = []
        for subject in subjects:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = train_and_evaluate(
                        subject=subject,
                        runs=[run],
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
                print(f"experiment {experiment_index}: subject {subject:03d}: skipped ({exc})")
                continue

            accuracy = result.test_accuracy
            subject_scores.append(accuracy)
            print(f"experiment {experiment_index}: subject {subject:03d}: accuracy = {accuracy:.4f}")

        mean_accuracy = sum(subject_scores) / len(subject_scores) if subject_scores else 0.0
        experiment_means.append(mean_accuracy)

    print(
        f"\nMean accuracy of the six different experiments for all {len(subjects)} "
        f"available subjects with {dim_red}:{n_components}:"
    )
    for experiment_index, mean_accuracy in enumerate(experiment_means):
        print(f"experiment {experiment_index}: accuracy = {mean_accuracy:.4f}")

    global_mean = sum(experiment_means) / len(experiment_means)
    print(f"Mean accuracy of 6 experiments: {global_mean:.4f}")
