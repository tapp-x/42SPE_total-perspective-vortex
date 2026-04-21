import argparse
from pathlib import Path
from urllib.request import urlopen

from pipeline_config import parse_runs, parse_subjects

EEGBCI_BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as output_file:
        output_file.write(response.read())


def import_eegbci_data(subjects, runs, dataset_root, force_update=False):
    imported_paths = []

    for subject in subjects:
        subject_folder = f"S{subject:03d}"
        for run in runs:
            filename = f"{subject_folder}R{run:02d}.edf"
            destination = Path(dataset_root) / subject_folder / filename
            source_url = f"{EEGBCI_BASE_URL}/{subject_folder}/{filename}"

            if destination.exists() and not force_update:
                print(f"Skipped existing file: {destination}")
                imported_paths.append(destination)
                continue

            print(f"Downloading {source_url}")
            download_file(source_url, destination)
            imported_paths.append(destination)

    return imported_paths


def main():
    parser = argparse.ArgumentParser(description="Import EEGBCI data into the project dataset layout.")
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        required=True,
        help="Subjects to import (e.g. 1 2 3) or all",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="Runs to import (e.g. 4 8 12) or all",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/files",
        help="Destination root matching the project dataset layout",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Redownload files and refresh local links if they already exist",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    runs = parse_runs(args.runs)
    dataset_root = Path(args.path)
    dataset_root.mkdir(parents=True, exist_ok=True)

    imported_paths = import_eegbci_data(
        subjects=subjects,
        runs=runs,
        dataset_root=dataset_root,
        force_update=args.force_update,
    )

    print("\n--- IMPORT COMPLETE ---")
    print(f"Subjects: {subjects}")
    print(f"Runs: {runs}")
    print(f"Imported files: {len(imported_paths)}")
    if imported_paths:
        print(f"Dataset root: {dataset_root}")
        print(f"First file: {imported_paths[0]}")


if __name__ == "__main__":
    main()
