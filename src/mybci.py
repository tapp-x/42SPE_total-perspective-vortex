import argparse
import sys

def print_usage():
    print("Usage:")
    print("  python src/mybci.py <subject> <run> [runs ...] train [options]")
    print("  python src/mybci.py <subject> <run> [runs ...] predict [options]")
    print("  python src/mybci.py <subject> <run> [runs ...] benchmark [options]")
    print("  python src/mybci.py benchmark --subjects 1 2 3 --runs 4 8 12 [options]")


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
    if not argv:
        print_usage()
        return

    if argv[0] == "benchmark":
        from benchmark import main as benchmark_main

        sys.argv = ["benchmark.py", *argv[1:]]
        benchmark_main()
        return

    dispatch_subject_style(argv)


if __name__ == "__main__":
    main()
