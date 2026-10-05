import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--label-csv", type=str, default=None)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    return parser.parse_args()


def run_command(stage_label, cmd):
    print(stage_label)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    converted_dir = output_root / "converted_csv"

    convert_cmd = [
        sys.executable,
        str(REPO_ROOT / "convert_spectra_to_csv.py"),
        "--input-path",
        str(Path(args.input_path).expanduser().resolve()),
        "--output-dir",
        str(converted_dir),
    ]
    if args.label_csv:
        convert_cmd.extend(["--label-csv", str(Path(args.label_csv).expanduser().resolve())])
    else:
        convert_cmd.extend(["--default-label", "0"])

    dataset_cmd = [
        sys.executable,
        str(REPO_ROOT / "make_spectrum_pair_dataset.py"),
        "--peaks-csv",
        str(converted_dir / "peaks.csv"),
        "--labels-csv",
        str(converted_dir / "labels.csv"),
        "--output-root",
        str(output_root),
        "--task",
        args.task,
    ]

    results_dir = output_root / "results"
    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_bace_ablation.py"),
        "--task",
        args.task,
        "--data_root",
        str(output_root),
        "--variants",
        "V0-pairlearn",
        "--seeds",
        "9",
        "--epochs",
        "1",
        "--logreg_epochs",
        "3",
        "--batch_size",
        "2",
        "--gpu",
        "-1",
        "--use_pair_update",
        "1",
        "--pair_input_attr_name",
        "pair_attr",
        "--pair_edge_attr_dim",
        "2",
        "--debug_pair_path",
        "--results_dir",
        str(results_dir),
    ]

    run_command("[stage 1] converting spectra", convert_cmd)
    run_command("[stage 2] building dataset", dataset_cmd)
    run_command("[stage 3] training", train_cmd)

    print(f"[done] converted_csv={converted_dir}")
    print(f"[done] processed_dir={output_root / 'processed'}")
    print(f"[done] results_dir={results_dir}")


if __name__ == "__main__":
    main()
