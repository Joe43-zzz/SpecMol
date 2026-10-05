import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(REPO_ROOT / "down_task"))
    parser.add_argument("--task", type=str, default="bace")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logreg-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--results-dir", type=str, default=str(REPO_ROOT / "results" / "mock_spectrum_pairlearn"))
    parser.add_argument("--seeds", type=str, default="9")
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_bace_ablation.py"),
        "--task",
        args.task,
        "--data_root",
        str(Path(args.data_root).expanduser().resolve()),
        "--variants",
        "V0-pairlearn",
        "--seeds",
        args.seeds,
        "--epochs",
        str(args.epochs),
        "--logreg_epochs",
        str(args.logreg_epochs),
        "--batch_size",
        str(args.batch_size),
        "--gpu",
        str(args.gpu),
        "--use_pair_update",
        "1",
        "--pair_input_attr_name",
        "pair_attr",
        "--pair_edge_attr_dim",
        "2",
        "--debug_pair_path",
        "--results_dir",
        str(Path(args.results_dir).expanduser().resolve()),
    ]
    print("[mock-spectrum] launching command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
