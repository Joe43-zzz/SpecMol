import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ABLATION = REPO_ROOT / "scripts" / "run_bace_ablation.py"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bace")
    parser.add_argument("--data_root", type=str, default=str(REPO_ROOT))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logreg_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--heads", type=str, default="1,3")
    parser.add_argument("--clamps", type=str, default="-3:3,-5:5")
    parser.add_argument("--dropouts", type=str, default="0.0,0.1")
    parser.add_argument("--layers", type=str, default="1")
    parser.add_argument("--seeds", type=str, default="9,19,29")
    parser.add_argument("--timeout_sec", type=int, default=1800)
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    return parser.parse_args()


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_clamps(text):
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split(":")
        pairs.append((float(lo), float(hi)))
    return pairs


def read_single_run_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows[0] if rows else None


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["pair_update_heads"], row["pair_update_layers"], row["pair_update_clamp_min"], row["pair_update_clamp_max"], row["dropout"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, group in grouped.items():
        success_group = [row for row in group if row["status"] == "ok"]
        valid_values = [float(row["best_valid_auc"]) for row in success_group]
        test_values = [float(row["best_test_auc"]) for row in success_group]
        if valid_values:
            valid_mean = sum(valid_values) / len(valid_values)
            test_mean = sum(test_values) / len(test_values)
            valid_std = (sum((v - valid_mean) ** 2 for v in valid_values) / len(valid_values)) ** 0.5
            test_std = (sum((v - test_mean) ** 2 for v in test_values) / len(test_values)) ** 0.5
        else:
            valid_mean = valid_std = test_mean = test_std = ""
        summary_rows.append(
            {
                "pair_update_heads": key[0],
                "pair_update_layers": key[1],
                "pair_update_clamp_min": key[2],
                "pair_update_clamp_max": key[3],
                "dropout": key[4],
                "runs_total": len(group),
                "runs_ok": len(success_group),
                "runs_failed": len(group) - len(success_group),
                "valid_auc_mean": valid_mean,
                "valid_auc_std": valid_std,
                "test_auc_mean": test_mean,
                "test_auc_std": test_std,
            }
        )
    return summary_rows


def main():
    args = parse_args()
    heads_list = parse_int_list(args.heads)
    layers_list = parse_int_list(args.layers)
    clamp_pairs = parse_clamps(args.clamps)
    dropout_list = parse_float_list(args.dropouts)
    seeds = parse_int_list(args.seeds)
    results_dir = Path(args.results_dir).expanduser().resolve()

    all_run_rows = []
    for heads, layers, (clamp_min, clamp_max), dropout, seed in itertools.product(
        heads_list, layers_list, clamp_pairs, dropout_list, seeds
    ):
        run_name = f"h{heads}_l{layers}_c{clamp_min}_{clamp_max}_d{dropout}_s{seed}".replace("-", "m").replace(".", "p")
        run_dir = results_dir / "bondpair_stability" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "run.log"
        cmd = [
            sys.executable,
            str(RUN_ABLATION),
            "--task",
            args.task,
            "--data_root",
            str(Path(args.data_root).expanduser().resolve()),
            "--epochs",
            str(args.epochs),
            "--logreg_epochs",
            str(args.logreg_epochs),
            "--batch_size",
            str(args.batch_size),
            "--gpu",
            str(args.gpu),
            "--dropout",
            str(dropout),
            "--pair_update_layers",
            str(layers),
            "--pair_update_alpha_override",
            "0.0",
            "--pair_update_heads_override",
            str(heads),
            "--pair_update_clamp_min",
            str(clamp_min),
            "--pair_update_clamp_max",
            str(clamp_max),
            "--variants",
            "V1-bondpair",
            "--seeds",
            str(seed),
            "--results_dir",
            str(run_dir),
        ]
        print(f"[stability] launching {run_name}")
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_sec,
            )

        run_csv = run_dir / "bace_ablation_runs.csv"
        row = {
            "variant": "V1-bondpair",
            "seed": seed,
            "pair_update_heads": heads,
            "pair_update_layers": layers,
            "pair_update_clamp_min": clamp_min,
            "pair_update_clamp_max": clamp_max,
            "dropout": dropout,
            "exit_code": completed.returncode,
            "status": "ok" if completed.returncode == 0 and run_csv.exists() else "failed",
            "log_path": str(log_path),
        }
        if row["status"] == "ok":
            run_metrics = read_single_run_csv(run_csv)
            row["best_valid_auc"] = run_metrics["best_valid_auc"]
            row["best_test_auc"] = run_metrics["best_test_auc"]
            row["best_epoch"] = run_metrics["best_epoch"]
        else:
            row["best_valid_auc"] = ""
            row["best_test_auc"] = ""
            row["best_epoch"] = ""
        all_run_rows.append(row)
        print(
            f"[stability] finished {run_name} status={row['status']} "
            f"exit_code={row['exit_code']} valid={row['best_valid_auc']} test={row['best_test_auc']}"
        )

    runs_path = results_dir / "bondpair_stability_runs.csv"
    summary_path = results_dir / "bondpair_stability_summary.csv"
    write_csv(
        runs_path,
        [
            "variant",
            "seed",
            "pair_update_heads",
            "pair_update_layers",
            "pair_update_clamp_min",
            "pair_update_clamp_max",
            "dropout",
            "exit_code",
            "status",
            "best_valid_auc",
            "best_test_auc",
            "best_epoch",
            "log_path",
        ],
        all_run_rows,
    )
    summary_rows = summarize_rows(all_run_rows)
    write_csv(
        summary_path,
        [
            "pair_update_heads",
            "pair_update_layers",
            "pair_update_clamp_min",
            "pair_update_clamp_max",
            "dropout",
            "runs_total",
            "runs_ok",
            "runs_failed",
            "valid_auc_mean",
            "valid_auc_std",
            "test_auc_mean",
            "test_auc_std",
        ],
        summary_rows,
    )
    print(f"[stability] wrote_runs_csv={runs_path}")
    print(f"[stability] wrote_summary_csv={summary_path}")


if __name__ == "__main__":
    main()
