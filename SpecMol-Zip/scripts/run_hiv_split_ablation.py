import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_bace_ablation import summarize_rows, write_csv  # noqa: E402


DEFAULT_VARIANTS = ["V0", "V0-pairlearn", "V0-pairlearn-v2", "V0-ctrl"]
DEFAULT_SEEDS = [9, 19, 29]
VARIANT_SLUGS = {
    "V0": "v0",
    "V0-pairlearn": "v0pair",
    "V0-pairlearn-v2": "v0pairv2",
    "V0-ctrl": "v0ctrl",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(REPO_ROOT / "molecular_benchmarks" / "hiv"))
    parser.add_argument("--results-root", type=str, default=str(REPO_ROOT / "results" / "molecular_benchmarks"))
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logreg-epochs", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--job-timeout-sec", type=int, default=0)
    parser.add_argument("--debug-pair-path", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def parse_csv_list(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def result_dir_for(results_root, variant, seed):
    slug = VARIANT_SLUGS[variant]
    return results_root / f"hiv_{slug}_s{seed}"


def result_csv_for(job_dir):
    return job_dir / "hiv_ablation_runs.csv"


def log_path_for(results_root, variant, seed):
    slug = VARIANT_SLUGS[variant]
    return results_root / f"hiv_{slug}_s{seed}.log"


def job_is_complete(job_dir):
    csv_path = result_csv_for(job_dir)
    return csv_path.exists() and csv_path.stat().st_size > 0


def load_run_row(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly 1 row in {csv_path}, found {len(rows)}")
    row = rows[0]
    int_fields = [
        "seed",
        "use_pair_update",
        "pair_update_layers",
        "pair_update_heads",
        "pair_edge_attr_dim",
        "pretrain_epochs",
        "logreg_epochs",
        "best_epoch",
    ]
    float_fields = [
        "pair_update_alpha",
        "pair_update_clamp_min",
        "pair_update_clamp_max",
        "best_pretrain_loss",
        "best_valid_auc",
        "best_test_auc",
    ]
    for field in int_fields:
        row[field] = int(row[field])
    for field in float_fields:
        row[field] = float(row[field])
    return row


def run_job(args, variant, seed, results_root):
    job_dir = result_dir_for(results_root, variant, seed)
    log_path = log_path_for(results_root, variant, seed)
    if job_is_complete(job_dir):
        print(f"[split] skip variant={variant} seed={seed} reason=existing_result")
        return {"variant": variant, "seed": seed, "status": "skipped", "job_dir": str(job_dir), "log_path": str(log_path)}

    job_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "run_bace_ablation.py"),
        "--task",
        "hiv",
        "--data_root",
        str(Path(args.data_root).expanduser().resolve()),
        "--variants",
        variant,
        "--seeds",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--logreg_epochs",
        str(args.logreg_epochs),
        "--batch_size",
        str(args.batch_size),
        "--gpu",
        str(args.gpu),
        "--results_dir",
        str(job_dir),
    ]
    if args.debug_pair_path and variant in {"V0-pairlearn", "V0-pairlearn-v2"}:
        command.append("--debug_pair_path")

    print(f"[split] start variant={variant} seed={seed} log={log_path}")
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            timeout=args.job_timeout_sec if args.job_timeout_sec > 0 else None,
            check=False,
        )

    if completed.returncode != 0:
        print(f"[split] fail variant={variant} seed={seed} returncode={completed.returncode}")
        return {
            "variant": variant,
            "seed": seed,
            "status": "failed",
            "returncode": completed.returncode,
            "job_dir": str(job_dir),
            "log_path": str(log_path),
        }

    if not job_is_complete(job_dir):
        print(f"[split] fail variant={variant} seed={seed} reason=missing_result_csv")
        return {
            "variant": variant,
            "seed": seed,
            "status": "failed",
            "returncode": 0,
            "job_dir": str(job_dir),
            "log_path": str(log_path),
            "reason": "missing_result_csv",
        }

    print(f"[split] done variant={variant} seed={seed} result={result_csv_for(job_dir)}")
    return {"variant": variant, "seed": seed, "status": "completed", "job_dir": str(job_dir), "log_path": str(log_path)}


def aggregate_results(results_root, variants, seeds):
    run_rows = []
    completed_jobs = []
    missing_jobs = []
    for variant in variants:
        for seed in seeds:
            job_dir = result_dir_for(results_root, variant, seed)
            csv_path = result_csv_for(job_dir)
            if csv_path.exists():
                run_rows.append(load_run_row(csv_path))
                completed_jobs.append((variant, seed))
            else:
                missing_jobs.append((variant, seed))

    aggregate_dir = results_root / "hiv"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    runs_path = aggregate_dir / "hiv_ablation_runs.csv"
    summary_path = aggregate_dir / "hiv_ablation_summary.csv"
    status_path = aggregate_dir / "hiv_job_status.csv"

    if run_rows:
        write_csv(
            runs_path,
            [
                "variant",
                "seed",
                "data_root",
                "use_pair_update",
                "pair_update_layers",
                "pair_update_heads",
                "pair_update_alpha",
                "pair_update_mode",
                "pair_update_clamp_min",
                "pair_update_clamp_max",
                "pair_edge_attr_dim",
                "pair_input_attr_name",
                "pretrain_epochs",
                "logreg_epochs",
                "best_pretrain_loss",
                "best_valid_auc",
                "best_test_auc",
                "best_epoch",
            ],
            run_rows,
        )
        summary_rows = summarize_rows(run_rows)
        write_csv(
            summary_path,
            [
                "variant",
                "seeds",
                "runs",
                "valid_auc_mean",
                "valid_auc_std",
                "test_auc_mean",
                "test_auc_std",
            ],
            summary_rows,
        )
    else:
        summary_rows = []

    status_rows = []
    for variant in variants:
        for seed in seeds:
            job_dir = result_dir_for(results_root, variant, seed)
            status_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "status": "completed" if (variant, seed) in completed_jobs else "missing",
                    "result_csv": str(result_csv_for(job_dir)),
                    "log_path": str(log_path_for(results_root, variant, seed)),
                }
            )
    write_csv(status_path, ["variant", "seed", "status", "result_csv", "log_path"], status_rows)
    return runs_path, summary_path, status_path, summary_rows, missing_jobs


def main():
    args = parse_args()
    variants = parse_csv_list(args.variants)
    seeds = parse_int_list(args.seeds)
    results_root = Path(args.results_root).expanduser().resolve()

    failures = []
    completed_or_skipped = []
    for variant in variants:
        for seed in seeds:
            result = run_job(args, variant, seed, results_root)
            completed_or_skipped.append(result)
            if result["status"] == "failed":
                failures.append(result)
                if not args.continue_on_error:
                    break
        if failures and not args.continue_on_error:
            break

    runs_path, summary_path, status_path, summary_rows, missing_jobs = aggregate_results(results_root, variants, seeds)
    print(f"[split] aggregate_runs={runs_path}")
    print(f"[split] aggregate_summary={summary_path}")
    print(f"[split] aggregate_status={status_path}")

    for row in summary_rows:
        print(
            f"[split-summary] variant={row['variant']} runs={row['runs']} "
            f"test_auc={row['test_auc_mean']:.6f}±{row['test_auc_std']:.6f}"
        )

    if missing_jobs:
        print(f"[split] missing_jobs={missing_jobs}")
    if failures:
        print(f"[split] failed_jobs={failures}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
