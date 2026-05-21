#!/usr/bin/env python
"""Task-agnostic results collector for V0 / V2-T5 / T6 / T6-safe training logs.

Parses stdout log files for a single task and outputs a unified JSON with
per-seed details + variant summaries. Schema is stable across tasks
(bace, bbbp, hiv, clintox, tox21).

Usage:
    python hpc/collect_results.py --task bbbp --log-dir hpc/logs/bbbp
    python hpc/collect_results.py --task bace --output hpc/results/bace_all_results.json

Output schema:
{
  "task": "bbbp",
  "experiment": "V0/V2-T5/T6 Comparison (Uni-Mol fold split)",
  "eval_seeds": [9, 19, 29],
  "metric_name": "auc",
  "variants": {
    "v0": {
      "results_per_seed": {
        "seed_9": {
          "pretrain_best_epoch": int,
          "pretrain_final_loss": float,
          "early_stopped": bool,
          "eval_splits": {"split_9": {"test_metric": float, "best_eval_epoch": int}, ...},
          "mean_test_metric": float,
          "std_test_metric": float
        }, ...
      },
      "summary": {
        "grand_mean": float, "grand_std": float, "n_metrics": int,
        "per_seed_means": [float, ...]
      }
    }, ...
  }
}
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


# Regex patterns for log parsing
BEST_AUC_RE = re.compile(
    r"Best Test Auc for (?P<task>\w+) in Epoch (?P<epoch>\d+) is (?P<value>[0-9.eE+-]+)"
)
BEST_RMSE_RE = re.compile(
    r"Best Test RMSE for (?P<task>\w+) in Epoch (?P<epoch>\d+) is (?P<value>[0-9.eE+-]+)"
)
LOSS_RE = re.compile(r"In Epoch (\d+)th, the train loss is (\d+\.\d+)")
EARLY_STOP_RE = re.compile(r"Early stopping!")
LOADING_RE = re.compile(r"Loading (\d+)th eppoch")

# Variant name patterns (matches both interactive stdout.log and sbatch *.log/*.out)
VARIANT_PATTERNS = {
    "v0":      re.compile(r"v0_seed(\d+)(?:_stdout\.log|_\d+\.log|_\d+\.out)$"),
    "v2t5":    re.compile(r"v2(?:t5)?_seed(\d+)(?:_stdout\.log|_\d+\.log|_\d+\.out)$"),
    "t6":      re.compile(r"t6_seed(\d+)(?:_stdout\.log|_\d+\.log|_\d+\.out)$"),
    "t6safe0": re.compile(r"t6safe0_seed(\d+)(?:_stdout\.log|_\d+\.log|_\d+\.out)$"),
    "t6safe":  re.compile(r"t6safe_seed(\d+)(?:_stdout\.log|_\d+\.log|_\d+\.out)$"),
    "t7":      re.compile(r"t7_seed(\d+)(?:_stdout\.log|_\d+\.log|_\d+\.out)$"),
}


def parse_log(path: Path, eval_seeds, metric_name: str = "auc"):
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None

    result = {}

    losses = [(int(m.group(1)), float(m.group(2))) for m in LOSS_RE.finditer(text)]
    if losses:
        best_epoch, best_loss = min(losses, key=lambda x: x[1])
        result["pretrain_best_epoch"] = best_epoch
        result["pretrain_final_loss"] = round(best_loss, 4)
        result["early_stopped"] = bool(EARLY_STOP_RE.search(text))

    loading = LOADING_RE.findall(text)
    if loading:
        result["pretrain_best_epoch"] = int(loading[0])

    metric_re = BEST_RMSE_RE if metric_name == "rmse" else BEST_AUC_RE
    best_matches = [
        (int(m.group("epoch")), float(m.group("value")))
        for m in metric_re.finditer(text)
    ]

    splits = {}
    for i, (epoch, value) in enumerate(best_matches):
        seed = eval_seeds[i] if i < len(eval_seeds) else i
        splits[seed] = {"test_metric": value, "best_eval_epoch": epoch}

    if splits:
        result["eval_splits"] = {f"split_{k}": v for k, v in sorted(splits.items())}
        vals = [v["test_metric"] for v in splits.values()]
        result["mean_test_metric"] = round(float(np.mean(vals)), 4)
        result["std_test_metric"] = round(float(np.std(vals)), 4)

    return result if result else None


def collect_variant(log_dir: Path, variant: str, pattern: re.Pattern, eval_seeds, metric_name: str):
    variant_results = {}
    all_vals = []

    # Order logs by mtime DESC so the freshest log per (variant, seed) wins.
    # Earlier sorted-by-name ordering caused old historical jobids to shadow
    # the latest re-submissions when both had usable content.
    paths = sorted(
        list(log_dir.glob("*.log")) + list(log_dir.glob("*.out")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for log_path in paths:
        match = pattern.search(log_path.name)
        if not match:
            continue
        pretrain_seed = int(match.group(1))
        seed_key = f"seed_{pretrain_seed}"
        if seed_key in variant_results:
            # Already kept the freshest log for this seed; skip older ones.
            continue
        result = parse_log(log_path, eval_seeds, metric_name=metric_name)
        if result is None:
            print(f"  WARNING: {log_path.name} no parseable content")
            continue
        if not result.get("eval_splits"):
            # Pretraining log without a downstream eval is not useful — keep
            # searching the next-older log for this seed instead of locking
            # the empty-eval one in.
            print(f"  WARNING: {log_path.name} parsed but has no eval_splits, trying older")
            continue
        variant_results[seed_key] = result
        for v in result["eval_splits"].values():
            all_vals.append(v["test_metric"])

    if not variant_results:
        return None

    summary = {}
    if all_vals:
        arr = np.array(all_vals)
        summary["grand_mean"] = round(float(arr.mean()), 4)
        summary["grand_std"] = round(float(arr.std()), 4)
        summary["n_metrics"] = len(all_vals)
        per_seed_means = [
            r["mean_test_metric"] for r in variant_results.values()
            if "mean_test_metric" in r
        ]
        summary["per_seed_means"] = per_seed_means

    return {"results_per_seed": variant_results, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="bace | bbbp | hiv | clintox | tox21 | freesolv")
    parser.add_argument("--log-dir", default=None, help="default: hpc/logs/<task>")
    parser.add_argument("--eval-seeds", default="9,19,29")
    parser.add_argument("--output", default=None, help="default: hpc/results/<task>_all_results.json")
    parser.add_argument("--variants", default="v0,v2t5,t6,t6safe0,t6safe,t7",
                        help="comma-separated variants to collect")
    parser.add_argument("--metric", default="auto", choices=["auto", "auc", "rmse"],
                        help="auto picks rmse for freesolv, auc otherwise")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else Path("hpc/logs") / args.task
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]
    output_path = Path(args.output) if args.output else Path("hpc/results") / f"{args.task}_all_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variants_to_collect = [v.strip() for v in args.variants.split(",") if v.strip()]

    metric_name = args.metric
    if metric_name == "auto":
        metric_name = "rmse" if args.task == "freesolv" else "auc"

    if not log_dir.exists():
        print(f"ERROR: log dir not found: {log_dir}")
        return 1

    all_results = {}
    for variant in variants_to_collect:
        pattern = VARIANT_PATTERNS.get(variant)
        if pattern is None:
            print(f"  SKIP: unknown variant '{variant}'")
            continue
        collected = collect_variant(log_dir, variant, pattern, eval_seeds, metric_name)
        if collected is not None:
            all_results[variant] = collected
            s = collected["summary"]
            print(f"{variant}: grand_mean={s.get('grand_mean', 'N/A')} std={s.get('grand_std', 'N/A')} n={s.get('n_metrics', 0)}")
        else:
            print(f"{variant}: no logs found")

    print("\n" + "=" * 60)
    print(f"Task: {args.task}  Metric: {metric_name}")
    print(f"{'Variant':<10} {'Mean':>10} {'Std':>8} {'N':>4}")
    print("-" * 60)
    for variant in variants_to_collect:
        if variant in all_results:
            s = all_results[variant]["summary"]
            print(f"{variant:<10} {s.get('grand_mean', 'N/A'):>10} "
                  f"{s.get('grand_std', 'N/A'):>8} {s.get('n_metrics', 0):>4}")
        else:
            print(f"{variant:<10} {'(no data)':>10}")
    print("=" * 60)

    out_data = {
        "task": args.task,
        "experiment": f"{args.task.upper()} V0/V2-T5/T6 Comparison (Uni-Mol fold split)",
        "eval_seeds": eval_seeds,
        "metric_name": metric_name,
        "variants": all_results,
    }
    output_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
