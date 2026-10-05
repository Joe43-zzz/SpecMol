#!/usr/bin/env python
"""Collect BBBP results from V0/V2-T5/T6 training logs into a unified JSON.

Parses stdout log files and outputs a comparison table.

Usage:
    python hpc/collect_bbbp_results.py --log-dir hpc/logs/bbbp
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


BEST_AUC_RE = re.compile(
    r"Best Test Auc for (?P<task>\w+) in Epoch (?P<epoch>\d+) is (?P<value>[0-9.eE+-]+)"
)
LOSS_RE = re.compile(
    r"In Epoch (\d+)th, the train loss is ([\d.]+)"
)
EARLY_STOP_RE = re.compile(r"Early stopping!")
LOADING_RE = re.compile(r"Loading (\d+)th eppoch")

VARIANT_PATTERNS = {
    # Matches both interactive (v0_seed9_stdout.log) and sbatch (v0_seed9_12345.out)
    "v0":   re.compile(r"v0_seed(\d+)(?:_stdout\.log|_\d+\.out)$"),
    "v2t5": re.compile(r"v2(?:t5)?_seed(\d+)(?:_stdout\.log|_\d+\.out)$"),
    "t6":   re.compile(r"t6_seed(\d+)(?:_stdout\.log|_\d+\.out)$"),
}


def parse_log(path, eval_seeds):
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None

    result = {}

    # Parse loss curve
    losses = [(int(m.group(1)), float(m.group(2))) for m in LOSS_RE.finditer(text)]
    if losses:
        best_epoch, best_loss = min(losses, key=lambda x: x[1])
        result["pretrain_best_epoch"] = best_epoch
        result["pretrain_final_loss"] = round(best_loss, 4)
        result["early_stopped"] = bool(EARLY_STOP_RE.search(text))

    loading = LOADING_RE.findall(text)
    if loading:
        result["pretrain_best_epoch"] = int(loading[0])

    # Parse downstream AUCs
    best_matches = [
        (int(m.group("epoch")), float(m.group("value")))
        for m in BEST_AUC_RE.finditer(text)
    ]

    splits = {}
    for i, (epoch, auc) in enumerate(best_matches):
        seed = eval_seeds[i] if i < len(eval_seeds) else i
        splits[seed] = {"test_auc": auc, "best_eval_epoch": epoch}

    if splits:
        result["eval_splits"] = {f"split_{k}": v for k, v in sorted(splits.items())}
        auc_vals = [v["test_auc"] for v in splits.values()]
        result["mean_test_auc"] = round(np.mean(auc_vals), 4)
        result["std_test_auc"] = round(np.std(auc_vals), 4)

    return result if result else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="hpc/logs/bbbp")
    parser.add_argument("--eval-seeds", default="9,19,29")
    parser.add_argument("--output", default="hpc/results/bbbp_all_results.json")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for variant, pattern in VARIANT_PATTERNS.items():
        variant_results = {}
        all_aucs = []

        for log_path in sorted(log_dir.glob("*.log")):
            match = pattern.search(log_path.name)
            if not match:
                continue
            pretrain_seed = int(match.group(1))
            result = parse_log(log_path, eval_seeds)
            if result is None:
                print(f"  WARNING: {log_path.name} no parseable content")
                continue
            variant_results[f"seed_{pretrain_seed}"] = result
            if "eval_splits" in result:
                for v in result["eval_splits"].values():
                    all_aucs.append(v["test_auc"])

        if variant_results:
            summary = {}
            if all_aucs:
                arr = np.array(all_aucs)
                summary["grand_mean"] = round(float(arr.mean()), 4)
                summary["grand_std"] = round(float(arr.std()), 4)
                summary["n_aucs"] = len(all_aucs)
                per_seed_means = [
                    r["mean_test_auc"] for r in variant_results.values()
                    if "mean_test_auc" in r
                ]
                summary["per_seed_means"] = per_seed_means

            all_results[variant] = {
                "results_per_seed": variant_results,
                "summary": summary,
            }
            print(f"{variant}: grand_mean={summary.get('grand_mean', 'N/A')} "
                  f"std={summary.get('grand_std', 'N/A')}")

    # Comparison table
    print("\n" + "=" * 60)
    print(f"{'Variant':<10} {'Mean AUC':>10} {'Std':>8} {'N':>4}")
    print("-" * 60)
    for variant in ["v0", "v2t5", "t6"]:
        if variant in all_results:
            s = all_results[variant]["summary"]
            print(f"{variant:<10} {s.get('grand_mean', 'N/A'):>10} "
                  f"{s.get('grand_std', 'N/A'):>8} {s.get('n_aucs', 0):>4}")
        else:
            print(f"{variant:<10} {'(no data)':>10}")
    print("=" * 60)

    # Save
    out_data = {
        "experiment": "BBBP V0/V2-T5/T6 Comparison (Uni-Mol fold split)",
        "task": "bbbp",
        "eval_seeds": eval_seeds,
        "variants": all_results,
    }
    output_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
