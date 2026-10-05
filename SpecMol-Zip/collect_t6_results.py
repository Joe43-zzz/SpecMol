"""Collect T6 BACE results from training logs.

Parses t6_seed{S}_stdout.log files for final loss, best epoch, and downstream
AUC scores. Outputs t6_bace_results.json in the same format as v2_t5_bace_results.json.

Usage:
    python collect_t6_results.py [--log-dir .] [--seeds 9,19,29]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def parse_log(path):
    """Parse a single T6 stdout log file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None

    # Parse epoch/loss lines: "In Epoch 42th, the train loss is 18.667..."
    loss_pat = re.compile(r"In Epoch (\d+)th, the train loss is ([\d.]+)")
    losses = [(int(m.group(1)), float(m.group(2))) for m in loss_pat.finditer(text)]

    # Parse AUC lines: "best_test_auc: 0.8567" or "Test AUC: 0.8567"
    auc_pat = re.compile(r"(?:best_test_auc|Test AUC)[:\s]+([\d.]+)", re.IGNORECASE)
    aucs = [float(m.group(1)) for m in auc_pat.finditer(text)]

    # Parse eval split lines: "split_seed 9 ... best_test_auc 0.8567"
    split_pat = re.compile(
        r"split_seed\s+(\d+).*?best_test_auc\s+([\d.]+)", re.IGNORECASE
    )
    splits = {int(m.group(1)): float(m.group(2)) for m in split_pat.finditer(text)}

    # Parse early stopping: "Early stop at epoch 500"
    es_pat = re.compile(r"Early stop at epoch (\d+)")
    es_match = es_pat.search(text)

    # Parse bias values if present
    bias_init_pat = re.compile(r"bias_init[:\s]+([\d.]+)")
    bias_final_pat = re.compile(r"bias_final[:\s]+([\d.]+)")
    bias_init = None
    bias_final = None
    m = bias_init_pat.search(text)
    if m:
        bias_init = float(m.group(1))
    m = bias_final_pat.search(text)
    if m:
        bias_final = float(m.group(1))

    result = {}

    if losses:
        best_epoch, best_loss = min(losses, key=lambda x: x[1])
        result["pretrain_best_epoch"] = best_epoch
        result["pretrain_final_loss"] = round(best_loss, 4)
        result["early_stopped"] = es_match is not None
        # Sample loss history (first 3 + last 3)
        loss_vals = [l for _, l in losses]
        if len(loss_vals) > 6:
            sample = loss_vals[:3] + loss_vals[-3:]
        else:
            sample = loss_vals
        result["loss_history_sample"] = [round(v, 6) for v in sample]

    if bias_init is not None:
        result["bias_init"] = bias_init
    if bias_final is not None:
        result["bias_final"] = bias_final

    if splits:
        result["eval_splits"] = {}
        for split_seed, auc in sorted(splits.items()):
            result["eval_splits"][f"split_{split_seed}"] = {"test_auc": auc}
        result["mean_test_auc"] = round(
            np.mean(list(splits.values())), 4
        )
    elif aucs:
        result["all_aucs"] = aucs
        result["mean_test_auc"] = round(np.mean(aucs), 4)

    return result if result else None


def main():
    parser = argparse.ArgumentParser(description="Collect T6 BACE results from logs")
    parser.add_argument("--log-dir", default=".", help="Directory containing t6_seed*_stdout.log")
    parser.add_argument("--seeds", default="9,19,29", help="Comma-separated seed list")
    parser.add_argument("--output", default=None, help="Output JSON path (default: t6_bace_results.json)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    output = Path(args.output) if args.output else log_dir / "t6_bace_results.json"

    results_per_seed = {}
    all_aucs = []

    for seed in seeds:
        log_path = log_dir / f"t6_seed{seed}_stdout.log"
        if not log_path.exists():
            print(f"WARNING: {log_path} not found, skipping seed {seed}")
            continue

        result = parse_log(log_path)
        if result is None:
            print(f"WARNING: {log_path} has no parseable content, skipping")
            continue

        results_per_seed[f"seed_{seed}"] = result
        print(f"seed_{seed}: {result.get('mean_test_auc', 'N/A')}")

        # Collect all individual AUCs for grand summary
        if "eval_splits" in result:
            for v in result["eval_splits"].values():
                all_aucs.append(v["test_auc"])
        elif "all_aucs" in result:
            all_aucs.extend(result["all_aucs"])

    # Build summary
    summary = {}
    if all_aucs:
        arr = np.array(all_aucs)
        summary["all_test_aucs"] = [round(v, 4) for v in arr.tolist()]
        summary["grand_mean"] = round(float(arr.mean()), 4)
        summary["grand_std"] = round(float(arr.std()), 4)
        per_seed_means = [
            r["mean_test_auc"]
            for r in results_per_seed.values()
            if "mean_test_auc" in r
        ]
        if per_seed_means:
            summary["per_seed_means"] = per_seed_means

    out_data = {
        "experiment": "T6 Bidirectional Pair-Node Update on BACE",
        "model": "LH_Direct_V2 + NodeToPairUpdate (ChebNetII_V2 + dynamic pair + FP MLP)",
        "data_root": "down_task_v2",
        "results_per_seed": results_per_seed,
        "summary": summary,
    }

    output.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {output}")

    if summary:
        print(f"Grand mean AUC: {summary['grand_mean']} +/- {summary['grand_std']}")
    else:
        print("No AUC results found yet (training may still be running)")


if __name__ == "__main__":
    main()
