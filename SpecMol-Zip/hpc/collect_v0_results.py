#!/usr/bin/env python
"""Collect V0 baseline Slurm logs into JSON and CSV summaries."""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


CLASS_LOG_RE = re.compile(r"v0_(?P<task>[a-z0-9]+)_seed(?P<seed>\d+)_.*\.(?:out|log)$")
BEST_AUC_RE = re.compile(
    r"Best Test Auc for (?P<task>[A-Za-z0-9_]+) in Epoch (?P<epoch>\d+) is (?P<value>[0-9.eE+-]+)"
)
FREESOLV_RE = re.compile(r"freesolv_seed(?P<seed>\d+)\.json$")


def parse_int_csv(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_detail_row(dataset, metric, direction, pretrain_seed, eval_split_seed, value, source_log):
    return {
        "row_type": "detail",
        "dataset": dataset,
        "metric": metric,
        "direction": direction,
        "pretrain_seed": pretrain_seed,
        "eval_split_seed": eval_split_seed,
        "value": value,
        "mean": "",
        "std": "",
        "n": "",
        "source_log": source_log,
    }


def make_summary_row(dataset, metric, direction, values):
    mean = statistics.fmean(values) if values else ""
    std = statistics.pstdev(values) if len(values) > 1 else 0.0 if values else ""
    return {
        "row_type": "summary",
        "dataset": dataset,
        "metric": metric,
        "direction": direction,
        "pretrain_seed": "",
        "eval_split_seed": "",
        "value": "",
        "mean": round(mean, 6) if values else "",
        "std": round(std, 6) if values else "",
        "n": len(values),
        "source_log": "",
    }


def parse_classification_logs(logs_dir, eval_seeds):
    rows = []
    for path in sorted(logs_dir.glob("v0_*_seed*_*.out")):
        name_match = CLASS_LOG_RE.match(path.name)
        if not name_match:
            continue

        pretrain_seed = int(name_match.group("seed"))
        fallback_task = name_match.group("task")
        occurrence = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                result_match = BEST_AUC_RE.search(line)
                if not result_match:
                    continue
                task = result_match.group("task").lower() or fallback_task
                eval_split_seed = (
                    eval_seeds[occurrence]
                    if occurrence < len(eval_seeds)
                    else occurrence + 1
                )
                rows.append(
                    make_detail_row(
                        dataset=task,
                        metric="auc",
                        direction="higher_is_better",
                        pretrain_seed=pretrain_seed,
                        eval_split_seed=eval_split_seed,
                        value=float(result_match.group("value")),
                        source_log=str(path),
                    )
                )
                rows[-1]["best_eval_epoch"] = int(result_match.group("epoch"))
                occurrence += 1
    return rows


def parse_freesolv_results(results_dir):
    rows = []
    for path in sorted(results_dir.glob("freesolv_seed*.json")):
        name_match = FREESOLV_RE.match(path.name)
        if not name_match:
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        for key, result in sorted(data.get("results_per_seed", {}).items()):
            seed_match = re.search(r"(\d+)", key)
            pretrain_seed = int(seed_match.group(1)) if seed_match else int(name_match.group("seed"))
            rows.append(
                make_detail_row(
                    dataset="freesolv",
                    metric="rmse",
                    direction="lower_is_better",
                    pretrain_seed=pretrain_seed,
                    eval_split_seed="",
                    value=float(result["best_test_rmse"]),
                    source_log=str(path),
                )
            )
            rows[-1]["best_eval_epoch"] = result.get("best_eval_epoch", "")
    return rows


def summarize(detail_rows):
    values_by_dataset = defaultdict(list)
    meta_by_dataset = {}
    for row in detail_rows:
        key = (row["dataset"], row["metric"], row["direction"])
        values_by_dataset[key].append(float(row["value"]))
        meta_by_dataset[key] = (row["dataset"], row["metric"], row["direction"])

    summaries = []
    for key in sorted(values_by_dataset):
        dataset, metric, direction = meta_by_dataset[key]
        summaries.append(make_summary_row(dataset, metric, direction, values_by_dataset[key]))
    return summaries


def write_csv(path, rows):
    fieldnames = [
        "row_type",
        "dataset",
        "metric",
        "direction",
        "pretrain_seed",
        "eval_split_seed",
        "best_eval_epoch",
        "value",
        "mean",
        "std",
        "n",
        "source_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="hpc/logs")
    parser.add_argument("--results", default="hpc/results")
    parser.add_argument("--out", default="hpc/results/v0_summary")
    parser.add_argument("--eval-seeds", default="9,19,29")
    args = parser.parse_args()

    logs_dir = Path(args.logs)
    results_dir = Path(args.results)
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    eval_seeds = parse_int_csv(args.eval_seeds)
    detail_rows = parse_classification_logs(logs_dir, eval_seeds)
    detail_rows.extend(parse_freesolv_results(results_dir))
    summary_rows = summarize(detail_rows)
    rows = detail_rows + summary_rows

    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "details": detail_rows,
                "summaries": summary_rows,
            },
            handle,
            indent=2,
        )
    write_csv(csv_path, rows)

    print(f"[collect] detail_rows={len(detail_rows)} summary_rows={len(summary_rows)}")
    print(f"[collect] wrote {json_path}")
    print(f"[collect] wrote {csv_path}")


if __name__ == "__main__":
    main()
