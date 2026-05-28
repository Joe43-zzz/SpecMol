"""Aggregate per-seed regression HPC results into canonical results JSONs.

Generalizes paper/aggregate_freesolv_seeds.py to {esol, lipo, freesolv} via a
task-config table. Reads `hpc/results/{task}_{variant}_seed{S}.json` per-seed
files produced by `hpc/run_regression_v2t5.sbatch`.

Reads:
    hpc/results/{task}_v2t5_seed{9,19,29}.json
    hpc/results/{task}_t7_seed{9,19,29}.json    (optional)
Writes:
    {task}_v2t5_results.json              (V2-T5 + Uni-Mol pair)
    {task}_t7_bare_results.json           (T7 + Uni-Mol pair, if seeds present)

Usage:
    python paper/aggregate_regression_seeds.py --task esol
    python paper/aggregate_regression_seeds.py --task lipo
    python paper/aggregate_regression_seeds.py            # all 3
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HPC_RESULTS = REPO / "hpc" / "results"

TASKS = ("esol", "lipo", "freesolv")
SEEDS = (9, 19, 29)

VARIANT_CONFIG = {
    "v2t5": {
        "filename_template": "{task}_v2t5_results.json",
        "experiment_template": "{task_upper} V2-T5 (LH_Direct_V2 + Uni-Mol pair_repr)",
        "data_root_template": "down_task_{task}_unimol_v2",
        "pair_source": "Uni-Mol encoder_pair_rep (64-dim)",
    },
    "t7": {
        "filename_template": "{task}_t7_bare_results.json",
        "experiment_template": "{task_upper} T7 (LH_Direct_V2 + Uni-Mol pair + pair-biased attention)",
        "data_root_template": "down_task_{task}_unimol_v2",
        "pair_source": "Uni-Mol encoder_pair_rep (64-dim)",
    },
}


def aggregate(task: str, variant: str) -> None:
    cfg = VARIANT_CONFIG[variant]
    per_seed: dict[str, dict] = {}
    missing: list[int] = []
    for seed in SEEDS:
        path = HPC_RESULTS / f"{task}_{variant}_seed{seed}.json"
        if not path.exists():
            missing.append(seed)
            continue
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        seed_block = data.get("results_per_seed", {}).get(f"seed_{seed}")
        if seed_block is None:
            print(f"[warn] {path.name} has no results_per_seed.seed_{seed} entry")
            continue
        per_seed[f"seed_{seed}"] = seed_block

    if missing:
        print(f"[warn] {task}/{variant}: missing per-seed results for seeds {missing}")
    if not per_seed:
        print(f"[skip] {task}/{variant}: no per-seed results found")
        return

    rmses = [v["best_test_rmse"] for v in per_seed.values()]
    summary = {
        "all_test_rmses": rmses,
        "mean_rmse": round(statistics.mean(rmses), 4),
        "std_rmse": round(statistics.stdev(rmses) if len(rmses) > 1 else 0.0, 4),
    }

    merged = {
        "task": task,
        "variant": variant,
        "experiment": cfg["experiment_template"].format(
            task=task, task_upper=task.upper()
        ),
        "data_root": cfg["data_root_template"].format(task=task),
        "metric": "RMSE (lower is better)",
        "pair_repr_source": cfg["pair_source"],
        "results_per_seed": per_seed,
        "summary": summary,
    }

    out_path = REPO / cfg["filename_template"].format(task=task)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(
        f"[ok] {task}/{variant}: wrote {out_path.name}  "
        f"mean={summary['mean_rmse']}  std={summary['std_rmse']}  n={len(per_seed)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="Aggregate one task (default: all)")
    parser.add_argument("--variant", choices=list(VARIANT_CONFIG), default=None,
                        help="Aggregate one variant (default: all)")
    args = parser.parse_args()

    tasks = (args.task,) if args.task else TASKS
    variants = (args.variant,) if args.variant else tuple(VARIANT_CONFIG)

    for task in tasks:
        for variant in variants:
            aggregate(task, variant)


if __name__ == "__main__":
    main()
