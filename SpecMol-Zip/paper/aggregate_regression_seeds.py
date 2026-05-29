"""Aggregate per-seed regression HPC results into canonical results JSONs.

Reads `hpc/results/<prefix>_seed{9,19,29}.json` (one seed per file, schema
``{results_per_seed: {seed_N: {best_test_rmse, ...}}, summary: {...}}``, as
written by `hpc/run_regression_v2t5.sbatch`) and writes one aggregated
`<task>_<variant>_results.json` per (task, variant) that
`paper/make_tables.py` consumes.

Variants: v0, v2t5, t6, t7. The FreeSolv V2-T5 per-seed files carry the
``freesolv_unimol_v2t5`` prefix (the ``down_task_freesolv_unimol_v2`` pipeline,
GBF pair features) and aggregate to ``freesolv_unimol_v2t5_results.json``, the
file ``make_tables.py`` prefers for the FreeSolv V2-T5 cell. All other
(task, variant) pairs follow the plain ``<task>_<variant>`` convention.

Usage:
    python paper/aggregate_regression_seeds.py                    # all tasks, all variants
    python paper/aggregate_regression_seeds.py --task esol        # one task
    python paper/aggregate_regression_seeds.py --variant t6       # one variant
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HPC_RESULTS = REPO / "hpc" / "results"

TASKS = ("esol", "lipo", "freesolv")
VARIANTS = ("v0", "v2t5", "t6", "t7")
# FreeSolv V0/V2-T5 extended to n=9 (Phase B3, 2026-05-30). Other (task,variant)
# pairs only have seeds 9/19/29 on disk; aggregate() reports n from the files
# that actually exist (len(rmses)), so listing 9 seeds here is harmless for them
# (it just emits a "missing seeds" warning and aggregates the partial set).
SEEDS = (9, 19, 29, 39, 49, 59, 69, 79, 89)

EXPERIMENT = {
    "v0": "{U} V0 (LH_Direct 2D baseline)",
    "v2t5": "{U} V2-T5 (LH_Direct_V2 + Uni-Mol pair_repr static gate)",
    "t6": "{U} T6 (LH_Direct_V2 dynamic pair-node update + Uni-Mol pair_repr)",
    "t7": "{U} T7 (LH_Direct_V2 + Uni-Mol pair + pair-biased attention)",
}


def _names(task: str, variant: str) -> tuple[str, str]:
    """Return ``(per_seed_prefix, output_filename)`` for a (task, variant).

    FreeSolv V2-T5 per-seed results live under the ``freesolv_unimol_v2t5``
    prefix and aggregate to ``freesolv_unimol_v2t5_results.json`` (the name
    make_tables.py prefers); everything else uses ``<task>_<variant>``.
    """
    if task == "freesolv" and variant == "v2t5":
        return "freesolv_unimol_v2t5", "freesolv_unimol_v2t5_results.json"
    out = {
        "v0": f"{task}_v0_results.json",
        "v2t5": f"{task}_v2t5_results.json",
        "t6": f"{task}_t6_results.json",
        "t7": f"{task}_t7_bare_results.json",
    }[variant]
    return f"{task}_{variant}", out


def aggregate(task: str, variant: str) -> None:
    seed_prefix, out_name = _names(task, variant)
    per_seed: dict[str, dict] = {}
    missing: list[int] = []
    for seed in SEEDS:
        path = HPC_RESULTS / f"{seed_prefix}_seed{seed}.json"
        if not path.exists():
            missing.append(seed)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get("results_per_seed", {}).get(f"seed_{seed}")
        if block is None:
            print(f"[warn] {path.name}: no results_per_seed.seed_{seed} entry")
            continue
        per_seed[f"seed_{seed}"] = block

    if not per_seed:
        print(f"[skip] {task}/{variant}: no per-seed files found"
              + (f" (missing seeds {missing})" if missing else ""))
        return
    if missing:
        print(f"[warn] {task}/{variant}: missing per-seed results for seeds {missing}"
              f" -- aggregating partial n={len(per_seed)}")

    rmses = [v["best_test_rmse"] for v in per_seed.values() if "best_test_rmse" in v]
    summary = {
        "all_test_rmses": rmses,
        "mean_rmse": round(statistics.mean(rmses), 4),
        "std_rmse": round(statistics.stdev(rmses) if len(rmses) > 1 else 0.0, 4),
        "n_seeds": len(rmses),
    }
    merged = {
        "task": task,
        "variant": variant,
        "experiment": EXPERIMENT[variant].format(U=task.upper()),
        "data_root": f"down_task_{task}_unimol_v2",
        "metric": "RMSE (lower is better)",
        "results_per_seed": per_seed,
        "summary": summary,
    }
    (REPO / out_name).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[ok] {task}/{variant}: wrote {out_name}  "
          f"mean={summary['mean_rmse']}  std={summary['std_rmse']}  n={len(rmses)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="Aggregate one task (default: all)")
    parser.add_argument("--variant", choices=VARIANTS, default=None,
                        help="Aggregate one variant (default: all)")
    args = parser.parse_args()

    tasks = (args.task,) if args.task else TASKS
    variants = (args.variant,) if args.variant else VARIANTS

    for task in tasks:
        for variant in variants:
            aggregate(task, variant)


if __name__ == "__main__":
    main()
