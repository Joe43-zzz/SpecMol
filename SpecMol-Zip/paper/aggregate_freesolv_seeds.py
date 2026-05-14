"""Aggregate per-seed FreeSolv HPC results into canonical results JSONs.

Reads:
    hpc/results/freesolv_v0_seed{9,19,29}.json
    hpc/results/freesolv_v2t5_seed{9,19,29}.json
Writes:
    freesolv_baseline_results.json
    freesolv_v2t5_results.json

Each per-seed JSON is the output of run_freesolv_{baseline,v2t5}.py with
--results-path set to a per-seed file. We just merge their results_per_seed
dicts and recompute the summary.

Usage:
    python paper/aggregate_freesolv_seeds.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HPC_RESULTS = REPO / "hpc" / "results"

VARIANTS = {
    "baseline": {
        "prefix": "freesolv_v0_seed",
        "out": REPO / "freesolv_baseline_results.json",
        "experiment": "FreeSolv Baseline (2D-only LH_Direct)",
        "data_root": "down_task_freesolv_2d",
    },
    "v2t5": {
        "prefix": "freesolv_v2t5_seed",
        "out": REPO / "freesolv_v2t5_results.json",
        "experiment": "FreeSolv V2-T5 (LH_Direct_V2 + GBF pair_repr)",
        "data_root": "down_task_freesolv_v2",
    },
}

SEEDS = (9, 19, 29)


def aggregate(variant_key: str, cfg: dict) -> None:
    per_seed: dict[str, dict] = {}
    missing: list[int] = []
    for seed in SEEDS:
        path = HPC_RESULTS / f"{cfg['prefix']}{seed}.json"
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
        print(f"[warn] {variant_key}: missing per-seed results for seeds {missing}")
    if not per_seed:
        print(f"[skip] {variant_key}: no per-seed results found")
        return

    rmses = [v["best_test_rmse"] for v in per_seed.values()]
    summary = {
        "all_test_rmses": rmses,
        "mean_rmse": round(statistics.mean(rmses), 4),
        "std_rmse": round(statistics.stdev(rmses) if len(rmses) > 1 else 0.0, 4),
    }

    merged = {
        "experiment": cfg["experiment"],
        "data_root": cfg["data_root"],
        "metric": "RMSE (lower is better)",
        "results_per_seed": per_seed,
        "summary": summary,
    }
    if variant_key == "v2t5":
        merged["pair_repr_source"] = "RDKit 3D + GBF expansion (64-dim, max_dist=10A, sigma=0.5)"

    cfg["out"].write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[ok] wrote {cfg['out'].name}  mean={summary['mean_rmse']}  std={summary['std_rmse']}")


def main() -> None:
    for key, cfg in VARIANTS.items():
        aggregate(key, cfg)


if __name__ == "__main__":
    main()
