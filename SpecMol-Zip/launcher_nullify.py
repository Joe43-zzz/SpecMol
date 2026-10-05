"""Launcher: run each nullify seed in a separate subprocess for RAM isolation."""

import subprocess
import sys
import time
import json
import os
import numpy as np

PYTHON = r"D:\SpecMol-Zip\.venv\Scripts\python.exe"
SCRIPT = "run_v2_t5_nullify_bace.py"
RESULTS_PATH = "v2_t5_nullify_bace_results.json"
SEEDS = [19, 29]  # seed 9 already done


def seed_done(seed):
    if not os.path.exists(RESULTS_PATH):
        return False
    with open(RESULTS_PATH) as f:
        d = json.load(f)
    return f"seed_{seed}" in d.get("results_per_seed", {})


def run_seed(seed):
    print(f"\n{'='*60}")
    print(f"LAUNCHER: Starting subprocess for seed {seed}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, "-u", SCRIPT, "--only-seed", str(seed)],
            timeout=4500,  # 75 min per seed
        )
        rc = result.returncode
    except subprocess.TimeoutExpired:
        rc = -1
        print(f"\nLAUNCHER: seed {seed} TIMED OUT after 4500s", flush=True)
    elapsed = time.time() - t0

    if rc == 0:
        print(f"\nLAUNCHER: seed {seed} completed OK in {elapsed:.0f}s", flush=True)
    elif rc != -1:
        print(f"\nLAUNCHER: seed {seed} FAILED (exit code {rc}) after {elapsed:.0f}s",
              flush=True)
    return rc


if __name__ == "__main__":
    print(f"LAUNCHER started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {PYTHON}")
    print(f"Seeds to run: {SEEDS}", flush=True)

    results = {}
    for seed in SEEDS:
        if seed_done(seed):
            print(f"\nLAUNCHER: seed {seed} already done, skipping", flush=True)
            results[seed] = "skipped"
            continue

        rc = run_seed(seed)
        results[seed] = "ok" if rc == 0 else f"failed(rc={rc})"

        if rc != 0:
            print(f"LAUNCHER: seed {seed} failed, continuing to next seed...",
                  flush=True)

        # Brief pause to let OS reclaim RAM
        print("LAUNCHER: Waiting 10s for OS memory reclaim...", flush=True)
        time.sleep(10)

    # Final summary
    print(f"\n{'='*60}")
    print(f"LAUNCHER DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for seed, status in results.items():
        print(f"  seed {seed}: {status}")

    # Print final comparison if results exist
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            d = json.load(f)
        completed = [s for s in [9, 19, 29]
                     if f"seed_{s}" in d.get("results_per_seed", {})]
        all_aucs = []
        for s in completed:
            all_aucs.extend([v["test_auc"]
                             for v in d["results_per_seed"][f"seed_{s}"]["eval_splits"].values()])
        if all_aucs:
            print(f"\nCompleted seeds: {completed} ({len(all_aucs)} cells)")
            print(f"V2-T5-NULLIFY: mean={np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
            print(f"V2-T5:         mean=0.837 +/- 0.009")
            print(f"Baseline V0:   mean=0.797 +/- 0.085")
            print(f"FP-only:       mean=0.846 +/- 0.010")
    print(f"{'='*60}", flush=True)
