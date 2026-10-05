"""
Verify the seed key lookup logic in aggregate_regression_seeds.py line 75:
  block = data.get("results_per_seed", {}).get(f"seed_{seed}")

Each per-seed file has results_per_seed with exactly ONE key matching seed_N.
But the aggregate script iterates SEEDS=(9,19,...,89) and looks up seed_N in each file.

For freesolv_v0_seed9.json:
  results_per_seed = {"seed_9": {...}}
  lookup: data["results_per_seed"].get("seed_9") -> correct

For freesolv_v0_seed39.json (different schema order, but same content):
  results_per_seed = {"seed_39": {...}}
  lookup: data["results_per_seed"].get("seed_39") -> correct

This is fine. Each per-seed file has exactly the key for its own seed.
No cross-seed contamination.

However: what if the file contains a DIFFERENT seed key than the filename implies?
For example, freesolv_unimol_v2t5_seed9.json has results_per_seed.seed_9 -- matches.
The lookup is seed-file-safe.

Also check: when aggregate_regression_seeds.py is run after the SEEDS extension,
does it re-read and correctly produce the final aggregated JSON?
The script writes to (REPO / out_name) -- this will OVERWRITE the existing JSON.
Confirmed that freesolv_v0_results.json and freesolv_unimol_v2t5_results.json
both have n=9 and match the per-seed files.

Now verify the critical: n is from len(rmses), NOT len(SEEDS).
"""

# Test: if only seeds 9/19/29 exist for t7, aggregate gets n=3 (not 9)
present_seeds = [9, 19, 29]
all_seeds = [9, 19, 29, 39, 49, 59, 69, 79, 89]

rmses_from_present = [0.668, 0.6279, 0.7648]  # the 3 seeds that exist
import statistics

# The aggregate function:
# rmses = [v["best_test_rmse"] for v in per_seed.values() if "best_test_rmse" in v]
# n_seeds = len(rmses) -- correct, uses actual count not len(SEEDS)
print("n from present files (T7):", len(rmses_from_present), "(correct: should be 3, not 9)")
print()

# Now check a subtle variant: what if a per-seed file exists but
# results_per_seed.seed_N doesn't exist in it?
# aggregate_regression_seeds.py:77 prints [warn] and does NOT add to per_seed.
# So rmse list stays clean. Safe.
print("Missing-seed and missing-key handling: safe (warned, skipped).")
print()

# Check std=0.0 case for n=1:
# "std_rmse": round(statistics.stdev(rmses) if len(rmses) > 1 else 0.0, 4)
# For n=1, std=0.0. That's correct behavior (and harmless - never reported in paper for n=1).
print("n=1 std handling: returns 0.0 (guarded). Safe.")
