"""
Verify T7 aggregation:
- aggregate_regression_seeds.py uses prefix 'freesolv_t7' for (freesolv, t7)
- It looks for hpc/results/freesolv_t7_seed{9,19,29,...,89}.json
- Seeds 39-89 do NOT exist for freesolv_t7 (only seed9/19/29 exist)
- So it aggregates n=3 (partial set, emits warning)
- The result in freesolv_t7_bare_results.json should be the 3-seed mean
"""
import statistics

# From per-seed files:
t7_seeds_data = {
    9:  0.668,
    19: 0.6279,
    29: 0.7648,
}

rmses = list(t7_seeds_data.values())
m = statistics.mean(rmses)
s = statistics.stdev(rmses)
print("T7 FreeSolv (n=3 from hpc/results/freesolv_t7_seed*.json)")
print("  RMSEs:", rmses)
print("  mean@4dp:", round(m,4), " std@4dp:", round(s,4))
print("  mean@3dp:", round(m,3), " std@3dp:", round(s,3))
print()
print("freesolv_t7_bare_results.json summary: mean=0.6869, std=0.0704, n=3")
print("  mean match:", round(m,4)==0.6869, " std match:", round(s,4)==round(0.07038,4))
print()

# Also confirm make_tables reads this correctly via _freesolv_per_seed
# _freesolv_per_seed: reads results_per_seed values, each with best_test_rmse
# From the JSON: seed_9=0.668, seed_19=0.6279, seed_29=0.7648  -> [0.668, 0.6279, 0.7648]
# Then _stats: mean=0.6869, std=0.0704 -> fmt: "0.687 +/- 0.070"  matches table.
print("Table T7 cell: '0.687 +/- 0.070'")
print("  fmt match: '0.687 pm 0.070' =", f"{round(m,3):.3f} pm {round(s,3):.3f}")
