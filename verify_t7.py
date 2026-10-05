import statistics

# T7 FreeSolv: 3 seeds
t7_rmses = [0.668, 0.6279, 0.7648]
t7_mean = statistics.mean(t7_rmses)
t7_std = statistics.stdev(t7_rmses)
print("FreeSolv T7 (n=3)")
print("  RMSEs:", t7_rmses)
print("  mean@6dp:", round(t7_mean, 6))
print("  sample_std@6dp:", round(t7_std, 6))
print("  mean@3dp:", round(t7_mean, 3), " std@3dp:", round(t7_std, 3))
print()
print("Table T7 cell: 0.687 +/- 0.070")
print("  mean match:", round(t7_mean,3)==0.687, " std match:", round(t7_std,3)==0.070)
print()

# Now check: make_tables reads freesolv_t7_bare_results.json via:
# t7_seeds = _freesolv_per_seed(t7_raw)  which reads results_per_seed.*.best_test_rmse
# = [0.668, 0.6279, 0.7648]  <- correct
# HOWEVER: parse_regression_matrix also has ("T7", "freesolv_t7_bare_results.json")
# Let's check what _freesolv_per_seed returns for that file
# It reads results_per_seed values, each has best_test_rmse. Correct.

# Make sure T7 is not double-inserted
# In main(): freesolv = parse_regression_matrix("freesolv")
# parse_regression_matrix includes ("T7", "freesolv_t7_bare_results.json")
# Then separately: for (freesolv, "freesolv_t7_bare_results.json"): if "T7" in table: continue
# So T7 is only inserted ONCE - by parse_regression_matrix, and the loop at ~519 skips it.
print("T7 double-insertion check: parse_regression_matrix inserts T7 first,")
print("  then the loop at line ~519 checks 'if T7 in table: continue' -- safe, no double insert")
