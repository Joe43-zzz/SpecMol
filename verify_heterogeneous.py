"""
Check the heterogeneous pair_repr_source in freesolv_unimol_v2t5 files.
Seeds 9/19/29 say 'RDKit 3D + GBF expansion' (GBF surrogate).
Seeds 39-89 say 'Uni-Mol encoder_pair_rep (64-dim)' (actual Uni-Mol).

This means the freesolv_unimol_v2t5_results.json is a MIX of two
different pair_repr sources. The table caption says:
'FreeSolv V2-T5/T7 use an RDKit-GBF pair surrogate'
But seeds 39-89 use actual Uni-Mol. This is a DATA CONTAMINATION issue.

Let us quantify the impact.
"""

# Seeds 9/19/29: GBF surrogate (from freesolv_unimol_v2t5_seed*.json)
gbf_seeds = {9: 0.6412, 19: 0.6529, 29: 0.619}

# Seeds 39-89: Uni-Mol actual (from freesolv_unimol_v2t5_seed*.json)
unimol_seeds = {39: 0.7526, 49: 0.6381, 59: 0.6856, 69: 0.6403, 79: 0.8071, 89: 0.6233}

import statistics

gbf_rmses = list(gbf_seeds.values())
unimol_rmses = list(unimol_seeds.values())
all_rmses = gbf_rmses + unimol_rmses

print("=== Heterogeneous pair_repr_source in freesolv_unimol_v2t5 ===")
print()
print(f"Seeds 9/19/29 (GBF surrogate):  {gbf_rmses}")
print(f"  mean={statistics.mean(gbf_rmses):.4f}  std={statistics.stdev(gbf_rmses):.4f} (n=3)")
print()
print(f"Seeds 39-89 (Uni-Mol actual):    {unimol_rmses}")
print(f"  mean={statistics.mean(unimol_rmses):.4f}  std={statistics.stdev(unimol_rmses):.4f} (n=6)")
print()
print(f"MIXED (all 9 seeds):             {all_rmses}")
print(f"  mean={statistics.mean(all_rmses):.4f}  std={statistics.stdev(all_rmses):.4f} (n=9)")
print()
print("Table caption claims: 'FreeSolv V2-T5/T7 use an RDKit-GBF pair surrogate'")
print("ACTUAL: Seeds 9/19/29 use GBF; seeds 39-89 use Uni-Mol encoder_pair_rep")
print("=> Caption is INCORRECT for seeds 39-89.")
print()

# Also check: what does the experiment field say in freesolv_unimol_v2t5_results.json?
# experiment: "FREESOLV V2-T5 (LH_Direct_V2 + Uni-Mol pair_repr static gate)"
# This says Uni-Mol pair_repr for ALL seeds, but seeds 9/19/29 actually used GBF.
print("freesolv_unimol_v2t5_results.json experiment field:")
print("  'FREESOLV V2-T5 (LH_Direct_V2 + Uni-Mol pair_repr static gate)'")
print("  But seeds 9/19/29 used GBF pair_repr (pair_repr_source from per-seed files).")
print()

# Is there a fair-comparison concern?
# The 9 runs are NOT all the same variant (GBF vs Uni-Mol pair).
# The aggregated number blends two different experimental conditions.
# This means the n=9 V2-T5 FreeSolv cell is internally heterogeneous.
print("SEVERITY: This is a protocol violation -- mixing two different pair_repr sources")
print("in one n=9 aggregate violates the 'same variant' comparison requirement.")
print()

# Check data_root too:
# seed9 file: data_root=down_task_freesolv_unimol_v2 (GBF)
# seed39 file: data_root=down_task_freesolv_unimol_v2 (Uni-Mol)
# Same data_root, different pair_repr - the variation is in the model, not the split.
# GBF produces the pair features at runtime from RDKit, Uni-Mol uses frozen encoder.
# These are different models being treated as the same variant.
print("data_root is the same (down_task_freesolv_unimol_v2) for all seeds.")
print("The difference is in the pair_repr MODEL (GBF vs Uni-Mol encoder).")
print("These are two different variants being aggregated as one.")
