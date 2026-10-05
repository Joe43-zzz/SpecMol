import statistics

v0_rmses = [0.6486, 0.6456, 0.7015, 0.6762, 0.6372, 0.62, 0.6604, 0.6169, 0.6149]
v0_mean = statistics.mean(v0_rmses)
v0_std = statistics.stdev(v0_rmses)
print("FreeSolv V0 (n=9)")
print("  RMSEs:", v0_rmses)
print("  mean =", round(v0_mean, 6))
print("  sample_std(ddof=1) =", round(v0_std, 6))
print("  mean@4dp:", round(v0_mean,4), " std@4dp:", round(v0_std,4))
print("  mean@3dp:", round(v0_mean,3), " std@3dp:", round(v0_std,3))
print()

v2t5_rmses = [0.6412, 0.6529, 0.619, 0.7526, 0.6381, 0.6856, 0.6403, 0.8071, 0.6233]
v2t5_mean = statistics.mean(v2t5_rmses)
v2t5_std = statistics.stdev(v2t5_rmses)
print("FreeSolv V2-T5 unimol (n=9)")
print("  RMSEs:", v2t5_rmses)
print("  mean =", round(v2t5_mean, 6))
print("  sample_std(ddof=1) =", round(v2t5_std, 6))
print("  mean@4dp:", round(v2t5_mean,4), " std@4dp:", round(v2t5_std,4))
print("  mean@3dp:", round(v2t5_mean,3), " std@3dp:", round(v2t5_std,3))
print()

print("Table renders: V0=0.647+-0.029  V2-T5=0.673+-0.065")
print()
print("V0  mean match:", round(v0_mean,3)==0.647, "std match:", round(v0_std,3)==0.029)
print("V2T5 mean match:", round(v2t5_mean,3)==0.673, "std match:", round(v2t5_std,3)==0.065)
print()
print("JSON summary check:")
print("  v0 json mean=0.6468 match:", round(v0_mean,4)==0.6468, " std=0.0291 match:", round(v0_std,4)==0.0291)
print("  v2t5 json mean=0.6733 match:", round(v2t5_mean,4)==0.6733, " std=0.0648 match:", round(v2t5_std,4)==0.0648)
