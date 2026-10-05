# FreeSolv 0.6746 Verification Report

## Finding 1: RMSE formula — **safe**

Code in `run_freesolv_v2t5.py:196,210`:
```python
val_rmse = torch.sqrt(loss_fn(val_preds, val_y)).item()   # loss_fn = MSELoss(mean)
best_test_rmse = torch.sqrt(loss_fn(test_preds, test_y)).item()
```
RMSE = sqrt(mean((pred - y)^2)). Verified numerically: torch result matches manual numpy. Formula is correct.

## Finding 2: Test set is held-out — **safe**

- Train/val/test loaded from separate .pt files (`freesolv_train.pt`, `freesolv_valid.pt`, `freesolv_test.pt`)
- Split sizes: train=513, val=64, test=65, total=642
- SMILES sets verified disjoint: 0 overlap between any pair
- Model selection: best val RMSE selects which epoch's test RMSE is reported. Test set is never used for training or model selection.
- Pretrain uses `freesolv_all.pt` (all 642 molecules) for contrastive learning, which includes test molecules — but pretrain is self-supervised (no labels used), so this is not a label leak.

## Finding 3: Labels are z-scored — **suspicious (misleading, not wrong)**

DeepChem 2.8.0 `load_freesolv()` applies `NormalizationTransformer` by default. The shipped `freesolv.csv.gz` already contains z-scored labels (mean=0, std=1). Our code called `load_freesolv(featurizer='Raw', splitter=None)` without `transformers=[]`, so we got z-scored labels.

Estimated conversion from two known molecules (methanol, benzene):
- Raw FreeSolv mean ~ -3.80 kcal/mol, std ~ 3.84 kcal/mol

| Reported (z-score) | Actual (kcal/mol) |
|---------------------|-------------------|
| V2-T5: 0.6746 | ~2.59 kcal/mol |
| Baseline: 0.7604 | ~2.92 kcal/mol |
| Ridge: 1.0465 | ~4.02 kcal/mol |
| Uni-Mol (literature): — | 1.48 kcal/mol |

**Our 0.6746 is NOT in kcal/mol. In kcal/mol it's ~2.6, which is worse than Uni-Mol's 1.48.** The number is real but the unit is wrong for literature comparison. The relative comparison (V2-T5 vs Baseline) remains valid since both use the same labels.

## Finding 4: Split protocol — **safe (but different from literature)**

- `ScaffoldSplitter(seed=42)`, frac 80/10/10
- DeepChem default for FreeSolv is `splitter='random'`, not scaffold. We used scaffold, which is harder.
- Uni-Mol paper likely uses random split. Split mismatch means our absolute RMSE is not directly comparable to Uni-Mol's.
- Test set: 65 molecules (all SMILES listed and inspected; chemically diverse, no obvious anomalies).

## Finding 5: Simple baseline sanity — **safe**

Same data, same split:

| Method | Test RMSE (z-score) |
|--------|---------------------|
| Mean predictor | 1.1658 |
| Test label std | 0.9456 |
| Ridge (best alpha=10) | 1.0465 |
| Our Baseline | 0.7604 |
| Our V2-T5 | 0.6746 |

Ridge barely beats mean prediction. Our models substantially beat Ridge. The ordering is plausible. No evidence of data leak.

## Summary

| Check | Verdict |
|-------|---------|
| RMSE formula correct | **safe** |
| Test set held-out, no label leak | **safe** |
| Labels are z-scored (not kcal/mol) | **suspicious** — number real, unit misleading for literature comparison |
| Split is scaffold (literature uses random) | **safe** — valid internally, not comparable to random-split benchmarks |
| Simple baseline confirms magnitude | **safe** — 0.67 is plausible given Ridge=1.05 on same data |

**Bottom line**: The 0.6746 number is real and correctly computed. The V2-T5 vs Baseline comparison is fair. But it's in z-score units, not kcal/mol. In kcal/mol it would be ~2.6, which is not state-of-the-art. Any reporting must clarify the label normalization.
