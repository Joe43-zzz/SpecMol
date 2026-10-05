# FreeSolv Experiment Results Report

## Status: COMPLETE (all 6/6 cells done)

- Started: 2026-05-03 00:27
- Finished: 2026-05-03 12:57 (wall clock inflated by machine sleep overnight)

## Final Results

| Method | Seed 9 | Seed 19 | Seed 29 | **Mean RMSE** | **Std** |
|--------|--------|---------|---------|---------------|---------|
| **Baseline (2D-only)** | 0.7221 | 0.7320 | 0.8272 | **0.7604** | 0.0474 |
| **V2-T5 (GBF 3D)** | 0.6297 | 0.6889 | 0.7053 | **0.6746** | 0.0325 |

**V2-T5 improves over Baseline by 11.3% in RMSE** (0.7604 -> 0.6746), with lower variance.

### PairToEdgeWeight Bias Evolution
| Seed | Bias Init | Bias Final | Delta |
|------|-----------|------------|-------|
| 9 | 5.0000 | 4.7091 | -0.291 |
| 19 | 5.0000 | 4.8128 | -0.187 |
| 29 | 5.0000 | 4.7984 | -0.202 |

Bias decreases from 5.0 (sigmoid~0.993) to ~4.77 (sigmoid~0.991). The model learns to slightly modulate edge weights using 3D information, improving RMSE consistently across all seeds.

## Step 1: Inspection Results

| Question | Answer |
|----------|--------|
| Regression support | NO in original code -- added MSE/RMSE downstream eval path |
| FreeSolv data | DeepChem 2.8.0 loader works, 642 valid molecules |
| Uni-Mol pair_repr | BACE only -- generated RDKit 3D + GBF (64-dim) pair_repr |
| make_bace reusable | Partially -- wrote new `prepare_freesolv_data.py` |

**Execution path**: C (full build -- regression + data pipeline + pair_repr generation)

## Step 2: Files Created

| File | Change |
|------|--------|
| `prepare_freesolv_data.py` | Download FreeSolv via DeepChem, scaffold split, build .pt for baseline + V2-T5 with GBF pair_repr |
| `run_freesolv_baseline.py` | LH_Direct baseline with MSE loss + RMSE eval downstream |
| `run_freesolv_v2t5.py` | LH_Direct_V2 with GBF pair_repr, MSE/RMSE downstream |
| `sanity_check_freesolv.py` | Verify data loads, forward/backward works for both variants |
| `launcher_freesolv.py` | Serial subprocess launcher, 90min timeout per cell |
| `dataset/freesolv/raw/smiles.csv` | 642 molecules, labels in [-5.64, 1.88] (kcal/mol) |
| `down_task_freesolv_2d/processed/` | Baseline .pt files (train=513, val=64, test=65) |
| `down_task_freesolv_v2/processed/` | V2-T5 .pt files with pair_repr_edge (same split) |

**No existing files modified.** All BACE code untouched.

## Experiment Design

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pretrain loss | Contrastive (ours_loss) | Task-agnostic, same as BACE |
| Downstream loss | MSELoss | Regression task |
| Downstream metric | RMSE (lower=better) | Standard for FreeSolv |
| Pretrain epochs | 1000 (patience=100) | Same as BACE |
| Eval epochs | 2000 | Same as BACE |
| Batch size | 256 | Fits 4GB VRAM |
| hid_dim | 512 | Same as BACE |
| Seeds | 9, 19, 29 | 3-seed reduced set |
| Split | Scaffold 80/10/10 | DeepChem ScaffoldSplitter |
| V2-T5 pair_repr | RDKit 3D GBF (64-dim) | Proxy for Uni-Mol |

## Key Decision: GBF pair_repr instead of Uni-Mol

No Uni-Mol pair_repr exists for FreeSolv. Generated pair representations from RDKit 3D conformers using Gaussian Basis Function expansion:
- 3D coordinates via `AllChem.EmbedMolecule` + `MMFFOptimizeMolecule`
- Pairwise Euclidean distances -> 64 Gaussian basis functions (centers 0-10A, sigma=0.5)
- Produces [N,N,64] pair_repr per molecule, same format as Uni-Mol

**Caveat**: RDKit 3D is less accurate than Uni-Mol's learned 3D. Results may underestimate V2-T5 potential with proper Uni-Mol pair_repr.

## Fairness / Reproducibility Check

- Same scaffold split (DeepChem ScaffoldSplitter, default seed=42) for both variants
- Same pretrain hyperparams, same eval hyperparams
- Same 3 pretrain seeds (9, 19, 29)
- Only difference: V2-T5 has pair_repr_edge + PairToEdgeWeight module
- Baseline uses bond-only edge_index; V2-T5 additionally computes edge weights from pair_repr on those same bond edges

## Raw Result Files

- `freesolv_baseline_results.json` -- full baseline results
- `freesolv_v2t5_results.json` -- full V2-T5 results
- `launcher_freesolv.log` -- complete training log
- `launcher_freesolv_status.json` -- cell completion status

## Risks / Caveats

1. **Small test set** (65 molecules) -- results may have high variance due to test set size
2. **GBF vs Uni-Mol** -- RDKit conformers are MMFF-optimized, not Uni-Mol encoder outputs. The pair_repr quality is lower.
3. **Single split** -- only one scaffold split used (no eval_seeds rotation like BACE). Adding more splits would improve statistical confidence.
4. **Label range** -- FreeSolv labels span [-5.64, 1.88] kcal/mol. RMSE of ~0.67 is moderate for this range.
