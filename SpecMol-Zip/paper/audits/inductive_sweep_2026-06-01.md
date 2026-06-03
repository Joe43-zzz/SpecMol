# Inductive-protocol sweep archive (2026-06-01)

The paper's 'deep model is weak (BACE V0 0.758)' is a measurement artifact of an under-trained frozen probe -- 0.758 does not reproduce even under its own protocol (->0.823). Properly measured, deep V0 reaches BACE finetune 0.886 (~RF 0.894, tie) and BBBP finetune 0.864 (>RF 0.81). 3D injection (V2-T5/T7) = random = V0 robustly, including under the clean inductive (train-only) pretraining protocol. Transductive pretraining ('all') gives ~0 benefit to classification finetune and only a moderate FreeSolv-regression edge.

**Protocol**: 1000ep pretrain, n=3 init seeds, single fixed split; frozen=2000ep LogReg, finetune=encoder unfrozen (40ep, patience 15). `--pretrain_split all|train` controls transductive vs inductive pretraining.

## Frozen vs Finetune (V0)

| dataset | metric | split | frozen | finetune | Δ(ft−fr) |
|---|---|---|---|---|---|
| bace | Auc | all | 0.8224 | 0.8859 | +0.0635 |
| bace | Auc | train | 0.8741 | 0.875 | +0.0009 |
| bbbp | Auc | all | 0.7626 | 0.8604 | +0.0978 |
| bbbp | Auc | train | 0.7846 | 0.8643 | +0.0797 |
| freesolv | RMSE | all | 0.6559 | 0.6021 | -0.0538 |
| freesolv | RMSE | train | 0.6577 | 0.6656 | +0.0079 |

## Finetune 3D arms (mean)

| dataset | metric | V0 all/train | V2T5 all/train | T7 all/train | random all/train |
|---|---|---|---|---|---|
| bace | Auc | 0.8859 / 0.875 | 0.8801 / 0.8798 | 0.8857 / 0.8836 | 0.8725 / 0.8861 |
| bbbp | Auc | 0.8604 / 0.8643 | 0.8428 / 0.8635 | 0.8634 / 0.8621 | 0.8471 / 0.8515 |
| freesolv | RMSE | 0.6021 / 0.6656 | 0.6632 / 0.6469 | 0.6285 / 0.6645 | 0.6303 / 0.6307 |

## Decision gates

### gate1_inductive_preserves_conclusions
- **verdict**: YES
- note: 3D=random holds under both all and train (all 6 finetune cells tie within seed noise); finetune numbers barely move all->train for classification.

### gate2_real_finetune_vs_frozen_delta
- **verdict**: +0.00 to +0.10, dataset-dependent (the '+11' was a cross-protocol artifact vs paper 0.758)
- bbbp: finetune beats frozen by +0.079..+0.097 (both protocols)
- bace: all: +0.064 ; train: +0.001 (tie)
- freesolv: all: finetune better by 0.054 RMSE ; train: finetune worse by 0.008 (tie)

### gate3_leakage_magnitude_train_vs_all_finetune
- **verdict**: classification ~0, FreeSolv regression moderate
- bace_V0: -0.0109
- bbbp_V0: 0.0039
- freesolv_V0_rmse: 0.0635

### gate4_0758_reconciliation_P6
- **verdict**: 0.758 does NOT reproduce; it is an under-trained-probe artifact
- P6_scaffold_3split_frozen_V0_mean: 0.8234
- single_fold_frozen_V0_all_mean: 0.8224
- paper_canonical: 0.758
- note: P6 reproduces the canonical protocol (3 scaffold splits, all-pretrain, frozen V0, 1000ep) -> 0.823, not 0.758. Properly-trained frozen V0 ~0.82 regardless of split.

## Provenance

- runner: `run_overnight_sweep.sh (slot scheduler, 4 GPUs)`
- logs: `thk:~/specmol/logs/sweep_*.log`
- full per-seed data: `inductive_sweep_2026-06-01.json` (33 jobs, all rc=0)