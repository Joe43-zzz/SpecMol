# Uni-Mol end-to-end finetune baseline — runbook

**Why:** the paper's `Uni-Mol` row is a *frozen* Uni-Mol embedding + RandomForest
control, not Uni-Mol's own performance. The original Uni-Mol paper (ICLR 2023)
reports end-to-end *finetune* numbers (BACE 0.857, ClinTox 0.919, ESOL 0.788,
FreeSolv 1.480, Lipo 0.603). A reviewer reads a row labelled "Uni-Mol" as the
latter. This produces the real finetune number on the **same matched split**
every other cell uses, so it is protocol-matched to V0/V2-T5/T7/RF.

## Files
- `reproduce_unimol_finetune.py` — reuses `baselines_matched.load_split` (identical
  fold0=test partition); estimator = `unimol_tools.MolTrain` finetune + `MolPredict`.
  bbbp/bace re-pointed at `paper/audits/{bbbp,bace}_fold_recovered.csv`.
- `hpc/run_unimol_finetune.sbatch` — gpu partition, `conda activate specmol`.

## Run (MBZUAI HPC, GPU node only)
```bash
# 1. connect VPN, ssh mbzuai-hpc, cd to repo, git pull
cd ~/zhoutianyang/SpecMol/SpecMol-Zip

# 2. smoke test ONE cheap cell first (FreeSolv, 642 mols) to catch API issues
DATASETS="freesolv" SEEDS="9" EPOCHS=20 sbatch hpc/run_unimol_finetune.sbatch
#    watch hpc/logs/unimol_finetune_*.out ; confirm it writes unimol_finetune_results.json

# 3. the two sharpest cells (where it most changes the story)
DATASETS="bace clintox" sbatch hpc/run_unimol_finetune.sbatch

# 4. the rest -> all six
DATASETS="bbbp freesolv esol lipo" sbatch hpc/run_unimol_finetune.sbatch
```
Per-dataset checkpointing into `unimol_finetune_results.json`; rerunning resumes/accumulates.

## Expected wall-clock (RTX 5000 Ada, 3 seeds, 50 epochs)
FreeSolv/ClinTox/BACE/BBBP (≤2k mols): ~10–30 min each. ESOL ~30 min. Lipo (4.2k): ~1–2 h.
All six ≈ 4–6 GPU-h. Comfortable within the 12 h wall.

## Risks to watch (check the log)
1. **Seed variance.** `MolTrain` has no `seed` arg; we perturb global RNGs instead.
   The script prints `WARNING: <metric> identical across all N seeds` if the std is
   a fake 0. If you see that, the std is not a real seed std — report n=3 mean and
   say std is the internal-CV spread, or run a single-cell sensitivity check.
2. **API drift.** Verified against unimol_tools `main` (MolTrain: task/data_type/
   epochs/learning_rate/batch_size/kfold/metrics/save_path/remove_hs/smiles_col/
   target_cols; fit(data=csv_path); MolPredict(load_model).predict(data=csv_path)).
   If the HPC-installed version differs, the smoke cell (step 2) surfaces it cheaply.
3. **ClinTox.** task auto-set to `multilabel_classification` (2 tasks); metric is
   macro-AUC over tasks, matching the paper's ClinTox convention.

## Integrating results into the paper
1. **Rename the existing control row** `Uni-Mol` → `Uni-Mol-emb+RF` (frozen probe)
   in `paper/tables/*.tex` captions / `paper/make_tables.py`, so it no longer reads
   as Uni-Mol's own performance.
2. **Add a new row** `Uni-Mol (finetune)` from `unimol_finetune_results.json`.
3. Update narrative:
   - If finetuned Uni-Mol still **loses to RF** on BACE/ESOL/Lipo → strengthens C2
     (fingerprint saturation beats even a properly-finetuned 3D SOTA).
   - Where finetuned Uni-Mol **beats** RF (likely ClinTox) → state it honestly;
     it bounds the "fingerprint is the ceiling" claim to the saturated endpoints.
   - V2-T5 (frozen-pair injection) sitting **below** finetuned Uni-Mol is consistent
     with the paper's "frozen injection doesn't recover the 3D signal" framing.
