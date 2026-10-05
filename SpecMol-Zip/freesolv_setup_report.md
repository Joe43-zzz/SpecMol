# FreeSolv Setup Report

## Step 1: Project Inspection

### Q1: Regression support?
**NO.** `main_pretrain.py` uses `BCEWithLogitsLoss` + `roc_auc_score`. `clr_tasks` dict only has classification tasks. Downstream eval uses `calculate_auc` with ROC-AUC. FreeSolv is regression (hydration free energy).

### Q2: TestbedDataset can load FreeSolv?
**Not directly.** Expects `dataset/{task}/raw/smiles.csv`. No FreeSolv CSV exists. DeepChem 2.8.0 has `dc.molnet.load_freesolv()` available in .venv. Also, `BalancingTransformer` is classification-only — must skip for regression.

### Q3: Uni-Mol pair_repr for FreeSolv?
**NO.** `unimol_out/pair_rep/smiles_unimol/` only has BACE data (indices 000388-000xxx). No Uni-Mol weights/inference for FreeSolv. **Decision**: Generate pair_repr from RDKit 3D conformers using Gaussian Basis Function (GBF) expansion to dim=64. This mimics Uni-Mol's pair representation while being self-contained.

### Q4: make_bace_from_unimol.py reusable?
**Partially.** Core helpers (build_all_pairs_edge_index, attach pair_repr to Data) are reusable. But it's hardcoded to BACE CSV/SDF/fold structure. Writing a new `prepare_freesolv_data.py` that reuses the graph-building logic.

## Execution Path: C (full build)

1. Create FreeSolv CSV from DeepChem
2. Build .pt datasets for both baseline (2D-only) and V2-T5 (with GBF pair_repr)
3. Write regression-capable run scripts (MSE pretrain loss doesn't apply — contrastive pretrain is task-agnostic; only downstream eval changes to MSE/RMSE)
4. Sanity check
5. Launch experiments

## Environment
- Python: `D:/SpecMol-Zip/.venv/Scripts/python.exe`
- PyTorch 1.13.1+cu117, PyG 2.3.1, RDKit, DeepChem 2.8.0
- GPU: NVIDIA RTX 3050 Laptop 4GB VRAM
- FreeSolv: ~642 molecules (very small, fast training)

## Files Created/Modified
- `prepare_freesolv_data.py` — download + process FreeSolv into .pt for baseline + V2
- `run_freesolv_baseline.py` — baseline (LH_Direct) with regression downstream
- `run_freesolv_v2t5.py` — V2-T5 (LH_Direct_V2) with regression downstream
- `sanity_check_freesolv.py` — verify data loading + forward/backward
- `launcher_freesolv.py` — serial subprocess launcher with timeout
- `freesolv_setup_report.md` — this file
- `freesolv_results_report.md` — final report (written after launch)

## Experiment Design: Reduced version (6 cells)
- 3 baseline cells (pretrain seeds 9, 19, 29) x 1 eval (scaffold split from data prep)
- 3 V2-T5 cells (pretrain seeds 9, 19, 29) x 1 eval
- Estimated time: ~1-2h total (FreeSolv is tiny, 642 mols)
