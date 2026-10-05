# PROJECT_STATE.md

Last updated: 2026-05-03

---

## Section 1: Task History

### T1: Code Understanding / Initial Analysis
- **Goal**: Understand SpecMol codebase structure, data pipeline, model architecture.
- **Status**: Complete
- **Key output**: CLAUDE.md, AGENTS.md
- **Key finding**: Project uses dual-path Chebyshev spectral GNN (high-pass + low-pass) with fingerprint MLP, contrastive pretrain, linear probe eval.

### T2: Baseline Cleanup
- **Goal**: Clean model_gnn_pre.py, ensure reproducible V0 baseline on BACE.
- **Status**: Complete
- **Key output**: `model_gnn_pre.py` (cleaned LH_Direct, ChebNetII, LogReg), `baseline_B_results.json`
- **Key finding**: Baseline V0 on BACE with original DeepChem ScaffoldSplitter: 0.797 mean AUC (3 pretrain seeds x 3 eval splits = 9 cells). This result was later partially invalidated — see T5.5.

### T3: Uni-Mol Data Pipeline
- **Goal**: Consume Uni-Mol scaffold fold CSV + SDF + pair_rep, rebuild BACE .pt files.
- **Status**: Complete
- **Key output**: `make_bace_from_unimol.py`, `down_task_v2/processed/bace_*.pt`
- **Key decision**: Uni-Mol fold split (k=10, seed=42, fold 0=test, fold 1=val). All-pairs edge_index stored as `pair_edge_index` alongside original chem bond `edge_index`.

### T4: PairToEdgeWeight Module
- **Goal**: Map all-pairs pair_repr to scalar edge weights on chem bond edges.
- **Status**: Complete
- **Key output**: `pair_to_edge_weight.py`
- **Key design**: MLP(pair_dim -> hidden -> 1), symmetrize (i,j)+(j,i) before sigmoid. Bias init +5 so sigmoid starts near 1 (equivalent to unweighted). Searchsorted-based lookup avoids O(N^2) table. 5 unit tests pass.

### T5: V2 Single-Direction Implementation (V2-T5)
- **Goal**: Implement static-pair weighted Laplacian ChebNet II. Pair_repr from Uni-Mol is used once to compute edge weights; no iterative update.
- **Status**: Complete
- **Key output**: `model_gnn_pre_v2.py` (LH_Direct_V2), `LH_Direct_ChebnetII_prop_v2.py` (ChebnetII_prop_V2), `run_v2_t5_bace.py`, `v2_t5_bace_results.json`
- **Key result**: BACE V2-T5 grand mean AUC = 0.8373 (9 cells). Uses `down_task_v2` (Uni-Mol fold split).
- **Key finding**: Bias moves from 5.0 to ~4.84, meaning sigmoid stays at ~0.992. The dynamic edge weight path barely activates.

### T5.5: Split Confound Discovery
- **Goal**: Discovered that T2 baseline used DeepChem ScaffoldSplitter while T5 V2-T5 used Uni-Mol fold split. Direct comparison invalid.
- **Status**: Complete (the problem was identified)
- **Key decision**: Must re-run baseline on same Uni-Mol fold split before any comparison.

### T5.5b: Strict Comparison (BACE, Uni-Mol fold split)
- **Goal**: Run Baseline V0 + FP-only on BACE with Uni-Mol fold split to enable fair comparison with V2-T5.
- **Status**: Complete
- **Key output**: `make_baseline_unifold.py`, `run_baseline_unifold.py`, `down_task_unifold/processed/bace_*.pt`, `baseline_unifold_results.json`
- **Key results**:
  - Baseline V0 (unifold): 0.7971 mean AUC, std 0.0846 (9 cells)
  - FP-only: 0.8459 mean AUC, std 0.0103 (9 cells)
- **Key finding**: FP-only (0.846) beats GNN baseline (0.797) on this split. V2-T5 (0.837) beats GNN baseline but not FP-only. The GNN component may be hurting rather than helping on BACE with this split.

### T6: NodeToPairUpdate (Bidirectional K-step Update)
- **Goal**: Iterative pair_repr update using ChebNet propagation states T_k, mirroring Uni-Mol's QK^T mechanism.
- **Status**: Code written, NOT integrated into main model path
- **Key output**: `pair_repr_init.py` (GBF init), `node_to_pair_update.py` (NodeToPairUpdate), `pair_to_node_gate.py` (abandoned per memory #22), `pair_update_block.py` (PairUpdateBlock)
- **What exists**: Standalone modules with unit tests. Never wired into LH_Direct_V2 or any run script.

### T7: V2 Integration into Main Pipeline
- **Goal**: Wire V2 into main_pretrain.py via --use_v2 flag for unified experiment management.
- **Status**: Partial — `main_pretrain.py` has `--use_v2` flag that imports LH_Direct_V2, but downstream eval is still classification-only (BCEWithLogitsLoss + AUC). The standalone `run_v2_t5_bace.py` was used for actual experiments instead.

### T8: Nullify Ablation
- **Goal**: Run V2-T5 with edge_weight forced to 1 (PairToEdgeWeight.nullify=True) to measure whether pair_repr architecture overhead itself helps, separate from the 3D signal.
- **Status**: Partial — seed_9 complete, seed_19 was in progress (pretrain epoch ~550) when killed by FreeSolv task. Seed_29 not started.
- **Key output**: `run_v2_t5_nullify_bace.py`, `launcher_nullify.py`, `v2_t5_nullify_bace_results.json`
- **Key result (seed_9 only)**: Nullify AUC = 0.8789 (mean of 3 eval splits). Higher than V2-T5 (0.837) and FP-only (0.846). Raises questions about whether the learned edge weights are helpful or harmful.
- **Caveat**: Single seed. Cannot draw conclusions without seeds 19 and 29.

### T9: Report for 学姐
- **Goal**: Prepare presentation-ready summary of all results.
- **Status**: Not done. Multiple drafts discussed but never finalized.

### FreeSolv Pilot (not in original T1-T9 roadmap)
- **Goal**: Extend SpecMol to FreeSolv regression task as second dataset validation.
- **Status**: Complete
- **Key output**: `prepare_freesolv_data.py`, `run_freesolv_baseline.py`, `run_freesolv_v2t5.py`, `launcher_freesolv.py`, `sanity_check_freesolv.py`, `freesolv_baseline_results.json`, `freesolv_v2t5_results.json`
- **Key changes**: Added regression downstream eval (MSE loss, RMSE metric). Generated pair_repr from RDKit 3D via GBF expansion (not Uni-Mol). Separate data dirs: `down_task_freesolv_2d/`, `down_task_freesolv_v2/`.
- **Key result**: V2-T5 RMSE 0.6746 vs Baseline RMSE 0.7604 (11.3% improvement, 3 seeds).

---

## Section 2: Key Numbers

All numbers verified against result JSON files on 2026-05-03.

### BACE (Classification, metric = ROC-AUC, higher is better)

| Variant | Split Protocol | n cells | Mean AUC | Std | JSON file | Note |
|---------|---------------|---------|----------|-----|-----------|------|
| Baseline V0 (old split) | DeepChem Scaffold | 9 | 0.7968 | 0.0389 | `baseline_B_results.json` | Invalidated: split mismatch with V2-T5 |
| Baseline V0 (unifold) | Uni-Mol k10 fold | 9 | 0.7971 | 0.0846 | `baseline_unifold_results.json` | Fair comparison baseline |
| FP-only | Uni-Mol k10 fold | 9 | 0.8459 | 0.0103 | `baseline_unifold_results.json` | No GNN, fingerprint MLP only |
| V2-T5 | Uni-Mol k10 fold | 9 | 0.8373 | 0.0094 | `v2_t5_bace_results.json` | Static pair-weighted Laplacian |
| V2-T5-nullify | Uni-Mol k10 fold | 3 | 0.8789 | - | `v2_t5_nullify_bace_results.json` | seed_9 only, 3 eval splits |

**Discrepancies from user-provided table**:
- User said "Baseline V0 (旧 split) 0.797 / 0.039" — JSON says 0.7968 / 0.0389. The summary field `all_9_test_aucs` contains only 3 values (bug in summary generation), but per-seed data has all 9 and recomputed mean is 0.7968. Rounding consistent.
- User said "V2-T5-nullify 0.879" — JSON says 0.8789 (seed_9 mean across 3 eval splits). Consistent.
- All other numbers match JSON within rounding.

### FreeSolv (Regression, metric = RMSE, lower is better)

| Variant | Split Protocol | n cells | Mean RMSE | Std | JSON file | Note |
|---------|---------------|---------|-----------|-----|-----------|------|
| Baseline | DeepChem Scaffold | 3 | 0.7604 | 0.0474 | `freesolv_baseline_results.json` | Single split, 3 pretrain seeds |
| V2-T5 (GBF 3D) | DeepChem Scaffold | 3 | 0.6746 | 0.0325 | `freesolv_v2t5_results.json` | RDKit 3D + GBF, not Uni-Mol pair_repr |

---

## Section 3: Code File Inventory

### Model files

| File | Function | Used by |
|------|----------|---------|
| `model_gnn_pre.py` | LH_Direct (baseline V0), ChebNetII, LogReg, MLP | Baseline runs, downstream eval for all variants |
| `model_gnn_pre_v2.py` | LH_Direct_V2 (V2 with PairToEdgeWeight) | V2-T5 runs (BACE + FreeSolv) |
| `LH_Direct_ChebnetII_prop.py` | ChebnetII_prop — baseline spectral propagation layer | model_gnn_pre.py |
| `LH_Direct_ChebnetII_prop_v2.py` | ChebnetII_prop_V2 — V2 propagation (requires 1-D edge_weight, no fallback) | model_gnn_pre_v2.py |
| `pair_to_edge_weight.py` | PairToEdgeWeight (T4 module: pair_repr -> scalar edge weight per chem bond) | model_gnn_pre_v2.py |
| `pair_repr_init.py` | GaussianBasisProjection (GBF distance init for pair_repr) | T6 modules only, not in main path |
| `node_to_pair_update.py` | NodeToPairUpdate (T6: Q/K projection of T_k states for pair update) | Not used in any run script |
| `pair_to_node_gate.py` | PairToNodeGate (abandoned, memory #22 says replaced by simpler approach) | Not used |
| `pair_update_block.py` | PairUpdateBlock (multi-head attention-style pair update) | Not used in V2-T5; designed for V2-stable |

### Data pipeline files

| File | Function | Used by |
|------|----------|---------|
| `utils_fp_downstream.py` | TestbedDataset, CombinedFingerprintsFeaturizer, baseline data loading | Baseline runs, also imported by other data builders |
| `create_data_DC.py` | `smile_to_graph()` — SMILES to PyG graph conversion | All data pipelines |
| `make_bace_from_unimol.py` | Build BACE .pt from Uni-Mol fold CSV + pair_rep .pt files | V2-T5 BACE data (`down_task_v2`) |
| `make_baseline_unifold.py` | Build baseline BACE .pt using Uni-Mol fold split (no pair_repr) | Unifold baseline data (`down_task_unifold`) |
| `prepare_freesolv_data.py` | Download FreeSolv, scaffold split, build .pt for baseline + V2 (GBF pair_repr) | FreeSolv data (`down_task_freesolv_2d`, `down_task_freesolv_v2`) |

### Run scripts

| File | Function | Status |
|------|----------|--------|
| `main_pretrain.py` | Original pretrain + eval script, classification only. Has `--use_v2` flag. | Used for T2 baseline. Not used for V2-T5 experiments. |
| `main_pretrain_seeds.py` | Multi-seed wrapper around main_pretrain.py | Used for early baseline runs |
| `run_v2_t5_bace.py` | V2-T5 on BACE (3 pretrain seeds x 3 eval splits) | Complete, results in `v2_t5_bace_results.json` |
| `run_v2_t5_nullify_bace.py` | V2-T5-nullify ablation on BACE | Partial (seed_9 done, seed_19 interrupted, seed_29 not started) |
| `run_baseline_unifold.py` | Baseline V0 + FP-only on BACE with Uni-Mol fold split | Complete, results in `baseline_unifold_results.json` |
| `run_freesolv_baseline.py` | FreeSolv baseline (2D-only, regression) | Complete, results in `freesolv_baseline_results.json` |
| `run_freesolv_v2t5.py` | FreeSolv V2-T5 (GBF pair_repr, regression) | Complete, results in `freesolv_v2t5_results.json` |
| `launcher_nullify.py` | Subprocess launcher for nullify seeds 19/29 | Ran, seed_19 killed mid-pretrain |
| `launcher_freesolv.py` | Subprocess launcher for all 6 FreeSolv cells | Complete |

### Other relevant files

| File | Function |
|------|----------|
| `nt_xent.py` | NT-Xent loss (not used; `ours_loss` in run scripts is used instead) |
| `encoder_gnn.py` | GATNet, GINNet (legacy encoders, not used in current experiments) |
| `pubchemfp.py` | PubChem fingerprint generation |
| `csv_dealer.py` | CSV utilities |
| `smoke_test.py` | General smoke test |
| `sanity_check_freesolv.py` | FreeSolv-specific sanity check (forward/backward for both variants) |
| `check_unimol_outputs.py` | Inspect Uni-Mol output files |
| `export_unimol_splits.py` | Export Uni-Mol split info |
| `tools/extract_unimol_pair.py` | Extract pair_rep from Uni-Mol inference outputs |
| `tools/smoke_test_pair_pipeline.py` | Smoke test for pair_rep pipeline |
| `scripts/run_bace_ablation.py` | Multi-variant BACE ablation runner |
| `scripts/run_hiv_split_ablation.py` | HIV split ablation (different dataset) |
| `scripts/run_bace_bondpair_stability.py` | Bondpair stability check |
| `scripts/run_mock_spectrum_pairlearn.py` | Mock spectrum experiment |
| `make_bace_ablation.py` | Build data for BACE ablation variants |
| `prepare_molnet_dataset.py` | General MoleculeNet dataset prep |
| `make_spectrum_pair_dataset.py` | Spectrum pair dataset builder |
| `convert_spectra_to_csv.py` | Spectra format conversion |
| `run_spectrum_pipeline.py` | End-to-end spectrum pipeline |

### Data directories

| Directory | Contents | Used by |
|-----------|----------|---------|
| `down_task/processed/` | BACE .pt (original DeepChem split) | T2 baseline (invalidated for V2 comparison) |
| `down_task_2d/processed/` | BACE .pt (2D-only, bond edges) | V0 ablation |
| `down_task_v2/processed/` | BACE .pt (Uni-Mol fold, with pair_repr_edge) | V2-T5, V2-T5-nullify |
| `down_task_unifold/processed/` | BACE .pt (Uni-Mol fold, no pair_repr) | Unifold baseline |
| `down_task_unimol/processed/` | BACE .pt (static Uni-Mol edge weights) | V1 (not run in current experiments) |
| `down_task_unimolupd/processed/` | BACE .pt (Uni-Mol pair update) | V2-stable (not run) |
| `down_task_ctrl/processed/` | BACE .pt (all-pairs, edge_weight=1) | V0-ctrl (not run) |
| `down_task_bondpair/processed/` | BACE .pt (bond+pair edges) | V1-bondpair (not run) |
| `down_task_freesolv_2d/processed/` | FreeSolv .pt (2D-only) | FreeSolv baseline |
| `down_task_freesolv_v2/processed/` | FreeSolv .pt (with GBF pair_repr) | FreeSolv V2-T5 |
| `dataset/freesolv/raw/` | FreeSolv CSV (642 mols) | prepare_freesolv_data.py |
| `unimol_out/` | Uni-Mol outputs (pair_rep, splits, sdf) | BACE only |

---

## Section 4: Known Caveats / Unresolved Issues

1. **BACE V2-T5-nullify incomplete**: Only seed_9 finished (AUC 0.8789). Seed_19 was mid-pretrain (~epoch 550) when killed. Seed_29 never started. Cannot draw reliable conclusions from 1 seed.

2. **Nullify result is suspiciously high**: seed_9 nullify (0.8789) > V2-T5 (0.8373) > FP-only (0.8459). If this holds across seeds, it suggests the learned edge weights in V2-T5 are actively hurting. But 1 seed is insufficient.

3. **Baseline AUC coincidence**: Baseline V0 on old split (0.7968) and on Uni-Mol fold split (0.7971) are nearly identical by coincidence. This caused delayed detection of the split confound.

4. **FP-only dominance on BACE**: FP-only (0.8459) beats GNN baseline (0.7971) on BACE with unifold split. This means the GNN component is not adding value on this particular dataset/split. V2-T5 (0.8373) is between FP-only and baseline, suggesting pair_repr helps the GNN but the GNN+FP ensemble still underperforms FP alone.

5. **FreeSolv pair_repr is GBF, not Uni-Mol**: FreeSolv V2-T5 uses RDKit 3D conformer distances expanded via 64-dim Gaussian Basis Functions. This is a different 3D source than BACE's Uni-Mol encoder pair_rep. Results are not directly comparable across datasets in terms of "Uni-Mol 3D benefit".

6. **FreeSolv RMSE context**: FreeSolv V2-T5 RMSE 0.6746 kcal/mol on a label range of [-5.64, 1.88]. No external benchmark comparison done. The user noted Uni-Mol paper reports 1.48 RMSE on FreeSolv but conditions (split, metric variant, label preprocessing) are unverified.

7. **Bias stays in saturation zone**: On both BACE and FreeSolv, PairToEdgeWeight bias moves from 5.0 to ~4.7-4.8 (sigmoid 0.993 -> 0.991). The edge weights are barely differentiated from 1.0, meaning the "dynamic" path is almost inactive. The model improvement may come from the architectural overhead (extra parameters, different gradient flow) rather than meaningful 3D signal utilization.

8. **学姐 split typo**: A known `beta_b_l` typo in the original split code has been noted but never fixed.

9. **T6 bidirectional update not tested**: NodeToPairUpdate, PairUpdateBlock, PairReprInit modules exist as standalone code with unit tests but are not wired into any model or run script. No experimental results for iterative pair_repr update.

10. **T9 report not delivered**: Multiple drafts discussed, never finalized or sent.

11. **Single scaffold split for FreeSolv**: Only one scaffold split used (DeepChem default). No eval_seeds rotation like BACE. Small test set (65 molecules) makes results statistically fragile.

12. **V1 (static Uni-Mol edge weights) never formally benchmarked**: `down_task_unimol/` data exists but no completed V1 experiment with the same split protocol as V2-T5/unifold baseline.

13. **`baseline_B_results.json` summary bug**: The `all_9_test_aucs` array contains only 3 values (one per pretrain seed for split_9 only), not the actual 9. The `grand_mean` and `grand_std` appear to be computed correctly over all 9 cells despite the truncated array.
