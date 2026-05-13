# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This project evaluates whether Uni-Mol 3D molecular information can improve SpecMol-style molecular property prediction. Three main experiment variants are compared:

- **V0**: 2D-only baseline (pure bond graph)
- **V1**: 2D + static Uni-Mol 3D edge weights
- **V2 / V2-stable**: 2D + learnable Uni-Mol pair update mechanism

**Current focus**: BACE (post-B4 re-run) + BBBP (done) + HIV (onboarding). Regression (FreeSolv) and multi-label (ClinTox/Tox21) deferred to a later phase.

Pipeline entry points after Stage 0/1 cleanup (2026-05-13):
- Task-agnostic orchestration: `TASK=<bace|bbbp|hiv> bash hpc/run_dataset_pipeline.sh [phase]`
- Multi-job submit: `bash hpc/submit_dataset_all.sh <task>`
- Unified collector: `python hpc/collect_results.py --task <task> --output hpc/results/<task>_all_results.json`
- Data dir convention: `down_task_<task>_v2/` (V2-T5/T6 with pair_repr) and `down_task_<task>_unifold/` (V0 on Uni-Mol fold split). BACE legacy `down_task_v2/` is the pre-B4 archive.

## Environment

System Python 3.12 (no conda needed):
```
C:\Users\zhoutianyang\AppData\Local\Programs\Python\Python312\python.exe
```
Key packages: PyTorch 2.11.0 (CPU), PyG 2.7.0, scipy, numpy, scikit-learn.

All code runs from inside `SpecMol-Zip/` (the inner directory).

## Common Commands

### Run single-seed pretraining + evaluation
```bash
python main_pretrain.py --task bace --path down_task --batch_size 512 --epochs 1000 --gpu 0 --hid_dim 512 --K 10 --random_seed 9
```

### Run multi-seed aggregated pretraining
```bash
python main_pretrain_seeds.py --task bace --path down_task --batch_size 512 --epochs 1000 --gpu 0
```

### Run BACE ablation (multiple variants)
```bash
python scripts/run_bace_ablation.py --task bace --data_root down_task --variants "V0,V1,V2-stable" --seeds "9,19,29" --batch_size 256 --epochs 1000 --logreg_epochs 2000 --gpu 0
```

### Run HIV split ablation
```bash
python scripts/run_hiv_split_ablation.py --data-root molecular_benchmarks/hiv --variants "V0,V0-ctrl" --seeds "9,19,29" --batch-size 512 --gpu 0
```

### End-to-end spectrum pipeline
```bash
python run_spectrum_pipeline.py --input-path /path/to/data.mgf --label-csv /path/to/labels.csv --output-root ./spectrum_output --task my_task
```

## Architecture

### Model: `LH_Direct` (`model_gnn_pre.py`)

The core model combines dual-path Chebyshev spectral GNN with a fingerprint MLP:

1. **ChebnetII encoder** (`LH_Direct_ChebnetII_prop.py`): Two parallel spectral convolution paths — high-pass and low-pass — using Chebyshev polynomial filters with learnable coefficients. Optional Uni-Mol pair weights injected as edge weights.
2. **PairUpdateBlock** (`pair_update_block.py`): Multi-head attention-style mechanism that iteratively refines pair representations from Uni-Mol (used in V0-pairlearn and V2-stable only).
3. **Fingerprint MLP**: Separate 3-layer FC network on molecular fingerprints (PubChem+MACCS+ErG, 1489 dims by default).
4. **Downstream head**: `LogReg` trained on concatenated `[h_gnn, h_fp]` frozen representations.

### Training Pipeline

**Pretraining** (`main_pretrain.py`): Contrastive learning (`ours_loss`) across 4 views — low-pass GNN, high-pass GNN, mixed GNN, fingerprint MLP — using NT-Xent loss with early stopping (patience=100).

**Downstream evaluation**: Frozen encoder + trained `LogReg` head, evaluated per seed; reports best test AUC. Standard seeds: `[9, 19, 29]` for benchmarks, `[9, 19, 29, 39, 49]` for broader reporting.

### Data Pipeline (`utils_fp_downstream.py`, `create_data_DC.py`)

Raw SMILES CSV → RDKit featurization (93-dim atom features, 11-dim bond features) → `TestbedDataset` (PyG `InMemoryDataset`) → scaffold split (80/10/10) → `.pt` files saved under `down_task_*/processed/`.

Each variant uses a different `down_task_*` directory because the processed `.pt` files differ in graph topology (e.g., all-pairs vs. bond-only edges) or edge attributes.

| Variant | Data root | Pair Update | Uni-Mol 3D |
|---------|-----------|-------------|------------|
| V0 | `down_task_2d` | No | No |
| V0-pairlearn | `down_task_2d` | Yes (3 heads) | No |
| V0-ctrl | `down_task_ctrl` | No | No (edge_weight=1) |
| V1-bondpair | `down_task_bondpair` | Yes | Yes |
| V1 | `down_task_unimol` | No | Yes (static) |
| V2-stable | `down_task_unimolupd` | Yes (1 head) | Yes |

Uni-Mol pair weights: `edge_weight_3d = softplus(mean_over_last_dim(encoder_pair_rep[i,j,:]))`.

## Critical Rules (from AGENTS.md)

### Reproducibility / Fair Comparison
- All V0/V1/V2 comparisons **must** use the same dataset, split protocol, seed list, metric, training budget, and reporting format. Never compare results from different split pipelines directly.
- This project is **highly sensitive to split inconsistency**. Before any change, identify which split protocol applies: original baseline CSV pipeline or Uni-Mol scaffold fold pipeline. If unsure, inspect first and explain the ambiguity — do not guess.

### 3D Integration
- Do not assume 3D signal automatically improves performance.
- Do not silently alter graph topology (edge set changes must be explicit and justified).
- If using `edge_weight_3d`, document: how distance is computed, normalization, bond type scaling, and fallback for missing 3D conformers.

### V2 Update Mechanism
Before implementing any update mechanism variant, clarify: what object is updated, when, whether it is one-shot or iterative, how fairness vs. V0/V1 is preserved, and what ablation is needed. Propose a design before coding.

### Code Organization
Do not silently move logic across files. File responsibilities are fixed:
- `utils_fp_downstream.py` — baseline dataset processing, fingerprint construction, dataset loading
- `make_bace_from_unimol.py` — Uni-Mol scaffold fold consumption, `.pt` rebuild, `edge_weight_3d` injection
- `model_gnn_pre.py` — `LH_Direct`, `ChebNetII`, `LogReg` definitions
- `main_pretrain.py` / `main_pretrain_seeds.py` — pretraining, downstream evaluation, seed reporting

### Per-Task Response Format
For every nontrivial task, respond with: Goal → Files inspected → Assumptions → Changes made → Fairness/reproducibility check → Exact run commands → Expected outputs → Risks or unresolved issues.

## Git Style

Small, scoped commits. Examples:
- `feat: add unimol-based bace split rebuild`
- `fix: align downstream eval with unimol fold protocol`
- `refactor: separate baseline and unimol dataset entrypoints`
- `docs: add experiment protocol for V0/V1/V2`

Do not bundle unrelated changes. Never silently change split protocol, metrics, or seed lists.

## HPC / Slurm Rules

When running on MBZUAI HPC or any Slurm-managed GPU server:

- GPU training must run through `sbatch` or `srun`; do not run GPU jobs directly on login nodes.
- Do not manually set or widen `CUDA_VISIBLE_DEVICES`; Slurm owns GPU visibility.
- Training commands should pass `--gpu 0` and treat `cuda:0` as the Slurm-visible allocated GPU.
- Do not hard-code physical GPU IDs such as `cuda:1`, `cuda:2`, etc.
- Before launching training, check `echo $CUDA_VISIBLE_DEVICES`, `squeue -u $USER`, and, when needed, `nvidia-smi`.
- Never use another user's GPU allocation; this can risk account suspension.
