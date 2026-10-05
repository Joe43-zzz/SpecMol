# SpecMol Project Instructions for Codex

## Project goal
This project evaluates whether Uni-Mol 3D information can improve SpecMol-style molecular property prediction.

The current research focus is not general code cleanup. The focus is:
1. define fair experiment variants,
2. keep data split protocol consistent,
3. implement only changes that preserve comparability,
4. produce runnable and reviewable outputs.

## Current experiment priority
Work on BACE first.

We currently care about three experiment variants:

- V0: 2D-only baseline
- V1: 2D + static Uni-Mol 3D edge weights
- V2: 2D + Uni-Mol update mechanism (optional extension, only after V0/V1 are stable)

Do not expand to more datasets unless explicitly asked.

## Ground truth code structure
Treat the current project as having these responsibilities:

- `utils_fp_downstream.py`
  - baseline dataset processing
  - fingerprint feature construction
  - default train/valid/test/all loading pipeline

- `make_bace_from_unimol.py`
  - consumes Uni-Mol exported scaffold folds and SDF
  - rebuilds BACE processed `.pt` files
  - injects `edge_weight_3d`

- `model_gnn_pre.py`
  - defines `LH_Direct`, `ChebNetII`, `LogReg`
  - consumes `edge_weight_3d` if present

- `main_pretrain.py` / `main_pretrain_seeds.py`
  - pretraining
  - downstream linear evaluation
  - seed-based reporting

Do not silently move logic across files unless refactor is explicitly requested.

## Fair comparison rules
All comparisons between V0, V1, and V2 must satisfy:

- same dataset
- same split protocol
- same seed list
- same evaluation metric
- same training budget unless explicitly justified
- same reporting format

Never compare results from different split pipelines as if they were directly comparable.

## Split protocol rules
This project is highly sensitive to split inconsistency.

Before making any change, determine which split protocol is the intended one:
- original baseline split from raw CSV pipeline
- Uni-Mol scaffold fold pipeline

If the experiment is a direct comparison across variants, all variants must use the same split protocol.

If unsure, do not guess. Inspect the current files and explain the ambiguity first.

## 3D integration rules
When handling Uni-Mol features:

- do not assume any 3D signal should automatically improve performance
- do not leak test information into preprocessing
- do not silently alter graph topology unless explicitly requested
- if using `edge_weight_3d`, document exactly:
  - how distance is computed
  - how it is normalized
  - how bond type scaling is applied
  - where fallback behavior is used when 3D conformers are missing

## Update-mechanism rules
For V2-style update mechanisms, do not implement directly before clarifying:

- what object is updated
- when update happens
- whether update is one-shot or iterative
- how fairness vs V0/V1 is preserved
- what ablation is needed

If these are unclear, propose a concrete design first before coding.

## Output format for every task
For every nontrivial task, respond with:

1. Goal
2. Files inspected
3. Assumptions
4. Changes made
5. Fairness / reproducibility check
6. Exact run commands
7. Expected outputs
8. Risks or unresolved issues

## HPC GPU safety rules
When running jobs on the MBZUAI HPC or any Slurm-managed GPU server:

- use only the GPU resources allocated by the active Slurm job
- do not manually override `CUDA_VISIBLE_DEVICES` to expose GPUs outside the allocation
- do not hard-code GPU IDs such as `cuda:1`, `cuda:2`, etc. unless they are relative to the Slurm-visible device set
- prefer `cuda`, `cuda:0`, or framework defaults after Slurm has set `CUDA_VISIBLE_DEVICES`
- before launching training, check `echo $CUDA_VISIBLE_DEVICES`, `squeue -u $USER`, and, when needed, `nvidia-smi`
- if no Slurm allocation is active, do not run GPU training on shared login nodes or unallocated GPUs
- never use another user's GPU allocation; violating this can risk account suspension

## Git discipline
Use small, scoped commits.
Recommended commit style:

- `feat: add unimol-based bace split rebuild`
- `fix: align downstream eval with unimol fold protocol`
- `refactor: separate baseline and unimol dataset entrypoints`
- `docs: add experiment protocol for V0/V1/V2`

Do not bundle unrelated changes in one commit.

## What not to do
Do not:

- silently change split protocol
- silently change metrics
- mix results from different seeds without saying so
- overwrite baseline behavior without documenting it
- “make it run” by removing key experiment constraints
- claim conclusions that are not supported by the current experiment output

## Preferred working style
For complex changes:
- inspect first
- propose a plan
- then patch code

For small fixes:
- patch directly
- still explain what changed and why

Always optimize for reproducibility over cleverness.
