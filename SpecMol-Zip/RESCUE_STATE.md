# SpecMol Rescue State

Last updated: 2026-05-14

This file is the current short-form source of truth for the post-B4 rescue
work. `PROJECT_STATE.md` is historical and should not be used as the current
experiment state without checking this file first.

## Current Priority

Focus on BACE post-B4 first. Do not start new T7/T8/E1, HIV, or FreeSolv work
until the canonical BACE table is rebuilt.

Canonical BACE outputs must use:

- `down_task_bace_v2/processed/` for V2-T5 and T6 data.
- `down_task_bace_unifold/processed/` for V0 data on the same Uni-Mol fold split.
- `hpc/results/bace_all_results.json` as the collected result file.

Legacy BACE roots such as `down_task_v2/`, `down_task_unifold/`, `down_task_2d/`,
and `down_task_bondpair/` are pre-B4 archives and must not be used for new
comparisons.

## Timeline

- 2026-04-28 to 2026-04-29: split confound discovered. Old DeepChem-split V0 and
  Uni-Mol-fold V2-T5 cannot be compared directly.
- 2026-05-03: `PROJECT_STATE.md` captured the old state. It predates B4 and is
  now archival.
- 2026-05-13: B3/B4 fixes landed.
  - B3: `PairToEdgeWeight` handles self-loops at runtime.
  - B4: `create_data_DC.py` emits bidirectional chemical bond edges. This changes
    processed `.pt` graph structure, so old processed BACE graph-branch results
    are not canonical.
- 2026-05-14: T7/T8/E1 became exploratory side tracks. They are paused until the
  BACE post-B4 V0/V2-T5/T6 table exists.

## Trust Status

Usable as provisional controls:

- BACE V0 and FP-only Uni-Mol fold results in `baseline_unifold_results.json`.

Plausible but needs log audit:

- BBBP post-B4 summary in `bbbp_all_results.json`.

Not canonical:

- `v2_t5_bace_results_DEPRECATED_pre_B4.json`
- `t6_bace_results_DEPRECATED_pre_B4.json`
- `v2_t5_nullify_bace_results_DEPRECATED_pre_B4.json`
- `freesolv_baseline_results_DEPRECATED_pre_B4.json`
- `freesolv_v2t5_results_DEPRECATED_pre_B4.json`
- Any T7/T8/E1 numbers that exist only in plans or logs without a canonical JSON.

## Required Run Commands

Run only inside a Slurm allocation or via `sbatch`/`srun` on HPC:

```bash
echo $CUDA_VISIBLE_DEVICES
squeue -u $USER
nvidia-smi
TASK=bace bash hpc/run_dataset_pipeline.sh prep
bash hpc/submit_dataset_all.sh bace "v0 v2 t6"
python hpc/collect_results.py --task bace --log-dir hpc/logs/bace --output hpc/results/bace_all_results.json
```

Optional BBBP audit recovery:

```bash
python hpc/collect_results.py --task bbbp --log-dir hpc/logs/bbbp --output hpc/results/bbbp_all_results.json
```

## Decision Rule

After `hpc/results/bace_all_results.json` exists:

- If V2-T5 or T6 clearly beats V0 and is competitive with FP-only, continue with
  conservative ablations.
- If neither beats V0/FP-only, write BACE as a control-heavy or negative case and
  do not keep trying to rescue BACE with new architecture variants.
