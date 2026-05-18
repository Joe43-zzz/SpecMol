# SpecMol Rescue State

Last updated: 2026-05-18 (overnight session)

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

## 2026-05-18 update: BACE narrative converged on "FP-saturated, not rescuable"

The overnight session closed three loose ends and the BACE chapter of
the paper is now defensible:

1. **RF on the matched Uni-Mol fold split** (`baselines_ml_results_unimol_fold.json`,
   30-seed Deng2023 protocol in `baselines_ml_results_deng30_unimol_fold.json`):
   BACE RF = 0.895 ± 0.003. Strictly beats deep FP-only (0.846), V2-T5 (0.837),
   and Chemprop (0.840). Confirms Deng et al. 2023 finding on the
   exact split the paper uses, removing the apples-to-oranges issue.

2. **V2-T5 saturation hypothesis refuted** (`mlp_phi_stats/bace_v2t5_seed{9,19,29}
   _b5_audit.json`): the bias term drifts only 5.0→4.77, but the MLP weights
   learn to shift the pre-sigmoid scalar by ~-2.6, so the trained per-bond
   gate distribution concentrates at sigma≈0.07. V2-T5 IS learning, just not
   what the original "near-identity preservation" framing claimed; the trained
   encoder converges to an aggressively sparsified Laplacian. Paper paragraphs
   corrected (commits eae9a426, 863fc2e0, b06793cd) and a figure embedded.

3. **Chemprop D-MPNN baseline added** (`baselines_chemprop_results_unimol_fold.json`):
   on BBBP and FreeSolv V2-T5 strictly beats Chemprop (+4.7 AUC on BBBP, -0.20
   RMSE on FreeSolv). On BACE Chemprop matches the V2/FP-only band but does
   not reach RF. Strengthens "BACE is FP-saturated" framing without weakening
   the positive V2 story.

The in-flight HPC bias_init=+1 sweep (jobs 135764-135766) is no longer
expected to change the bottom-line conclusion — the saturation refutation
already shows the MLP can shift the gate freely regardless of bias init.
The sweep result is still useful as a "does the optimization path differ"
sanity check but should not block paper writing.

Multi-label baseline infrastructure (ClinTox + Tox21) is now wired into
`baselines_ml.py` and `baselines_chemprop.py`. RF 3-seed smoke:
ClinTox macro_roc_auc 0.818 ± 0.021, Tox21 0.716 ± 0.006. 30-seed run in flight.

The remaining blockers (B3a/B4a, A4) all touch deep-model pipelines that
need HPC pair_extract + a multi-task `export_unimol_splits.py` refactor.
