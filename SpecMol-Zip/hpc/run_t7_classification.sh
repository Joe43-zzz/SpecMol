#!/usr/bin/env bash
# Re-run BBBP / BACE T7 on HPC so they share the SAME environment as the
# canonical V0 / V2-T5 runs (conda specmol, RTX 5000 Ada), fixing the
# local-vs-HPC provenance inconsistency for the T7 classification cells.
#
# Matched protocol: this calls the same sbatch as the canonical runs, so
# EVAL_EPOCHS (500), batch (512), K (10), hid_dim (512), eval_seeds (9,19,29)
# all inherit the V0/V2-T5 defaults — do NOT override them or the comparison
# stops being matched.
#
# Prereq: VPN up, `ssh mbzuai-hpc`, repo pulled, run from repo root.
#
# Usage:
#   bash hpc/run_t7_classification.sh
# Override task/seed list:
#   TASKS="bbbp"  SEEDS="9 19 29"  bash hpc/run_t7_classification.sh
#
# After ALL jobs finish (watch with: squeue -u $USER), aggregate + then
# update the paper's T7 table row from the new same-env numbers.

set -euo pipefail

TASKS="${TASKS:-bbbp bace}"
SEEDS="${SEEDS:-9 19 29}"
SBATCH_SCRIPT="hpc/run_dataset_training.sbatch"

[ -f "$SBATCH_SCRIPT" ] || { echo "ERROR: run from the repo root (missing $SBATCH_SCRIPT)"; exit 1; }
command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: 'sbatch' not found — are you on the HPC login node? (VPN up, then ssh mbzuai-hpc)"; exit 1; }

submitted=0
skipped=""
for TASK in $TASKS; do
  DATA="down_task_${TASK}_v2"
  if [ ! -d "$DATA/processed" ]; then
    echo "SKIP $TASK: T7 needs pair_repr data at '$DATA/processed', which is missing."
    if [ "$TASK" = "bace" ]; then
      echo "      NOTE: BACE v2 data may still be the pre-B4 archive in down_task_v2/;"
      echo "      regenerate it into down_task_bace_v2/ before submitting BACE T7."
    fi
    skipped="$skipped $TASK"
    continue
  fi
  for SEED in $SEEDS; do
    echo "submit -> TASK=$TASK VARIANT=t7 SEED=$SEED"
    sbatch "$SBATCH_SCRIPT" "$TASK" t7 "$SEED"
    submitted=$((submitted + 1))
  done
done

echo
echo "==================================================================="
echo "Submitted $submitted T7 job(s).${skipped:+  Skipped (missing data):$skipped}"
echo "Watch:   squeue -u \$USER"
echo
echo "When all finish, aggregate the new same-env T7 results:"
for TASK in $TASKS; do
  echo "  python hpc/collect_results.py --task $TASK --output hpc/results/${TASK}_all_results.json"
done
echo
echo "Then verify the collector merged the t7 variant, and update the T7 row"
echo "in paper/tables/classification.tex from the new HPC numbers (the local"
echo "*_t7_bare_results.json values can then be retired)."
echo "==================================================================="
