#!/usr/bin/env bash
###############################################################################
# Submit all training jobs for a given task via sbatch.
#
# Prerequisites: data prep done (TASK=<task> bash hpc/run_dataset_pipeline.sh prep)
#
# Usage:
#   bash hpc/submit_dataset_all.sh TASK [VARIANTS]
#
# Args:
#   TASK     : bace | bbbp | hiv | clintox | tox21
#   VARIANTS : space-separated, default "v0 v2 t6" (skip t6safe* unless needed)
#
# Env override:
#   PARTITION, GRES, CPUS, MEM, TIME, ACCOUNT, QOS, SEEDS
###############################################################################
set -euo pipefail

TASK="${1:?Usage: bash hpc/submit_dataset_all.sh TASK [VARIANTS]}"
VARIANTS="${2:-v0 v2 t6}"

PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-40G}"
TIME="${TIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
SEEDS="${SEEDS:-9 19 29}"

LOG_DIR="hpc/logs/${TASK}"
mkdir -p "$LOG_DIR" hpc/results

common_args=(
  "--partition=$PARTITION"
  "--gres=$GRES"
  "--cpus-per-task=$CPUS"
  "--mem=$MEM"
  "--time=$TIME"
)

if [ -n "$ACCOUNT" ]; then
  common_args+=("--account=$ACCOUNT")
fi
if [ -n "$QOS" ]; then
  common_args+=("--qos=$QOS")
fi

for variant in $VARIANTS; do
  for seed in $SEEDS; do
    job_name="${TASK}_${variant}_seed${seed}"
    echo "[submit] task=$TASK variant=$variant seed=$seed job=$job_name"
    sbatch \
      "${common_args[@]}" \
      "--job-name=$job_name" \
      "--output=${LOG_DIR}/${variant}_seed${seed}_%j.out" \
      "--error=${LOG_DIR}/${variant}_seed${seed}_%j.err" \
      hpc/run_dataset_training.sbatch "$TASK" "$variant" "$seed"
  done
done

n_var=$(echo $VARIANTS | wc -w)
n_seed=$(echo $SEEDS | wc -w)
echo ""
echo "Submitted $n_var variants x $n_seed seeds = $(( n_var * n_seed )) jobs for task=$TASK"
echo "Monitor: squeue -u \$USER"
echo "Collect: python hpc/collect_results.py --task $TASK --log-dir $LOG_DIR --output hpc/results/${TASK}_all_results.json"
