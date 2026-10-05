#!/usr/bin/env bash
###############################################################################
# Submit all FreeSolv training jobs (baseline + V2-T5) × 3 seeds via sbatch.
#
# Prerequisites: post-B4 data already generated:
#   down_task_freesolv_2d/processed/*.pt
#   down_task_freesolv_v2/processed/*.pt
# (run `python prepare_freesolv_data.py` inside an srun shell first if missing)
#
# Usage:
#   bash hpc/submit_freesolv_all.sh [VARIANTS]
#
# Args:
#   VARIANTS : space-separated, default "baseline v2t5"
#
# Env override:
#   PARTITION, GRES, CPUS, MEM, TIME, ACCOUNT, QOS, SEEDS
###############################################################################
set -euo pipefail

VARIANTS="${1:-baseline v2t5}"

PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-40G}"
TIME="${TIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
SEEDS="${SEEDS:-9 19 29}"

LOG_DIR="hpc/logs"
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
  case "$variant" in
    baseline) sbatch_file="hpc/run_freesolv_baseline.sbatch"; tag="freesolv_v0" ;;
    v2t5)     sbatch_file="hpc/run_freesolv_v2t5.sbatch";     tag="freesolv_v2t5" ;;
    *) echo "Unknown variant: $variant (expected: baseline | v2t5)" >&2; exit 1 ;;
  esac

  for seed in $SEEDS; do
    job_name="${tag}_seed${seed}"
    results_path="hpc/results/${tag}_seed${seed}.json"
    echo "[submit] variant=$variant seed=$seed job=$job_name"
    RESULTS_PATH="$results_path" sbatch \
      "${common_args[@]}" \
      "--job-name=$job_name" \
      "--output=${LOG_DIR}/${job_name}_%j.out" \
      "--error=${LOG_DIR}/${job_name}_%j.err" \
      "$sbatch_file" "$seed"
  done
done

n_var=$(echo $VARIANTS | wc -w)
n_seed=$(echo $SEEDS | wc -w)
echo ""
echo "Submitted $n_var variants × $n_seed seeds = $(( n_var * n_seed )) jobs for freesolv"
echo "Monitor: squeue -u \$USER"
echo "Aggregate after all done: python paper/aggregate_freesolv_seeds.py"
