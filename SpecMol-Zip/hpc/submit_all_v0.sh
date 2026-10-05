#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-48G}"
TIME_CLASS="${TIME_CLASS:-04:00:00}"
TIME_HIV="${TIME_HIV:-04:00:00}"
TIME_FREESOLV="${TIME_FREESOLV:-04:00:00}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
SEEDS="${SEEDS:-9 19 29}"
TASKS="${TASKS:-bace bbbp hiv clintox tox21}"

mkdir -p hpc/logs hpc/results hpc/slurm

common_args=(
  "--partition=$PARTITION"
  "--gres=$GRES"
  "--cpus-per-task=$CPUS"
  "--mem=$MEM"
)

if [ -n "$ACCOUNT" ]; then
  common_args+=("--account=$ACCOUNT")
fi

if [ -n "$QOS" ]; then
  common_args+=("--qos=$QOS")
fi

submit_classification() {
  local task="$1"
  local seed="$2"
  local time_limit="$TIME_CLASS"
  if [ "$task" = "hiv" ]; then
    time_limit="$TIME_HIV"
  fi

  local job_name="v0_${task}_seed${seed}"
  echo "[submit] classification task=$task seed=$seed time=$time_limit"
  sbatch \
    "${common_args[@]}" \
    "--time=$time_limit" \
    "--job-name=$job_name" \
    "--output=hpc/logs/${job_name}_%j.out" \
    "--error=hpc/logs/${job_name}_%j.err" \
    hpc/run_v0_baseline.sbatch "$task" "$seed"
}

submit_freesolv() {
  local seed="$1"
  local job_name="v0_freesolv_seed${seed}"
  echo "[submit] freesolv seed=$seed time=$TIME_FREESOLV"
  sbatch \
    "${common_args[@]}" \
    "--time=$TIME_FREESOLV" \
    "--job-name=$job_name" \
    "--output=hpc/logs/${job_name}_%j.out" \
    "--error=hpc/logs/${job_name}_%j.err" \
    hpc/run_freesolv_baseline.sbatch "$seed"
}

for task in $TASKS; do
  for seed in $SEEDS; do
    submit_classification "$task" "$seed"
  done
done

for seed in $SEEDS; do
  submit_freesolv "$seed"
done
