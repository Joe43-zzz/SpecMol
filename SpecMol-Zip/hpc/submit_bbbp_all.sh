#!/usr/bin/env bash
###############################################################################
# Submit all BBBP training jobs (V0 + V2-T5 + T6) via sbatch.
#
# Prerequisites: data preparation must be done first (run_bbbp_pipeline.sh prep)
#
# Usage:
#   bash hpc/submit_bbbp_all.sh
###############################################################################
set -euo pipefail

PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-48G}"
TIME="${TIME:-12:00:00}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
SEEDS="${SEEDS:-9 19 29}"
VARIANTS="${VARIANTS:-v0 v2 t6}"

mkdir -p hpc/logs/bbbp hpc/results

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
    job_name="bbbp_${variant}_seed${seed}"
    echo "[submit] variant=$variant seed=$seed job=$job_name"
    sbatch \
      "${common_args[@]}" \
      "--job-name=$job_name" \
      "--output=hpc/logs/bbbp/${variant}_seed${seed}_%j.out" \
      "--error=hpc/logs/bbbp/${variant}_seed${seed}_%j.err" \
      hpc/run_bbbp_training.sbatch "$variant" "$seed"
  done
done

echo ""
echo "Submitted $(echo $VARIANTS | wc -w) variants x $(echo $SEEDS | wc -w) seeds = $(( $(echo $VARIANTS | wc -w) * $(echo $SEEDS | wc -w) )) jobs"
echo "Monitor: squeue -u \$USER"
echo "Collect: python hpc/collect_bbbp_results.py --log-dir hpc/logs/bbbp"
