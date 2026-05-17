#!/usr/bin/env bash
#SBATCH --job-name=dataset_train
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Task-agnostic training sbatch: V0 / V2-T5 / T6 / T6-safe.
#
# Usage:
#   sbatch hpc/run_dataset_training.sbatch TASK VARIANT SEED
#
# Args:
#   TASK    : bace | bbbp | hiv | clintox | tox21
#   VARIANT : v0 | v2 | t6 | t6safe | t6safe0
#   SEED    : 9, 19, or 29 (or any int)
#
# Env override:
#   EPOCHS, BATCH_SIZE, HID_DIM, K, PATIENCE, EVAL_SEEDS, EVAL_EPOCHS, ENV_NAME
#
# Resource defaults (above): 4h / 40G / 1 GPU — fits MBZUAI QOS limits.
# Override via sbatch flags or wrapper script if needed.
#
# IMPORTANT (2026-05-14): default EVAL_EPOCHS=500 (reduced from main_pretrain.py
# implicit default 2000) so that 1 pretrain (≤1000 epoch) + 3 split eval ×500
# fits within 4h QOS wall clock. T6 jobs with implicit 2000 eval_epochs were
# observed to timeout. Override with EVAL_EPOCHS=2000 if you need full protocol
# and have a longer QOS allocation.

set -euo pipefail

TASK="${1:?Usage: sbatch hpc/run_dataset_training.sbatch TASK VARIANT SEED}"
VARIANT="${2:?Usage: sbatch hpc/run_dataset_training.sbatch TASK VARIANT SEED}"
SEED="${3:?Usage: sbatch hpc/run_dataset_training.sbatch TASK VARIANT SEED}"

EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
HID_DIM="${HID_DIM:-512}"
K="${K:-10}"
PATIENCE="${PATIENCE:-100}"
EVAL_SEEDS="${EVAL_SEEDS:-9,19,29}"
EVAL_EPOCHS="${EVAL_EPOCHS:-500}"
ENV_NAME="${ENV_NAME:-specmol}"

LOG_DIR="hpc/logs/${TASK}"
mkdir -p "$LOG_DIR" hpc/results pkl results

# Conda
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda not found" >&2; exit 1
fi
conda activate "$ENV_NAME"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname)"
echo "TASK=$TASK VARIANT=$VARIANT SEED=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

CMD="python main_pretrain.py --task $TASK --batch_size $BATCH_SIZE --epochs $EPOCHS"
CMD="$CMD --gpu 0 --hid_dim $HID_DIM --K $K --random_seed $SEED --patience $PATIENCE"
CMD="$CMD --eval_seeds $EVAL_SEEDS --eval_epochs $EVAL_EPOCHS"

case "$VARIANT" in
  v0)
    CMD="$CMD --path down_task_${TASK}_unifold"
    ;;
  v2)
    CMD="$CMD --path down_task_${TASK}_v2 --use_v2"
    ;;
  t6)
    CMD="$CMD --path down_task_${TASK}_v2 --use_v2 --t6"
    ;;
  t6safe)
    CMD="$CMD --path down_task_${TASK}_v2 --use_v2 --t6_safe --diagnostics"
    ;;
  t6safe0)
    CMD="$CMD --path down_task_${TASK}_v2 --use_v2 --t6_safe --t6_safe_frozen_zero --diagnostics"
    ;;
  *)
    echo "Unknown variant: $VARIANT (use v0, v2, t6, t6safe, or t6safe0)" >&2
    exit 1
    ;;
esac

# Optional extra CLI args appended verbatim. Used for one-off sweeps without
# forking this script (e.g. --bias_init 1.0 --dump_gate_stats for the
# saturation A/B). Empty default is a no-op.
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [ -n "$EXTRA_ARGS" ]; then
  CMD="$CMD $EXTRA_ARGS"
fi

# Tag suffix prevents log filename collisions when the same variant+seed is
# re-submitted with different EXTRA_ARGS. Defaults to empty for backward compat.
TAG_SUFFIX="${TAG_SUFFIX:-}"
LOG_FILE="${LOG_DIR}/${VARIANT}${TAG_SUFFIX}_seed${SEED}_${SLURM_JOB_ID:-local}.log"
echo "CMD: $CMD"
echo "LOG: $LOG_FILE"
eval "$CMD" 2>&1 | tee "$LOG_FILE"
