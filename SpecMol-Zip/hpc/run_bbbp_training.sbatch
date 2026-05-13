#!/usr/bin/env bash
#SBATCH --job-name=bbbp_train
#SBATCH --output=hpc/logs/bbbp/%x_%j.out
#SBATCH --error=hpc/logs/bbbp/%x_%j.err

# Generic BBBP training sbatch: supports V0, V2-T5, and T6 modes.
#
# Usage:
#   sbatch hpc/run_bbbp_training.sbatch VARIANT SEED
#
# VARIANT: v0 | v2 | t6
# SEED:    9, 19, or 29

set -euo pipefail

VARIANT="${1:?Usage: sbatch hpc/run_bbbp_training.sbatch VARIANT SEED}"
SEED="${2:?Usage: sbatch hpc/run_bbbp_training.sbatch VARIANT SEED}"
TASK="bbbp"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_SEEDS="${EVAL_SEEDS:-9,19,29}"
ENV_NAME="${ENV_NAME:-specmol}"

mkdir -p hpc/logs/bbbp hpc/results pkl results

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

echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "HOST=$(hostname)"
echo "VARIANT=$VARIANT SEED=$SEED TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Build command based on variant
CMD="python main_pretrain.py --task $TASK --batch_size $BATCH_SIZE --epochs $EPOCHS"
CMD="$CMD --gpu 0 --hid_dim 512 --K 10 --random_seed $SEED --patience 100"
CMD="$CMD --eval_seeds $EVAL_SEEDS"

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
  *)
    echo "Unknown variant: $VARIANT (use v0, v2, or t6)" >&2
    exit 1
    ;;
esac

echo "CMD: $CMD"
eval "$CMD"
