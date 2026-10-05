#!/usr/bin/env bash
#SBATCH --job-name=v0_baseline
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail

TASK="${1:?Usage: sbatch hpc/run_v0_baseline.sbatch TASK SEED [EPOCHS] [BATCH_SIZE]}"
SEED="${2:?Usage: sbatch hpc/run_v0_baseline.sbatch TASK SEED [EPOCHS] [BATCH_SIZE]}"
EPOCHS="${3:-1000}"
BATCH_SIZE="${4:-512}"
DATA_ROOT="${DATA_ROOT:-down_task_v0}"
EVAL_SEEDS="${EVAL_SEEDS:-9,19,29}"
ENV_NAME="${ENV_NAME:-specmol}"

mkdir -p hpc/logs hpc/results pkl results

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda was not found" >&2
  exit 1
fi

conda activate "$ENV_NAME"

echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "HOST=$(hostname)"
echo "TASK=$TASK"
echo "SEED=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_ROOT=$DATA_ROOT"

python main_pretrain.py \
  --task "$TASK" \
  --path "$DATA_ROOT" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --gpu 0 \
  --random_seed "$SEED" \
  --eval_seeds "$EVAL_SEEDS"
