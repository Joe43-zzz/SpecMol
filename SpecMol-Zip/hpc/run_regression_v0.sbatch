#!/usr/bin/env bash
#SBATCH --job-name=reg_v0
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# V0 (2D-only) baseline runner for regression tasks.
#
# Usage:
#   TASK=esol sbatch hpc/run_regression_v0.sbatch 9
#   TASK=lipo sbatch hpc/run_regression_v0.sbatch 19
#   TASK=freesolv sbatch hpc/run_regression_v0.sbatch 29

set -euo pipefail

TASK="${TASK:?TASK env var required: esol | lipo | freesolv}"
SEED="${1:?Usage: TASK=<esol|lipo|freesolv> sbatch hpc/run_regression_v0.sbatch SEED}"
ENV_NAME="${ENV_NAME:-specmol}"
DATA_ROOT="${DATA_ROOT:-down_task_${TASK}_unimol_v2}"
RESULTS_PATH="${RESULTS_PATH:-hpc/results/${TASK}_v0_seed${SEED}.json}"

mkdir -p hpc/logs hpc/results pkl results

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda was not found" >&2
  exit 1
fi

conda activate "$ENV_NAME"
export PYTHONUNBUFFERED=1

echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "HOST=$(hostname)"
echo "TASK=$TASK"
echo "VARIANT=v0"
echo "SEED=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_ROOT=$DATA_ROOT"
echo "RESULTS_PATH=$RESULTS_PATH"

python run_regression_v0.py \
  --task "$TASK" \
  --seed "$SEED" \
  --gpu 0 \
  --data-root "$DATA_ROOT" \
  --results-path "$RESULTS_PATH"
