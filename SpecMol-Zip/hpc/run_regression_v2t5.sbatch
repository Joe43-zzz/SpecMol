#!/usr/bin/env bash
#SBATCH --job-name=reg_v2t5
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Generic regression task V2-T5 / T7 runner. Use TASK env var to pick {esol, lipo, freesolv}.
#
# Usage:
#   TASK=esol sbatch hpc/run_regression_v2t5.sbatch 9
#   TASK=lipo USE_T7=1 sbatch hpc/run_regression_v2t5.sbatch 19
#
# Env vars:
#   TASK        (required) one of {esol, lipo, freesolv}
#   USE_T7      (default 0) set 1 to use T7 variant
#   ENV_NAME    (default specmol)
#   PAIR_DIM    (default 64)
#   DATA_ROOT   (default down_task_${TASK}_unimol_v2)
#   RESULTS_PATH (default hpc/results/${TASK}_${VARIANT}_seed${SEED}.json)

set -euo pipefail

TASK="${TASK:?TASK env var required: esol | lipo | freesolv}"
SEED="${1:?Usage: TASK=<esol|lipo|freesolv> sbatch hpc/run_regression_v2t5.sbatch SEED}"
ENV_NAME="${ENV_NAME:-specmol}"
USE_T7="${USE_T7:-0}"
if [ "$USE_T7" = "1" ]; then
  VARIANT_TAG="t7"
  T7_FLAG="--t7"
else
  VARIANT_TAG="v2t5"
  T7_FLAG=""
fi
PAIR_DIM="${PAIR_DIM:-64}"
DATA_ROOT="${DATA_ROOT:-down_task_${TASK}_unimol_v2}"
RESULTS_PATH="${RESULTS_PATH:-hpc/results/${TASK}_${VARIANT_TAG}_seed${SEED}.json}"

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

# Force unbuffered stdout so progress is visible in real time (1000-epoch
# pretrain otherwise silent until process exit due to block buffering).
export PYTHONUNBUFFERED=1

echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "HOST=$(hostname)"
echo "TASK=$TASK"
echo "VARIANT=$VARIANT_TAG"
echo "SEED=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_ROOT=$DATA_ROOT"
echo "PAIR_DIM=$PAIR_DIM"
echo "RESULTS_PATH=$RESULTS_PATH"
echo "USE_T7=$USE_T7"

python run_regression_v2t5.py \
  --task "$TASK" \
  --seed "$SEED" \
  --gpu 0 \
  --data-root "$DATA_ROOT" \
  --pair-dim "$PAIR_DIM" \
  --results-path "$RESULTS_PATH" \
  $T7_FLAG
