#!/usr/bin/env bash
#SBATCH --job-name=v0_freesolv
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail

SEED="${1:?Usage: sbatch hpc/run_freesolv_baseline.sbatch SEED}"
ENV_NAME="${ENV_NAME:-specmol}"
RESULTS_PATH="${RESULTS_PATH:-hpc/results/freesolv_seed${SEED}.json}"

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
echo "TASK=freesolv"
echo "SEED=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RESULTS_PATH=$RESULTS_PATH"

python run_freesolv_baseline.py \
  --seed "$SEED" \
  --gpu 0 \
  --results-path "$RESULTS_PATH"
