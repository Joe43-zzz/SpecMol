#!/usr/bin/env bash
#SBATCH --job-name=reg_pair_extract
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Extract Uni-Mol encoder_pair_rep for a regression task (esol/lipo/freesolv).
#
# Prerequisite: `python prepare_regression_unimol_input.py --task $TASK` already ran
# locally (or on a CPU node) and produced unimol_out_${TASK}/sdf/${TASK}_for_unimol.sdf
# + manifest.csv.
#
# Usage:
#   TASK=esol sbatch hpc/run_regression_pair_extract.sbatch
#   TASK=lipo sbatch hpc/run_regression_pair_extract.sbatch

set -euo pipefail

TASK="${TASK:?TASK env var required: esol | lipo | freesolv}"
ENV_NAME="${ENV_NAME:-specmol}"
UNIMOL_CKPT="${UNIMOL_CKPT:-$HOME/zhoutianyang/SpecMol/SpecMol-Zip/unimol_weights/mol_pre_all_h_220816.pt}"
UNIMOL_DICT="${UNIMOL_DICT:-$HOME/miniconda3/envs/specmol/lib/python3.10/site-packages/unimol_tools/weights/mol.dict.txt}"

SDF_PATH="unimol_out_${TASK}/sdf/${TASK}_for_unimol.sdf"
OUT_DIR="unimol_out_${TASK}/pair_rep"

mkdir -p hpc/logs "$OUT_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda was not found" >&2
  exit 1
fi

conda activate "$ENV_NAME"

if [ ! -f "$SDF_PATH" ]; then
  echo "Missing SDF: $SDF_PATH. Run prepare_regression_unimol_input.py --task $TASK first." >&2
  exit 1
fi

echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "HOST=$(hostname)"
echo "TASK=$TASK"
echo "SDF=$SDF_PATH"
echo "OUT=$OUT_DIR"
echo "CKPT=$UNIMOL_CKPT"
echo "DICT=$UNIMOL_DICT"

python tools/extract_unimol_pair.py \
  --sdf-path "$SDF_PATH" \
  --output-dir "$OUT_DIR" \
  --ckpt-path "$UNIMOL_CKPT" \
  --dict-path "$UNIMOL_DICT"

# Then trigger the .pt builder on CPU (fast):
python make_regression_unimol_data.py --task "$TASK"
