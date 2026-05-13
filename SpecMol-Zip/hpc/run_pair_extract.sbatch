#!/usr/bin/env bash
#SBATCH --job-name=pair_extract
#SBATCH --output=hpc/logs/%x_%A_%a.out
#SBATCH --error=hpc/logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Slurm array job for chunked Uni-Mol pair_rep extraction.
#
# Prerequisites:
#   1. SDF generated:  python export_unimol_task.py --task <TASK>
#   2. SDF split:      python tools/split_sdf.py --task <TASK> --chunk-size 5000
#
# Usage:
#   sbatch --array=0-N hpc/run_pair_extract.sbatch TASK
#   where N = manifest['n_chunks'] - 1 (printed by split_sdf.py)
#
# After all array tasks complete:
#   python tools/merge_pair_rep.py --task <TASK>

set -euo pipefail

TASK="${1:?Usage: sbatch --array=0-N hpc/run_pair_extract.sbatch TASK}"
ENV_NAME="${ENV_NAME:-specmol}"
BATCH_SIZE="${BATCH_SIZE:-8}"

CHUNK_IDX="${SLURM_ARRAY_TASK_ID:?This script requires SLURM_ARRAY_TASK_ID (use --array=0-N)}"
CHUNK_ID=$(printf "%02d" "$CHUNK_IDX")

SDF_PATH="unimol_out_${TASK}/sdf/chunks/smiles_unimol_chunk_${CHUNK_ID}.sdf"
OUT_DIR="unimol_out_${TASK}/pair_rep_chunks/chunk_${CHUNK_ID}"

if [ ! -f "$SDF_PATH" ]; then
  echo "ERROR: chunk SDF not found: $SDF_PATH" >&2
  echo "Did you run: python tools/split_sdf.py --task $TASK ?" >&2
  exit 1
fi

# Conda
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$ENV_NAME"

echo "JOB_ID=${SLURM_JOB_ID:-none} ARRAY_TASK_ID=$CHUNK_IDX HOST=$(hostname)"
echo "TASK=$TASK CHUNK=$CHUNK_ID SDF=$SDF_PATH OUT=$OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

mkdir -p "$OUT_DIR"

python tools/extract_unimol_pair.py \
  --sdf-path "$SDF_PATH" \
  --output-dir "$OUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --device cuda:0

echo "Done: chunk $CHUNK_ID for task $TASK"
