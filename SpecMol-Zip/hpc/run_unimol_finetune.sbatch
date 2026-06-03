#!/usr/bin/env bash
#SBATCH --job-name=unimol_finetune
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# End-to-end Uni-Mol FINETUNE baseline, matched to the SpecMol study splits.
# Replaces the misleading frozen-embedding "Uni-Mol" row with the real
# end-to-end-finetuned Uni-Mol number on the SAME train/test partition as
# V0/V2-T5/T7/RF (fold0=test for cls; split=='test' for reg).
#
# Usage:
#   sbatch hpc/run_unimol_finetune.sbatch
#   DATASETS="bace clintox" sbatch hpc/run_unimol_finetune.sbatch   # subset first
#   EPOCHS=80 SEEDS="9 19 29" sbatch hpc/run_unimol_finetune.sbatch
#
# Output: unimol_finetune_results.json (per-dataset checkpointed; resumable).

set -euo pipefail

ENV_NAME="${ENV_NAME:-specmol}"
# Start with the two sharpest cells; widen to all six once they look sane.
DATASETS="${DATASETS:-bace clintox bbbp freesolv esol lipo}"
SEEDS="${SEEDS:-9 19 29}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUT="${OUT:-unimol_finetune_results.json}"

mkdir -p hpc/logs hpc/results

# Conda activate
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

# Uni-Mol weights / dict (defaults match SpecMol HPC layout). unimol_tools also
# falls back to its bundled weights dir; these are exported for parity with the
# pair-extraction jobs.
export UNIMOL_CKPT="${UNIMOL_CKPT:-$HOME/zhoutianyang/SpecMol/SpecMol-Zip/unimol_weights/mol_pre_all_h_220816.pt}"
export UNIMOL_DICT="${UNIMOL_DICT:-$HOME/miniconda3/envs/${ENV_NAME}/lib/python3.10/site-packages/unimol_tools/weights/mol.dict.txt}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATASETS=$DATASETS  SEEDS=$SEEDS  EPOCHS=$EPOCHS  LR=$LR  BATCH=$BATCH_SIZE"

python reproduce_unimol_finetune.py \
  --datasets $DATASETS \
  --seeds $SEEDS \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --out "$OUT"

echo "DONE -> $OUT"
