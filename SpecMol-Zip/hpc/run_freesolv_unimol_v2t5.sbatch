#!/usr/bin/env bash
#SBATCH --job-name=freesolv_unimol
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# End-to-end FreeSolv-with-Uni-Mol-pair_repr pipeline.
#
# Steps:
#   1. prepare_freesolv_unimol_input.py  -> SDF + manifest.csv
#   2. tools/extract_unimol_pair.py      -> per-mol pair_rep.pt
#   3. make_freesolv_unimol_data.py      -> down_task_freesolv_unimol_v2/processed/*.pt
#   4. run_freesolv_v2t5.py (3 seeds)    -> hpc/results/freesolv_unimol_v2t5_seed{9,19,29}.json
#
# Usage:
#   sbatch hpc/run_freesolv_unimol_v2t5.sbatch
#   sbatch hpc/run_freesolv_unimol_v2t5.sbatch SKIP_EXTRACT=1   # data already built

set -euo pipefail

ENV_NAME="${ENV_NAME:-specmol}"
SEEDS="${SEEDS:-9 19 29}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

mkdir -p hpc/logs hpc/results pkl

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

# Uni-Mol weights / dict (defaults match SpecMol HPC layout)
export UNIMOL_CKPT="${UNIMOL_CKPT:-$HOME/zhoutianyang/SpecMol/SpecMol-Zip/unimol_weights/mol_pre_all_h_220816.pt}"
export UNIMOL_DICT="${UNIMOL_DICT:-$HOME/miniconda3/envs/${ENV_NAME}/lib/python3.10/site-packages/unimol_tools/weights/mol.dict.txt}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "UNIMOL_CKPT=$UNIMOL_CKPT"
echo "UNIMOL_DICT=$UNIMOL_DICT"
echo "SEEDS=$SEEDS  SKIP_EXTRACT=$SKIP_EXTRACT"

SDF_PATH="unimol_out_freesolv/sdf/freesolv_for_unimol.sdf"
PAIR_REP_DIR="unimol_out_freesolv/pair_rep"
DATA_ROOT="down_task_freesolv_unimol_v2"

if [ "$SKIP_EXTRACT" != "1" ]; then
  echo "[step 1] prepare_freesolv_unimol_input.py"
  python prepare_freesolv_unimol_input.py

  echo "[step 2] tools/extract_unimol_pair.py on $SDF_PATH"
  mkdir -p "$PAIR_REP_DIR"
  python tools/extract_unimol_pair.py \
    --sdf-path "$SDF_PATH" \
    --output-dir "$PAIR_REP_DIR" \
    --batch-size "$BATCH_SIZE" \
    --device cuda:0

  echo "[step 3] make_freesolv_unimol_data.py"
  python make_freesolv_unimol_data.py
else
  echo "[skip] step 1-3 (SKIP_EXTRACT=1)"
fi

# Probe pair_dim from the built dataset so --pair-dim is correct
PAIR_DIM=$(python - <<'PY'
import torch
from pathlib import Path
p = Path("down_task_freesolv_unimol_v2/processed/freesolv_train.pt")
collated, _ = torch.load(p, map_location="cpu")
print(int(collated.pair_repr_edge.size(-1)))
PY
)
echo "Detected pair_dim=$PAIR_DIM"

echo "[step 4] run_freesolv_v2t5.py with --data-root $DATA_ROOT --pair-dim $PAIR_DIM"
for SEED in $SEEDS; do
  RESULTS_PATH="hpc/results/freesolv_unimol_v2t5_seed${SEED}.json"
  echo "  seed=$SEED -> $RESULTS_PATH"
  python run_freesolv_v2t5.py \
    --seed "$SEED" \
    --gpu 0 \
    --data-root "$DATA_ROOT" \
    --pair-dim "$PAIR_DIM" \
    --results-path "$RESULTS_PATH"
done

echo "Done. Result JSONs in hpc/results/freesolv_unimol_v2t5_seed*.json"
