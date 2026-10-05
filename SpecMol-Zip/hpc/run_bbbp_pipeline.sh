#!/usr/bin/env bash
###############################################################################
# BBBP Full Pipeline on HPC: Data Prep -> V0 Baseline -> V2-T5 -> T6
#
# Run this script INSIDE an srun GPU session:
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=24:00:00 --pty bash
#   cd ~/zhoutianyang/SpecMol/SpecMol-Zip
#   bash hpc/run_bbbp_pipeline.sh [phase]
#
# Phases (run in order, or specify one to resume):
#   prep     - Data preparation (Uni-Mol export + pair_rep extract + dataset build)
#   v0       - V0 baseline (3 seeds)
#   v2       - V2-T5 static pair (3 seeds)
#   t6       - T6 dynamic pair-node update (3 seeds)
#   all      - Run all phases (default)
#   collect  - Collect and compare results
###############################################################################
set -euo pipefail

PHASE="${1:-all}"
TASK="bbbp"
SEEDS="9 19 29"
EVAL_SEEDS="9,19,29"
EPOCHS=1000
BATCH_SIZE=512
HID_DIM=512
PATIENCE=100
ENV_NAME="${ENV_NAME:-specmol}"
LOG_DIR="hpc/logs/bbbp"

mkdir -p "$LOG_DIR" hpc/results pkl results

# ---------- Conda activation ----------
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

echo "=== BBBP Pipeline Phase: $PHASE ==="
echo "HOST=$(hostname) GPU=$CUDA_VISIBLE_DEVICES"
echo "Python=$(which python) Torch=$(python -c 'import torch; print(torch.__version__)')"

# ==========================================================================
# Phase: PREP - Uni-Mol data preparation
# ==========================================================================
run_prep() {
  echo ""
  echo "========================================"
  echo "Phase 0: Data Preparation"
  echo "========================================"

  # 0a. Prepare raw CSV (if not exists)
  echo "[prep] Step 0a: Raw CSV..."
  python prepare_molnet_dataset.py --task "$TASK"

  # 0b. Export to Uni-Mol format (SDF + scaffold fold split)
  echo "[prep] Step 0b: Uni-Mol export (SDF + folds)..."
  python export_unimol_task.py --task "$TASK"

  # 0c. Extract pair representations
  # Needs UNIMOL_CKPT and UNIMOL_DICT env vars, or pass via args
  echo "[prep] Step 0c: Extract Uni-Mol pair representations..."
  SDF_DIR="unimol_out_${TASK}/sdf"
  SDF_FILE=$(ls "${SDF_DIR}"/*.sdf 2>/dev/null | head -1)
  if [ -z "${SDF_FILE:-}" ]; then
    echo "ERROR: No SDF file found in $SDF_DIR" >&2
    echo "  Make sure Step 0b completed successfully." >&2
    exit 1
  fi
  python tools/extract_unimol_pair.py \
    --sdf-path "$SDF_FILE" \
    --output-dir "unimol_out_${TASK}/pair_rep" \
    --batch-size 8 \
    --device cuda:0

  # 0d. Build V2 dataset (with Uni-Mol pair_repr)
  echo "[prep] Step 0d: Build V2 dataset..."
  python make_task_from_unimol.py --task "$TASK" --mode v2

  # 0e. Build baseline dataset (same fold split, no pair_repr)
  echo "[prep] Step 0e: Build baseline dataset (unifold split)..."
  python make_task_from_unimol.py --task "$TASK" --mode baseline

  echo "[prep] Data preparation complete!"
  echo "  V0 data: down_task_${TASK}_unifold/processed/"
  echo "  V2 data: down_task_${TASK}_v2/processed/"
}

# ==========================================================================
# Phase: V0 - Baseline (3 seeds)
# ==========================================================================
run_v0() {
  echo ""
  echo "========================================"
  echo "Phase 1: V0 Baseline (seeds: $SEEDS)"
  echo "========================================"

  for seed in $SEEDS; do
    echo ""
    echo "--- V0 seed=$seed ---"
    python main_pretrain.py \
      --task "$TASK" \
      --path "down_task_${TASK}_unifold" \
      --batch_size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --gpu 0 \
      --hid_dim "$HID_DIM" \
      --K 10 \
      --random_seed "$seed" \
      --patience "$PATIENCE" \
      --eval_seeds "$EVAL_SEEDS" \
      2>&1 | tee "${LOG_DIR}/v0_seed${seed}_stdout.log"
  done
}

# ==========================================================================
# Phase: V2 - V2-T5 Static Pair (3 seeds)
# ==========================================================================
run_v2() {
  echo ""
  echo "========================================"
  echo "Phase 2: V2-T5 Static Pair (seeds: $SEEDS)"
  echo "========================================"

  for seed in $SEEDS; do
    echo ""
    echo "--- V2-T5 seed=$seed ---"
    python main_pretrain.py \
      --task "$TASK" \
      --path "down_task_${TASK}_v2" \
      --batch_size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --gpu 0 \
      --hid_dim "$HID_DIM" \
      --K 10 \
      --random_seed "$seed" \
      --patience "$PATIENCE" \
      --eval_seeds "$EVAL_SEEDS" \
      --use_v2 \
      2>&1 | tee "${LOG_DIR}/v2t5_seed${seed}_stdout.log"
  done
}

# ==========================================================================
# Phase: T6 - Dynamic Bidirectional Pair-Node Update (3 seeds)
# ==========================================================================
run_t6() {
  echo ""
  echo "========================================"
  echo "Phase 3: T6 Dynamic Update (seeds: $SEEDS)"
  echo "========================================"

  for seed in $SEEDS; do
    echo ""
    echo "--- T6 seed=$seed ---"
    python main_pretrain.py \
      --task "$TASK" \
      --path "down_task_${TASK}_v2" \
      --batch_size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --gpu 0 \
      --hid_dim "$HID_DIM" \
      --K 10 \
      --random_seed "$seed" \
      --patience "$PATIENCE" \
      --eval_seeds "$EVAL_SEEDS" \
      --use_v2 --t6 \
      2>&1 | tee "${LOG_DIR}/t6_seed${seed}_stdout.log"
  done
}

# ==========================================================================
# Phase: COLLECT - Gather results
# ==========================================================================
run_collect() {
  echo ""
  echo "========================================"
  echo "Collecting Results"
  echo "========================================"
  python hpc/collect_bbbp_results.py \
    --log-dir "$LOG_DIR" \
    --output "hpc/results/bbbp_all_results.json"
}

# ==========================================================================
# Dispatch
# ==========================================================================
case "$PHASE" in
  prep)    run_prep ;;
  v0)      run_v0 ;;
  v2)      run_v2 ;;
  t6)      run_t6 ;;
  collect) run_collect ;;
  all)
    run_prep
    run_v0
    run_v2
    run_t6
    run_collect
    ;;
  *)
    echo "Unknown phase: $PHASE"
    echo "Usage: $0 {prep|v0|v2|t6|collect|all}"
    exit 1
    ;;
esac

echo ""
echo "=== Phase '$PHASE' complete ==="
