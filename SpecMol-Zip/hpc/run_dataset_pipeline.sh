#!/usr/bin/env bash
###############################################################################
# Task-agnostic dataset pipeline: Data Prep -> V0 / V2-T5 / T6 / T6-safe.
#
# Usage:
#   TASK=bace bash hpc/run_dataset_pipeline.sh [phase]
#   TASK=bbbp bash hpc/run_dataset_pipeline.sh [phase]
#   TASK=hiv  bash hpc/run_dataset_pipeline.sh [phase]
#
# Phases:
#   prep     - Data preparation (Uni-Mol export + pair_rep + dataset build)
#   v0       - V0 baseline (SEEDS seeds)
#   v2       - V2-T5 static pair (SEEDS seeds)
#   t6       - T6 dynamic pair-node update (SEEDS seeds)
#   t6safe0  - T6-safe frozen-zero gate sanity (SEEDS seeds)
#   t6safe   - T6-safe learnable gated update (SEEDS seeds)
#   all      - Run prep + v0 + v2 + t6 + collect (default)
#   collect  - Collect results into hpc/results/${TASK}_all_results.json
#
# Env vars (with defaults):
#   TASK         (required) bace | bbbp | hiv | clintox | tox21
#   SEEDS        "9 19 29"
#   EVAL_SEEDS   "9,19,29"
#   EPOCHS       1000
#   BATCH_SIZE   512
#   HID_DIM      512
#   K            10
#   PATIENCE     100
#   ENV_NAME     specmol
#   FORCE_PAIR_EXTRACT  0   set to 1 to re-extract Uni-Mol pair_repr even if cached
#   SKIP_PAIR_EXTRACT   0   set to 1 to skip extraction entirely (e.g. HIV uses chunked array job)
#   BASELINE_ONLY       0   set to 1 to prep V0 data only (skip step 0c pair extract + 0d V2 build)
#
# For HIV: run hpc/run_pair_extract.sbatch as Slurm array first, then
#   SKIP_PAIR_EXTRACT=1 TASK=hiv bash hpc/run_dataset_pipeline.sh prep
#
# For V0-only baselines (no pair_rep needed): BASELINE_ONLY=1 — prep stays CPU-only
# and no chunked array job is needed even for large tasks like HIV.
###############################################################################
set -euo pipefail

PHASE="${1:-all}"
TASK="${TASK:?TASK env var required: bace | bbbp | hiv | clintox | tox21}"
SEEDS="${SEEDS:-9 19 29}"
EVAL_SEEDS="${EVAL_SEEDS:-9,19,29}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
HID_DIM="${HID_DIM:-512}"
K="${K:-10}"
PATIENCE="${PATIENCE:-100}"
ENV_NAME="${ENV_NAME:-specmol}"
FORCE_PAIR_EXTRACT="${FORCE_PAIR_EXTRACT:-0}"
SKIP_PAIR_EXTRACT="${SKIP_PAIR_EXTRACT:-0}"
BASELINE_ONLY="${BASELINE_ONLY:-0}"

LOG_DIR="hpc/logs/${TASK}"
RESULT_FILE="hpc/results/${TASK}_all_results.json"
PAIR_REP_DIR="unimol_out_${TASK}/pair_rep"
DATA_V2="down_task_${TASK}_v2"
DATA_UNIFOLD="down_task_${TASK}_unifold"

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

echo "=== ${TASK^^} Pipeline Phase: $PHASE ==="
echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python=$(which python) Torch=$(python -c 'import torch; print(torch.__version__)')"
echo "Seeds=$SEEDS Epochs=$EPOCHS BatchSize=$BATCH_SIZE HidDim=$HID_DIM K=$K Patience=$PATIENCE"

# Helper: run main_pretrain.py with a variant-specific flag set
run_variant() {
  local variant_name="$1"
  local data_path="$2"
  shift 2
  local extra_flags="$*"

  for seed in $SEEDS; do
    echo ""
    echo "--- ${variant_name} seed=$seed ---"
    python main_pretrain.py \
      --task "$TASK" \
      --path "$data_path" \
      --batch_size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --gpu 0 \
      --hid_dim "$HID_DIM" \
      --K "$K" \
      --random_seed "$seed" \
      --patience "$PATIENCE" \
      --eval_seeds "$EVAL_SEEDS" \
      $extra_flags \
      2>&1 | tee "${LOG_DIR}/${variant_name}_seed${seed}_stdout.log"
  done
}

# ==========================================================================
# Phase: PREP
# ==========================================================================
run_prep() {
  echo ""
  echo "========================================"
  echo "Phase 0: Data Preparation ($TASK)"
  echo "========================================"

  echo "[prep] Step 0a: Raw CSV..."
  python prepare_molnet_dataset.py --task "$TASK"

  echo "[prep] Step 0b: Uni-Mol export (SDF + folds)..."
  if [ -d "unimol_out_${TASK}/sdf" ] && [ "$(ls -A unimol_out_${TASK}/sdf 2>/dev/null)" ] \
     && [ "${FORCE_PAIR_EXTRACT}" != "1" ]; then
    echo "[prep] SDF already present in unimol_out_${TASK}/sdf, skipping export."
  else
    python export_unimol_task.py --task "$TASK"
  fi

  if [ "${BASELINE_ONLY}" = "1" ]; then
    echo "[prep] BASELINE_ONLY=1: skip Step 0c (pair extract) + 0d (V2 build) — V0 needs no pair_rep."
  else
    echo "[prep] Step 0c: Extract Uni-Mol pair representations..."
    if [ "${SKIP_PAIR_EXTRACT}" = "1" ]; then
      echo "[prep] SKIP_PAIR_EXTRACT=1, assuming pair_rep already present at $PAIR_REP_DIR."
    elif [ -d "$PAIR_REP_DIR" ] && [ "$(ls -A $PAIR_REP_DIR 2>/dev/null)" ] \
         && [ "${FORCE_PAIR_EXTRACT}" != "1" ]; then
      echo "[prep] pair_rep already present at $PAIR_REP_DIR, skipping. Set FORCE_PAIR_EXTRACT=1 to redo."
    else
      SDF_FILE=$(ls unimol_out_${TASK}/sdf/*.sdf 2>/dev/null | head -1)
      if [ -z "${SDF_FILE:-}" ]; then
        echo "ERROR: No SDF in unimol_out_${TASK}/sdf" >&2; exit 1
      fi
      python tools/extract_unimol_pair.py \
        --sdf-path "$SDF_FILE" \
        --output-dir "$PAIR_REP_DIR" \
        --batch-size 8 \
        --device cuda:0
    fi

    echo "[prep] Step 0d: Build V2 dataset (with pair_repr)..."
    python make_task_from_unimol.py --task "$TASK" --mode v2
  fi

  echo "[prep] Step 0e: Build baseline unifold dataset (no pair_repr)..."
  python make_task_from_unimol.py --task "$TASK" --mode baseline

  echo "[prep] Done."
  echo "  V0 data: ${DATA_UNIFOLD}/processed/"
  if [ "${BASELINE_ONLY}" != "1" ]; then
    echo "  V2 data: ${DATA_V2}/processed/"
  fi
}

# ==========================================================================
# Training phases (delegate to run_variant)
# ==========================================================================
run_v0()       { echo ""; echo "=== Phase: V0 Baseline ==="; run_variant "v0"       "$DATA_UNIFOLD" ""; }
run_v2()       { echo ""; echo "=== Phase: V2-T5 Static ==="; run_variant "v2t5"    "$DATA_V2"      "--use_v2"; }
run_t6()       { echo ""; echo "=== Phase: T6 Dynamic ==="; run_variant "t6"       "$DATA_V2"      "--use_v2 --t6"; }
run_t6safe0()  { echo ""; echo "=== Phase: T6-safe frozen-zero ==="; run_variant "t6safe0" "$DATA_V2" "--use_v2 --t6_safe --t6_safe_frozen_zero --diagnostics"; }
run_t6safe()   { echo ""; echo "=== Phase: T6-safe learnable ==="; run_variant "t6safe"  "$DATA_V2" "--use_v2 --t6_safe --diagnostics"; }

# ==========================================================================
# Phase: COLLECT
# ==========================================================================
run_collect() {
  echo ""
  echo "========================================"
  echo "Collecting Results ($TASK)"
  echo "========================================"
  python hpc/collect_results.py \
    --task "$TASK" \
    --log-dir "$LOG_DIR" \
    --eval-seeds "$EVAL_SEEDS" \
    --output "$RESULT_FILE"
}

# ==========================================================================
# Dispatch
# ==========================================================================
case "$PHASE" in
  prep)    run_prep ;;
  v0)      run_v0 ;;
  v2)      run_v2 ;;
  t6)      run_t6 ;;
  t6safe0) run_t6safe0 ;;
  t6safe)  run_t6safe ;;
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
    echo "Usage: TASK=<task> $0 {prep|v0|v2|t6|t6safe0|t6safe|collect|all}"
    exit 1
    ;;
esac

echo ""
echo "=== Phase '$PHASE' on task '$TASK' complete ==="
