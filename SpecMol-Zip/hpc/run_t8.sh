#!/bin/bash
# T8 runs: faithful-ish Uni-Mol atom<->pair co-update STACK (top rung of the
# injection ladder T5->T6->T7->T8). Submits one sbatch job per (dataset, seed)
# through the canonical hpc/run_dataset_training.sbatch, so T8 shares the EXACT
# environment, eval protocol, and stdout->results convention as V0/V2-T5/T6/T7.
#
# IMPORTANT (review 2026-06-01): this script does NOT pass --out to
# main_pretrain.py (that flag does not exist; the old version crashed every job
# at startup). Results are scraped from the per-job stdout log that
# run_dataset_training.sbatch tees to hpc/logs/<task>/, exactly like the t7 path.
# Collect with hpc/collect_results.py after the queue drains.
#
# Usage (from repo root, on the HPC login node with VPN up):
#   bash hpc/run_t8.sh                       # ladder: BACE+BBBP, qk_hadamard, bias=5
#   PAIR_UPDATE=logits  bash hpc/run_t8.sh   # scalar-logit ablation
#   BIAS_INIT=1.0       bash hpc/run_t8.sh   # escape the pair_to_edge_weight sigmoid
#                                            #   saturation so the pair side can learn
#                                            #   (review H3: at bias=5 pair-side grad ~0)
#   TASKS="bbbp"  SEEDS="9 19 29"  bash hpc/run_t8.sh
#
# Env knobs:
#   PAIR_UPDATE="qk_hadamard"  qk_hadamard = multi-channel; logits = scalar ablation
#   BIAS_INIT=""               empty = default 5.0; set 1.0 to escape sigmoid saturation
#   SEEDS="9 19 29"            pretrain seeds (matched protocol)
#   TASKS="bace bbbp"          classification ladder datasets
#   DUMP_GATE=""               set "1" to also append --dump_gate_stats
#
# Dispatch by task type (review H2 fixed 2026-06-01):
#   classification (bace/bbbp/clintox/...) -> hpc/run_dataset_training.sbatch
#       (matched env, stdout->collector, no --out).
#   regression (freesolv/esol/lipo) -> run_regression_v2t5.py --t8 --results-path
#       (the SAME runner that produced the V2-T5/T7 regression cells, so the
#       comparison stays matched-protocol). FreeSolv uses its Uni-Mol pair root.

set -euo pipefail

PAIR_UPDATE="${PAIR_UPDATE:-qk_hadamard}"   # qk_hadamard | logits
BIAS_INIT="${BIAS_INIT:-}"                  # empty = default 5.0; 1.0 escapes saturation
SEEDS="${SEEDS:-9 19 29}"
TASKS="${TASKS:-bace bbbp}"
DUMP_GATE="${DUMP_GATE:-}"

command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: 'sbatch' not found — are you on the HPC login node? (VPN up, then ssh mbzuai-hpc)"; exit 1; }
[ -f "hpc/run_dataset_training.sbatch" ] || { echo "ERROR: run from repo root"; exit 1; }

# variant token for the classification sbatch; pair-update label for filenames.
if [ "$PAIR_UPDATE" = "logits" ]; then CLS_VARIANT="t8logits"; else CLS_VARIANT="t8"; fi

# Optional extra args (classification sbatch appends EXTRA_ARGS verbatim).
EXTRA=""
[ -n "$BIAS_INIT" ] && EXTRA="$EXTRA --bias_init $BIAS_INIT"
[ -n "$DUMP_GATE" ] && EXTRA="$EXTRA --dump_gate_stats"
SUFFIX=""
[ "$PAIR_UPDATE" = "logits" ] && SUFFIX="${SUFFIX}_logits"
[ -n "$BIAS_INIT" ] && SUFFIX="${SUFFIX}_b${BIAS_INIT}"

# Regression results-file label mirrors run_regression_v2t5.py's own naming.
if [ "$PAIR_UPDATE" = "logits" ]; then REG_TAG="t8_logits"; else REG_TAG="t8"; fi
REG_BIAS="${BIAS_INIT:-5.0}"   # run_regression_v2t5.py defaults bias_init to 5.0

echo "[t8] tasks=$TASKS seeds=$SEEDS pair_update=$PAIR_UPDATE "\
"bias_init=${BIAS_INIT:-default5.0} extra='${EXTRA}'"

submitted=0
for TASK in $TASKS; do
  case "$TASK" in
    bace|bbbp|clintox|hiv|tox21)
      DATA="down_task_${TASK}_v2"
      if [ ! -d "$DATA/processed" ]; then
        echo "SKIP $TASK: T8 needs pair_repr data at '$DATA/processed' (missing)."; continue
      fi
      for SEED in $SEEDS; do
        echo "submit (cls) -> TASK=$TASK VARIANT=$CLS_VARIANT SEED=$SEED"
        EXTRA_ARGS="$EXTRA" TAG_SUFFIX="$SUFFIX" \
          sbatch hpc/run_dataset_training.sbatch "$TASK" "$CLS_VARIANT" "$SEED"
        submitted=$((submitted + 1))
      done
      ;;
    freesolv|esol|lipo)
      # Regression root: FreeSolv MUST use its Uni-Mol pair root (not the GBF
      # surrogate) so the T8 cell is comparable to the V2-T5 Uni-Mol cell.
      # esol/lipo default to *_unimol_v2 (matches aggregate_regression_seeds.py's
      # data_root); override with REG_ROOT_ESOL / REG_ROOT_LIPO if your V2-T5
      # cell used a different root (they MUST match for a fair comparison).
      case "$TASK" in
        freesolv) ROOT="down_task_freesolv_unimol_v2" ;;
        esol)     ROOT="${REG_ROOT_ESOL:-down_task_esol_unimol_v2}" ;;
        lipo)     ROOT="${REG_ROOT_LIPO:-down_task_lipo_unimol_v2}" ;;
      esac
      for SEED in $SEEDS; do
        OUT="hpc/results/${TASK}_${REG_TAG}_seed${SEED}.json"
        echo "submit (reg) -> TASK=$TASK t8 pair_update=$PAIR_UPDATE SEED=$SEED root=$ROOT -> $OUT"
        sbatch --job-name="t8_${TASK}_${SEED}" --partition=gpu --gres=gpu:1 \
          --time=08:00:00 --mem=48G --cpus-per-task=8 \
          --output="hpc/logs/t8_${TASK}_${REG_TAG}_seed${SEED}_%j.log" \
          --wrap="bash -c 'source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true; conda activate specmol 2>/dev/null || true; python run_regression_v2t5.py --task $TASK --seed $SEED --t8 --t8_pair_update $PAIR_UPDATE --bias_init $REG_BIAS --data-root $ROOT --results-path $OUT'"
        submitted=$((submitted + 1))
      done
      ;;
    *) echo "ERROR: unknown task $TASK" >&2; exit 1 ;;
  esac
done

echo "[t8] submitted $submitted job(s). Watch: squeue -u \$USER"
echo "[t8] when done, collect from stdout logs:"
for TASK in $TASKS; do
  echo "  python hpc/collect_results.py --task $TASK --output hpc/results/${TASK}_all_results.json"
done
echo "[t8] sanity: grep the job logs for the [diagnostics] mode=current_t8 line"
echo "      (run with --diagnostics) to confirm node_gates/pair_gates moved off 0."
