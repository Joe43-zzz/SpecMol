#!/usr/bin/env bash
# Overnight inductive-protocol sweep -- SLOT SCHEDULER (keeps all 4 GPUs saturated).
# Matrix: {dataset x pretrain_split(all|train) x arm x mode} @ unified 1000-epoch budget.
# A GPU that frees immediately picks up the next job (no wave barrier / no T7 straggler idle).
# Per-job 3h timeout, failures isolated, auto summary + SWEEP_ALL_DONE flag.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/specmol/SpecMol/SpecMol-Zip
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 NUMEXPR_NUM_THREADS=20
export HF_ENDPOINT=https://hf-mirror.com
LOG=~/specmol/logs; mkdir -p "$LOG"

COMMON="--epochs 1000 --eval_seeds 9,19,29 --batch_size 256 --K 10 --hid_dim 512"
FT="--finetune --ft_max_epochs 40 --ft_patience 15 --ft_gate_lr 1e-2 --ft_encoder_lr 1e-4 --ft_head_lr 1e-3"

path_for(){ case $1 in bace) echo down_task_bace_v2;; freesolv) echo down_task_freesolv_unimol_v2;; bbbp) echo down_task_bbbp_v2;; bace_scaffold) echo down_task_bace_scaffold;; esac; }
task_for(){ case $1 in bace_scaffold) echo bace;; *) echo $1;; esac; }
arm_for(){ case $1 in V0) echo "";; V2T5) echo "--use_v2";; T7) echo "--use_v2 --t7";; random) echo "--use_v2 --randomize_pair";; esac; }
mode_for(){ case $1 in finetune) echo "$FT";; frozen) echo "";; esac; }

# Priority order (finetune decisive first: BACE -> FreeSolv -> BBBP, all+train; then frozen; then P6).
JOBS=(
 "bace all V0 finetune" "bace all V2T5 finetune" "bace all T7 finetune" "bace all random finetune"
 "bace train V0 finetune" "bace train V2T5 finetune" "bace train T7 finetune" "bace train random finetune"
 "freesolv all V0 finetune" "freesolv all V2T5 finetune" "freesolv all T7 finetune" "freesolv all random finetune"
 "freesolv train V0 finetune" "freesolv train V2T5 finetune" "freesolv train T7 finetune" "freesolv train random finetune"
 "bbbp all V0 finetune" "bbbp all V2T5 finetune" "bbbp all T7 finetune" "bbbp all random finetune"
 "bbbp train V0 finetune" "bbbp train V2T5 finetune" "bbbp train T7 finetune" "bbbp train random finetune"
 "bace all V0 frozen" "bace all V2T5 frozen" "bace train V0 frozen" "bace train V2T5 frozen"
 "freesolv all V0 frozen" "freesolv train V0 frozen" "bbbp all V0 frozen" "bbbp train V0 frozen"
 "bace_scaffold all V0 frozen"
)

LAST_PID=0
launch_idx(){   # $1=gpu $2=jobindex ; sets global LAST_PID
  local gpu=$1 idx=$2
  set -- ${JOBS[$idx]}; local ds=$1 sp=$2 arm=$3 mode=$4
  local TASK PATHA ARMF MODEF LF
  TASK=$(task_for $ds); PATHA=$(path_for $ds); ARMF=$(arm_for $arm); MODEF=$(mode_for $mode)
  LF=$LOG/sweep_${ds}_${sp}_${arm}_${mode}.log
  echo "[launch gpu$gpu job$idx] $ds $sp $arm $mode -> $LF  $(date +%H:%M:%S)"
  ( timeout 10800 env CUDA_VISIBLE_DEVICES=$gpu python -u main_pretrain.py \
      --task $TASK --path $PATHA --pretrain_split $sp $COMMON $ARMF $MODEF > "$LF" 2>&1
    echo "JOBDONE $ds $sp $arm $mode rc=$?" >> "$LF" ) &
  LAST_PID=$!
}

N=${#JOBS[@]}
echo "SWEEP_START $(date)  njobs=$N  (slot scheduler, 4 GPUs)"
declare -A PID2GPU
i=0
# initial fill: one job per GPU 0-3
for g in 0 1 2 3; do
  [ $i -ge $N ] && break
  launch_idx $g $i; PID2GPU[$LAST_PID]=$g; i=$((i+1))
done
# refill: whenever a job finishes, launch the next on the freed GPU
while [ ${#PID2GPU[@]} -gt 0 ]; do
  wait -n 2>/dev/null
  for pid in "${!PID2GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      g=${PID2GPU[$pid]}; unset 'PID2GPU[$pid]'
      if [ $i -lt $N ]; then
        launch_idx $g $i; PID2GPU[$LAST_PID]=$g; i=$((i+1))
      fi
    fi
  done
done

# --- summary ---
SUM=~/specmol/sweep_summary.txt
{ echo "=== SWEEP SUMMARY $(date) ==="
  printf "%-42s %s\n" "job(ds_split_arm_mode)" "test_metric(seed 9 19 29)"
  for f in "$LOG"/sweep_*.log; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .log | sed 's/^sweep_//')
    metric=$(grep -hE 'Best Test (Auc|RMSE)' "$f" | sed -E 's/.*is ([0-9.]+).*/\1/' | tr '\n' ' ')
    rc=$(grep -h '^JOBDONE' "$f" | tail -1)
    printf "%-42s %s  [%s]\n" "$name" "${metric:-NONE}" "$rc"
  done
} | tee "$SUM"
echo "SWEEP_ALL_DONE $(date)" | tee -a "$SUM"
