#!/usr/bin/env bash
# QM7 5-arm pilot (decision gate): does ANY 3D injection (esp. T8, the richest)
# beat a pair-shuffled random control on a GEOMETRY-RICH task, unlike saturated ADMET?
# 5 arms x finetune, 1000ep, slot-scheduled over 4 GPUs.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/specmol/SpecMol/SpecMol-Zip
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 NUMEXPR_NUM_THREADS=20
L=~/specmol/logs; mkdir -p "$L"
C="--task qm7 --path down_task_qm7_unimol_v2 --finetune --epochs 1000 --eval_seeds 9,19,29 --batch_size 256 --K 10 --hid_dim 512 --ft_max_epochs 40 --ft_patience 15 --ft_gate_lr 1e-2 --ft_encoder_lr 1e-4 --ft_head_lr 1e-3"
# arm name -> extra flags
declare -a NAMES=(V0 V2T5 T7 T8 random)
declare -a FLAGS=("" "--use_v2" "--use_v2 --t7" "--use_v2 --t8" "--use_v2 --randomize_pair")
echo "QM7PILOT start $(date)  arms=${#NAMES[@]}"
declare -A PID2GPU
i=0; N=${#NAMES[@]}
launch(){ local g=$1 idx=$2; local nm=${NAMES[$idx]}; local fl=${FLAGS[$idx]}
  local lf=$L/qm7_${nm}_finetune.log
  echo "[launch gpu$g] qm7 $nm finetune -> $lf"
  ( timeout 10800 env CUDA_VISIBLE_DEVICES=$g python -u main_pretrain.py $C $fl > "$lf" 2>&1
    echo "JOBDONE qm7 $nm rc=$?" >> "$lf" ) &
  LAST=$!; }
for g in 0 1 2 3; do [ $i -ge $N ] && break; launch $g $i; PID2GPU[$LAST]=$g; i=$((i+1)); done
while [ ${#PID2GPU[@]} -gt 0 ]; do
  wait -n 2>/dev/null
  for pid in "${!PID2GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      g=${PID2GPU[$pid]}; unset 'PID2GPU[$pid]'
      [ $i -lt $N ] && { launch $g $i; PID2GPU[$LAST]=$g; i=$((i+1)); }
    fi
  done
done
echo "QM7PILOT_DONE $(date)"
echo "=== QM7 test RMSE (finetune, lower=better) ==="
for nm in V0 V2T5 T7 T8 random; do printf "%-8s " "$nm:"; grep -hE "Best Test RMSE" $L/qm7_${nm}_finetune.log | sed -E 's/.*is ([0-9.]+) .*/\1/' | tr '\n' ' '; echo; done
