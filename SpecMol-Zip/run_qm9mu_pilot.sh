#!/usr/bin/env bash
# QM9 dipole(mu) 7-arm pilot: does the GEOMETRY effect (real-pair vs shuffled-pair)
# AMPLIFY on a geometry-pure target, vs QM7's composition-dominated atomization energy
# (where T7 geometry effect was ~0 and T8 only ~3.5)? Arms include the T7/T8 random
# controls so the architecture-vs-geometry split is measured directly.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/specmol/SpecMol/SpecMol-Zip
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 NUMEXPR_NUM_THREADS=20
L=~/specmol/logs; mkdir -p "$L"
C="--task qm9mu --path down_task_qm9mu_unimol_v2 --finetune --epochs 1000 --eval_seeds 9,19,29 --batch_size 1024 --K 10 --hid_dim 512 --ft_max_epochs 40 --ft_patience 15 --ft_gate_lr 1e-2 --ft_encoder_lr 1e-4 --ft_head_lr 1e-3"
declare -a NAMES=(V0 V2T5 random T7 T7random T8 T8random)
declare -a FLAGS=("" "--use_v2" "--use_v2 --randomize_pair" "--use_v2 --t7" "--use_v2 --t7 --randomize_pair" "--use_v2 --t8" "--use_v2 --t8 --randomize_pair")
echo "QM9MU_PILOT start $(date)  arms=${#NAMES[@]}"
declare -A PID2GPU
i=0; N=${#NAMES[@]}
launch(){ local g=$1 idx=$2; local nm=${NAMES[$idx]}; local fl=${FLAGS[$idx]}
  local lf=$L/qm9mu_${nm}_finetune.log
  echo "[launch gpu$g] qm9mu $nm -> $lf"
  ( timeout 14400 env CUDA_VISIBLE_DEVICES=$g python -u main_pretrain.py $C $fl > "$lf" 2>&1
    echo "JOBDONE qm9mu $nm rc=$?" >> "$lf" ) &
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
echo "QM9MU_PILOT_DONE $(date)"
echo "=== QM9-mu test RMSE (finetune, lower=better) ==="
for nm in V0 V2T5 random T7 T7random T8 T8random; do printf "%-10s " "$nm:"; grep -hE "Best Test RMSE" $L/qm9mu_${nm}_finetune.log | sed -E 's/.*is ([0-9.]+) .*/\1/' | tr '\n' ' '; echo; done
EOF_GUARD=1
