#!/usr/bin/env bash
# T8-on-BACE backfill: the user's atom<->pair co-update arm (--t8), same 1000ep
# protocol as the archived V0/V2-T5/T7/random sweep -> directly comparable.
# Closes the "your null is just a 64->1 scalar bottleneck" reviewer attack:
# if T8 (multi-channel, multi-layer) still ties random/V2-T5, the saturation null holds.
# --diagnostics logs the per-layer ReZero gates (do they move off 0 = does the optimizer use T8).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/specmol/SpecMol/SpecMol-Zip
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24
L=~/specmol/logs
C="--task bace --path down_task_bace_v2 --use_v2 --t8 --epochs 1000 --eval_seeds 9,19,29 --batch_size 256 --K 10 --hid_dim 512 --diagnostics"
FT="--finetune --ft_max_epochs 40 --ft_patience 15 --ft_gate_lr 1e-2 --ft_encoder_lr 1e-4 --ft_head_lr 1e-3"
echo "T8BACE start $(date)"
CUDA_VISIBLE_DEVICES=0 python -u main_pretrain.py $C --pretrain_split all   $FT > $L/sweep_bace_all_T8_finetune.log   2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u main_pretrain.py $C --pretrain_split train $FT > $L/sweep_bace_train_T8_finetune.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u main_pretrain.py $C --pretrain_split all        > $L/sweep_bace_all_T8_frozen.log     2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u main_pretrain.py $C --pretrain_split train      > $L/sweep_bace_train_T8_frozen.log   2>&1 &
wait
echo "T8BACE_DONE $(date)"
for f in bace_all_T8_finetune bace_train_T8_finetune bace_all_T8_frozen bace_train_T8_frozen; do
  printf "%-26s " "$f:"; grep -hE "Best Test Auc" $L/sweep_$f.log | sed -E 's/.*is ([0-9.]+).*/\1/' | tr '\n' ' '; echo
done
echo "=== T8 final gates (node/pair/ffn 是否离 0) ==="
grep -hE "current_t8|node_gate|pair_gate" $L/sweep_bace_all_T8_finetune.log | tail -6
