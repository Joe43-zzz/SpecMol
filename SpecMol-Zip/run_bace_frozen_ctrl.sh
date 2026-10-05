#!/usr/bin/env bash
# BACE V0 FROZEN-probe control: identical pretraining (epochs 300) + data as the
# fine-tune probe, but encoder FROZEN. Nails the "+11 fine-tune lift" claim by
# showing thk-frozen reproduces ~0.758 on the SAME data the fine-tune run used.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/specmol/SpecMol/SpecMol-Zip
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28 OPENBLAS_NUM_THREADS=28 NUMEXPR_NUM_THREADS=28
C="--task bace --path down_task_bace_v2 --epochs 300 --eval_seeds 9,19,29 --batch_size 256 --K 10 --hid_dim 512"
echo "BACE_FROZEN start $(date)"
CUDA_VISIBLE_DEVICES=0 python -u main_pretrain.py $C > ~/specmol/logs/bace_frozen_v0.log 2>&1
echo "BACE_FROZEN_DONE $(date)"
echo "=== test AUCs (frozen) ==="
grep -h "Best Test" ~/specmol/logs/bace_frozen_v0.log
