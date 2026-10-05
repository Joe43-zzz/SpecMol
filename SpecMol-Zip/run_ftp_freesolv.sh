#!/usr/bin/env bash
# FreeSolv 4-arm FINE-TUNE probe (regression, RMSE lower=better).
# Mirrors the BACE fine-tune probe: same finetune knobs, 3 init seeds.
# Q: does unfreezing the encoder reveal a 3D (V2-T5 / T7) edge over V0 that the
#    frozen probe hid? random arm = pair-shuffle control (real geometry vs noise).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/specmol/SpecMol/SpecMol-Zip
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24
C="--task freesolv --path down_task_freesolv_unimol_v2 --finetune --epochs 300 --ft_max_epochs 40 --ft_patience 15 --ft_gate_lr 1e-2 --ft_encoder_lr 1e-4 --ft_head_lr 1e-3 --eval_seeds 9,19,29 --batch_size 256 --K 10 --hid_dim 512"
echo "FTP_FREESOLV start $(date)"
CUDA_VISIBLE_DEVICES=0 python -u main_pretrain.py $C                           > ~/specmol/logs/ftpfs_v0.log     2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u main_pretrain.py $C --use_v2                  > ~/specmol/logs/ftpfs_v2t5.log   2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u main_pretrain.py $C --use_v2 --t7             > ~/specmol/logs/ftpfs_t7.log     2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u main_pretrain.py $C --use_v2 --randomize_pair > ~/specmol/logs/ftpfs_random.log 2>&1 &
wait
echo "FTP_FREESOLV_DONE $(date)"
echo "=== FreeSolv test RMSE (finetune, lower=better) ==="
for a in v0 v2t5 t7 random; do printf "%-8s " "$a:"; grep -h "Best Test RMSE" ~/specmol/logs/ftpfs_$a.log | sed -E 's/.*Epoch [0-9]+ is ([0-9.]+).*/\1/' | tr '\n' ' '; echo; done
