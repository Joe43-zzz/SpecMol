#!/usr/bin/env bash
# BBBP 4-arm FINE-TUNE probe (classification, AUC higher=better).
# Same finetune knobs as BACE/FreeSolv probes. Q: does unfreezing reveal a 3D
# (V2-T5/T7) edge over V0 that the frozen probe hid? random = pair-shuffle ctrl.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/specmol/SpecMol/SpecMol-Zip
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24
C="--task bbbp --path down_task_bbbp_v2 --finetune --epochs 300 --ft_max_epochs 40 --ft_patience 15 --ft_gate_lr 1e-2 --ft_encoder_lr 1e-4 --ft_head_lr 1e-3 --eval_seeds 9,19,29 --batch_size 256 --K 10 --hid_dim 512"
echo "FTP_BBBP start $(date)"
CUDA_VISIBLE_DEVICES=0 python -u main_pretrain.py $C                           > ~/specmol/logs/ftpbb_v0.log     2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u main_pretrain.py $C --use_v2                  > ~/specmol/logs/ftpbb_v2t5.log   2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u main_pretrain.py $C --use_v2 --t7             > ~/specmol/logs/ftpbb_t7.log     2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u main_pretrain.py $C --use_v2 --randomize_pair > ~/specmol/logs/ftpbb_random.log 2>&1 &
wait
echo "FTP_BBBP_DONE $(date)"
echo "=== BBBP test AUC (finetune, higher=better) ==="
for a in v0 v2t5 t7 random; do printf "%-8s " "$a:"; grep -h "Best Test Auc" ~/specmol/logs/ftpbb_$a.log | sed -E 's/.*Epoch [0-9]+ is ([0-9.]+).*/\1/' | tr '\n' ' '; echo; done
