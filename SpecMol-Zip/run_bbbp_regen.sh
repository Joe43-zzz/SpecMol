#!/usr/bin/env bash
# Regenerate BBBP V2 data on thk (transfer too slow: 400MB @ 33KB/s).
# Chain: export (SDF conformers + scaffold fold, CPU) -> extract pair_rep (GPU)
#        -> make_task v2 (assemble bbbp_{train,valid,test,all}.pt with pair).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/specmol/SpecMol/SpecMol-Zip
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24
export HF_ENDPOINT=https://hf-mirror.com
CKPT=~/miniconda3/envs/specmol/lib/python3.10/site-packages/unimol_tools/weights/mol_pre_all_h_220816.pt
DICT=~/miniconda3/envs/specmol/lib/python3.10/site-packages/unimol_tools/weights/mol.dict.txt

echo "=== STEP1 export (SDF+fold) $(date) ==="
python -u export_unimol_task.py --task bbbp --kfold 10 --split-seed 42
echo "--- SDF produced: ---"; ls -la unimol_out_bbbp/sdf/

echo "=== STEP2 extract pair_rep (GPU3) $(date) ==="
SDF=$(ls unimol_out_bbbp/sdf/*.sdf | head -1)
echo "using SDF=$SDF"
CUDA_VISIBLE_DEVICES=3 python -u tools/extract_unimol_pair.py --sdf-path "$SDF" --output-dir unimol_out_bbbp/pair_rep --ckpt-path "$CKPT" --dict-path "$DICT" --batch-size 4
echo "--- pair_rep count: ---"; find unimol_out_bbbp/pair_rep -type f | wc -l

echo "=== STEP3 make_task v2 $(date) ==="
python -u make_task_from_unimol.py --task bbbp --mode v2
echo "--- processed: ---"; ls -la down_task_bbbp_v2/processed/
echo "BBBP_REGEN_DONE $(date)"
