#!/bin/bash
# B2: matched RF (Morgan+RDKit2D, 30 seeds) + Uni-Mol-direct on bbbp/bace fold0=test.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/zhoutianyang/SpecMol/SpecMol-Zip
export CUDA_VISIBLE_DEVICES=
export UNIMOL_CKPT=$HOME/zhoutianyang/SpecMol/SpecMol-Zip/unimol_weights/mol_pre_all_h_220816.pt
export UNIMOL_DICT=$HOME/miniconda3/envs/specmol/lib/python3.10/site-packages/unimol_tools/weights/mol.dict.txt

python baselines_matched.py --datasets bbbp bace \
    --features morgan_rdkit unimol --n_seeds 30 \
    --out baselines_matched_bbbp_bace.json
echo "B2_BASELINES_DONE rc=$?"
