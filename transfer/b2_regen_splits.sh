#!/bin/bash
# B2: regenerate bbbp/bace Uni-Mol fold split CSVs (scaffold k10 seed42)
# Runs on ws-ia CPU; DataHub conformer gen is CPU-only.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/zhoutianyang/SpecMol/SpecMol-Zip
export UNIMOL_DICT=$HOME/miniconda3/envs/specmol/lib/python3.10/site-packages/unimol_tools/weights/mol.dict.txt

echo "### BBBP ###"
python export_unimol_splits.py --task bbbp --out-root unimol_out_bbbp
echo "### BACE ###"
python export_unimol_splits.py --task bace
echo "B2_SPLITS_DONE"
