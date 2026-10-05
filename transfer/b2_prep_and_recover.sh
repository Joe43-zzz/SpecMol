#!/bin/bash
# B2: prepare bace CSV (no DataHub) + recover bbbp/bace fold CSVs from .pt mol_ids.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specmol
cd ~/zhoutianyang/SpecMol/SpecMol-Zip

echo "### prepare-only bace (writes unimol_out/raw/smiles_unimol.csv, no DataHub) ###"
python export_unimol_splits.py --task bace --prepare-only

echo "### recover fold CSVs from .pt ###"
python hpc/b2_recover_folds.py
echo "B2_RECOVER_DONE rc=$?"
