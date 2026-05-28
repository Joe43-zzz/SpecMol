"""Prepare FreeSolv SDF + manifest for Uni-Mol pair_repr extraction.

Goal: produce the SDF/CSV inputs needed by tools/extract_unimol_pair.py so
that we can compute Uni-Mol encoder pair_repr for FreeSolv molecules and
build a `down_task_freesolv_unimol_v2/` dataset that is directly comparable
to the existing `down_task_freesolv_v2/` (RDKit 3D + GBF).

Reuses the *same* canonical SMILES list and DeepChem ScaffoldSplitter
(seed=42) as `prepare_freesolv_data.py`, so the resulting train/valid/test
split is identical molecule-by-molecule. Only the per-edge pair_repr will
differ downstream (GBF -> Uni-Mol encoder_pair_rep).

Outputs (under unimol_out_freesolv/):
  - sdf/freesolv_for_unimol.sdf     : 3D conformers in dataset order
  - manifest.csv                    : smiles, label, split, sdf_index, embed_ok
  - splits/freesolv_split_seed42.csv: index, smiles, label, split (mirror)

The SDF index column is the row index in the dataset *as written*, i.e.
the integer used by `tools/extract_unimol_pair.py` to name `000123/pair_rep.pt`.
Molecules that fail RDKit embedding are skipped from the SDF; the manifest
records `embed_ok=False` so downstream builders can skip them as well.
"""

import os
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
import torch
from deepchem.splits import ScaffoldSplitter

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.CRITICAL)


REPO = Path(__file__).resolve().parent
OUT_ROOT = REPO / "unimol_out_freesolv"
SDF_DIR = OUT_ROOT / "sdf"
SPLITS_DIR = OUT_ROOT / "splits"
MANIFEST_CSV = OUT_ROOT / "manifest.csv"
SDF_NAME = "freesolv_for_unimol.sdf"


def load_smiles_labels():
    """Mirror download_freesolv() in prepare_freesolv_data.py."""
    csv_path = REPO / "dataset" / "freesolv" / "raw" / "smiles.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, header=None)
        return df.iloc[:, 0].tolist(), df.iloc[:, 1].values.astype(np.float32)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tasks, datasets, _ = dc.molnet.load_freesolv(featurizer="Raw", splitter=None)
    full_ds = datasets[0]
    smiles = [str(s) for s in full_ds.ids]
    labels = full_ds.y[:, 0].astype(np.float32)

    keep_s, keep_l = [], []
    for s, l in zip(smiles, labels):
        if Chem.MolFromSmiles(s) is not None:
            keep_s.append(s)
            keep_l.append(l)
    keep_l = np.array(keep_l, dtype=np.float32)

    pd.DataFrame({"smiles": keep_s, "label": keep_l}).to_csv(
        csv_path, index=False, header=False
    )
    print(f"Saved {len(keep_s)} molecules to {csv_path}")
    return keep_s, keep_l


def scaffold_split(smiles, labels, seed=42):
    """Identical to prepare_freesolv_data.scaffold_split."""
    fps = np.zeros((len(smiles), 1))
    w = np.ones((len(smiles), 1))
    y = labels.reshape(-1, 1)
    dataset = dc.data.NumpyDataset(X=fps, y=y, w=w, ids=np.array(smiles))
    splitter = ScaffoldSplitter()
    return splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
    )


def embed_mol_3d(smile):
    """Generate 3D coords with RDKit ETKDG + MMFF (same as prepare_freesolv_data)."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
    if result == -1:
        result = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=42)
        if result == -1:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    return mol_h


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SDF_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    smiles, labels = load_smiles_labels()
    train_ds, val_ds, test_ds = scaffold_split(smiles, labels, seed=42)

    s2split = {}
    for s in train_ds.ids:
        s2split[str(s)] = "train"
    for s in val_ds.ids:
        s2split[str(s)] = "valid"
    for s in test_ds.ids:
        s2split[str(s)] = "test"

    sdf_path = SDF_DIR / SDF_NAME
    writer = Chem.SDWriter(str(sdf_path))

    rows = []
    n_embed_fail = 0
    sdf_index = 0
    for smile, label in zip(smiles, labels):
        split = s2split.get(smile)
        mol_h = embed_mol_3d(smile)
        if mol_h is None:
            n_embed_fail += 1
            rows.append({
                "smiles": smile,
                "label": float(label),
                "split": split,
                "sdf_index": -1,
                "embed_ok": False,
            })
            continue

        mol_h.SetProp("_Name", smile)
        writer.write(mol_h)
        rows.append({
            "smiles": smile,
            "label": float(label),
            "split": split,
            "sdf_index": sdf_index,
            "embed_ok": True,
        })
        sdf_index += 1

    writer.close()

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_CSV, index=False)
    manifest.to_csv(SPLITS_DIR / "freesolv_split_seed42.csv", index=False)

    n_total = len(manifest)
    n_ok = int(manifest["embed_ok"].sum())
    print(f"FreeSolv: {n_total} mols total, {n_ok} embedded, {n_embed_fail} skipped")
    print(f"SDF written: {sdf_path}")
    print(f"Manifest:    {MANIFEST_CSV}")
    print(
        "Splits  (in manifest): "
        f"train={int((manifest['split']=='train').sum())} "
        f"valid={int((manifest['split']=='valid').sum())} "
        f"test={int((manifest['split']=='test').sum())}"
    )


if __name__ == "__main__":
    main()
