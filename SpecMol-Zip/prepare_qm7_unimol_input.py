"""Prepare QM7 SDF + manifest for Uni-Mol pair_repr extraction (RDKit-conformer route).

Mirror of prepare_freesolv_unimol_input.py for QM7 (atomization energy, u0_atom).

Design choice (see plan): we use RDKit ETKDG conformers (NOT the QM7 ground-truth
DFT geometry), for three reasons: (1) it is the SAME geometry source as the ADMET
experiments, so the ADMET-vs-QM contrast is a clean controlled comparison where only
the *task* changes; (2) RDKit conformers are in-distribution for Uni-Mol (pretrained
on RDKit conformers), whereas exact DFT geometry is OOD; (3) it reuses the tested
FreeSolv pipeline and -- crucially -- avoids the fragile SDF<->csv positional-concat
alignment trap in DeepChem's gdb7.sdf loader (the SDF here is embedded FROM the same
SMILES that builds the 2D graph, so atom order is aligned by construction).
The exact-DFT-geometry path remains an optional later ablation.

Source: qm7.csv (columns: smiles, u0_atom), self-aligned -- each row is one
(smiles, target) pair, so there is no SDF<->csv alignment to get wrong.

Outputs (under unimol_out_qm7/):
  - sdf/qm7_for_unimol.sdf          : RDKit 3D conformers in dataset order
  - manifest.csv                   : smiles, label, split, sdf_index, embed_ok
  - splits/qm7_split_seed42.csv     : mirror
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import deepchem as dc
from deepchem.splits import RandomSplitter

RDLogger.logger().setLevel(RDLogger.CRITICAL)

REPO = Path(__file__).resolve().parent
QM7_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv"
RAW_CSV = REPO / "dataset" / "qm7" / "raw" / "smiles.csv"
OUT_ROOT = REPO / "unimol_out_qm7"
SDF_DIR = OUT_ROOT / "sdf"
SPLITS_DIR = OUT_ROOT / "splits"
MANIFEST_CSV = OUT_ROOT / "manifest.csv"
SDF_NAME = "qm7_for_unimol.sdf"


def load_smiles_labels():
    """Load (smiles, u0_atom) from qm7.csv; cache a header-less raw copy.

    qm7.csv is self-aligned (smiles,u0_atom per row) -> no SDF<->csv merge risk.
    We drop molecules that RDKit cannot parse or that are disconnected ('.'),
    since the 2D graph + fingerprint pipeline assumes a single connected molecule.
    """
    if RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV, header=None)
        smiles = [str(s) for s in df.iloc[:, 0].tolist()]
        labels = df.iloc[:, 1].values.astype(np.float32)
        return smiles, labels

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    local_csv = OUT_ROOT / "qm7.csv"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not local_csv.exists():
        print(f"Downloading {QM7_CSV_URL} ...")
        urllib.request.urlretrieve(QM7_CSV_URL, str(local_csv))
    raw = pd.read_csv(local_csv)  # header: smiles,u0_atom
    assert list(raw.columns[:2]) == ["smiles", "u0_atom"], raw.columns.tolist()

    keep_s, keep_l = [], []
    n_bad = n_frag = 0
    for s, l in zip(raw["smiles"].astype(str), raw["u0_atom"].values):
        m = Chem.MolFromSmiles(s)
        if m is None:
            n_bad += 1
            continue
        cs = Chem.MolToSmiles(m)
        if "." in cs:                      # disconnected fragment / radical complex
            n_frag += 1
            continue
        keep_s.append(cs)                  # canonical, connected
        keep_l.append(float(l))
    labels = np.array(keep_l, dtype=np.float32)
    pd.DataFrame({"smiles": keep_s, "label": labels}).to_csv(
        RAW_CSV, index=False, header=False
    )
    print(f"QM7: kept {len(keep_s)} (dropped {n_bad} unparseable, {n_frag} disconnected)"
          f" -> {RAW_CSV}")
    return keep_s, labels


def random_split(smiles, labels, seed=42):
    """RandomSplitter (QM convention; scaffold split is ill-posed for tiny QM mols)."""
    y = labels.reshape(-1, 1)
    fps = np.zeros((len(smiles), 1))
    w = np.ones((len(smiles), 1))
    ds = dc.data.NumpyDataset(X=fps, y=y, w=w, ids=np.array(smiles))
    return RandomSplitter().train_valid_test_split(
        ds, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
    )


def embed_mol_3d(smile):
    """RDKit ETKDG + MMFF, identical to the FreeSolv route."""
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
    train_ds, val_ds, test_ds = random_split(smiles, labels, seed=42)
    s2split = {}
    for s in train_ds.ids:
        s2split[str(s)] = "train"
    for s in val_ds.ids:
        s2split[str(s)] = "valid"
    for s in test_ds.ids:
        s2split[str(s)] = "test"

    writer = Chem.SDWriter(str(SDF_DIR / SDF_NAME))
    rows = []
    n_fail = 0
    sdf_index = 0
    for smile, label in zip(smiles, labels):
        split = s2split.get(smile)
        mol_h = embed_mol_3d(smile)
        if mol_h is None:
            n_fail += 1
            rows.append({"smiles": smile, "label": float(label), "split": split,
                         "sdf_index": -1, "embed_ok": False})
            continue
        mol_h.SetProp("_Name", smile)
        writer.write(mol_h)
        rows.append({"smiles": smile, "label": float(label), "split": split,
                     "sdf_index": sdf_index, "embed_ok": True})
        sdf_index += 1
    writer.close()

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_CSV, index=False)
    manifest.to_csv(SPLITS_DIR / "qm7_split_seed42.csv", index=False)

    n_ok = int(manifest["embed_ok"].sum())
    print(f"QM7: {len(manifest)} mols total, {n_ok} embedded, {n_fail} skipped")
    print(f"SDF: {SDF_DIR / SDF_NAME}")
    print("Splits: train={} valid={} test={}".format(
        int((manifest['split'] == 'train').sum()),
        int((manifest['split'] == 'valid').sum()),
        int((manifest['split'] == 'test').sum())))


if __name__ == "__main__":
    main()
