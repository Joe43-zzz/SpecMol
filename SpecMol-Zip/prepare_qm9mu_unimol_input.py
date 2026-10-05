"""Prepare a QM9 dipole-moment (mu) SUBSET for Uni-Mol pair_repr extraction.

Geometry-pure target: unlike QM7 atomization energy (composition-dominated),
the dipole moment mu is genuinely a function of the 3D charge distribution, so
it is the right target to test whether the 3D-injection geometry channel carries
usable signal (cf. the QM7 control finding: T7 gain was pure architecture, T8 had
only a ~3.5 RMSE geometry effect on the composition-dominated atomization energy).

Same RDKit-conformer route as prepare_qm7 (fair vs ADMET; Uni-Mol-in-distribution;
no SDF<->csv alignment trap -- SDF embedded from the same SMILES that builds the graph).

Subset: random SUBSAMPLE_N molecules (full QM9 is 134k; a subset is enough for the
go/no-go geometry test, scale up only if positive).

Source: qm9.csv columns -> mol_id, smiles, A, B, C, mu, alpha, homo, lumo, gap, ...
We use 'smiles' and 'mu'.

Outputs (under unimol_out_qm9mu/): sdf/qm9mu_for_unimol.sdf, manifest.csv, splits/.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import deepchem as dc
from deepchem.splits import RandomSplitter

RDLogger.logger().setLevel(RDLogger.CRITICAL)

REPO = Path(__file__).resolve().parent
QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
TARGET_COL = "mu"          # dipole moment (Debye); geometry-pure
SUBSAMPLE_N = 15000
SUBSAMPLE_SEED = 42
RAW_CSV = REPO / "dataset" / "qm9mu" / "raw" / "smiles.csv"
OUT_ROOT = REPO / "unimol_out_qm9mu"
SDF_DIR = OUT_ROOT / "sdf"
SPLITS_DIR = OUT_ROOT / "splits"
MANIFEST_CSV = OUT_ROOT / "manifest.csv"
SDF_NAME = "qm9mu_for_unimol.sdf"


def load_smiles_labels():
    if RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV, header=None)
        return [str(s) for s in df.iloc[:, 0]], df.iloc[:, 1].values.astype(np.float32)

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    local_csv = OUT_ROOT / "qm9.csv"
    if not local_csv.exists():
        print(f"Downloading {QM9_CSV_URL} (~30MB) ...")
        urllib.request.urlretrieve(QM9_CSV_URL, str(local_csv))
    raw = pd.read_csv(local_csv)
    assert "smiles" in raw.columns and TARGET_COL in raw.columns, raw.columns.tolist()

    # subsample BEFORE expensive embedding
    if SUBSAMPLE_N and len(raw) > SUBSAMPLE_N:
        raw = raw.sample(n=SUBSAMPLE_N, random_state=SUBSAMPLE_SEED).reset_index(drop=True)
        print(f"Subsampled QM9 to {len(raw)} (seed {SUBSAMPLE_SEED})")

    keep_s, keep_l = [], []
    n_bad = n_frag = 0
    for s, l in zip(raw["smiles"].astype(str), raw[TARGET_COL].values):
        m = Chem.MolFromSmiles(s)
        if m is None:
            n_bad += 1
            continue
        cs = Chem.MolToSmiles(m)
        if "." in cs:
            n_frag += 1
            continue
        keep_s.append(cs)
        keep_l.append(float(l))
    labels = np.array(keep_l, dtype=np.float32)
    pd.DataFrame({"smiles": keep_s, "label": labels}).to_csv(RAW_CSV, index=False, header=False)
    print(f"QM9-{TARGET_COL}: kept {len(keep_s)} (dropped {n_bad} bad, {n_frag} disconnected) -> {RAW_CSV}")
    return keep_s, labels


def random_split(smiles, labels, seed=42):
    y = labels.reshape(-1, 1)
    ds = dc.data.NumpyDataset(X=np.zeros((len(smiles), 1)), y=y,
                              w=np.ones((len(smiles), 1)), ids=np.array(smiles))
    return RandomSplitter().train_valid_test_split(
        ds, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)


def embed_mol_3d(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, randomSeed=42) == -1:
        if AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=42) == -1:
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
        mol_h = embed_mol_3d(smile)
        if mol_h is None:
            n_fail += 1
            rows.append({"smiles": smile, "label": float(label),
                         "split": s2split.get(smile), "sdf_index": -1, "embed_ok": False})
            continue
        mol_h.SetProp("_Name", smile)
        writer.write(mol_h)
        rows.append({"smiles": smile, "label": float(label),
                     "split": s2split.get(smile), "sdf_index": sdf_index, "embed_ok": True})
        sdf_index += 1
    writer.close()

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_CSV, index=False)
    manifest.to_csv(SPLITS_DIR / "qm9mu_split_seed42.csv", index=False)
    print(f"QM9-{TARGET_COL}: {len(manifest)} total, {int(manifest['embed_ok'].sum())} embedded, {n_fail} skipped")
    print("Splits: train={} valid={} test={}".format(
        int((manifest['split'] == 'train').sum()),
        int((manifest['split'] == 'valid').sum()),
        int((manifest['split'] == 'test').sum())))


if __name__ == "__main__":
    main()
