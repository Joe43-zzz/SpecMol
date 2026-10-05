"""Prepare {esol, lipo, freesolv} SDF + manifest for Uni-Mol pair_repr extraction.

Generic version of prepare_freesolv_unimol_input.py. Routes by --task.

The FreeSolv pipeline (prepare_freesolv_unimol_input.py + make_freesolv_unimol_data.py
+ run_freesolv_v2t5.py) is preserved as-is for backward compatibility. New regression
tasks (ESOL, Lipo) flow through this generic script.

Usage:
    python prepare_regression_unimol_input.py --task esol
    python prepare_regression_unimol_input.py --task lipo

Outputs (under unimol_out_{task}/):
  - sdf/{task}_for_unimol.sdf         : 3D conformers in dataset order
  - manifest.csv                       : smiles, label, split, sdf_index, embed_ok
  - splits/{task}_split_seed42.csv     : index, smiles, label, split (mirror)
"""

import argparse
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
from deepchem.splits import ScaffoldSplitter
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.CRITICAL)


REPO = Path(__file__).resolve().parent


TASK_CONFIG = {
    "freesolv": {
        "loader": dc.molnet.load_freesolv,
        "label_dim": 1,
    },
    "esol": {
        "loader": dc.molnet.load_delaney,
        "label_dim": 1,
    },
    "lipo": {
        "loader": dc.molnet.load_lipo,
        "label_dim": 1,
    },
}


def load_smiles_labels(task, cfg):
    """Reuse cached CSV at dataset/{task}/raw/smiles.csv if present, else download."""
    csv_path = REPO / "dataset" / task / "raw" / "smiles.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, header=None)
        return df.iloc[:, 0].tolist(), df.iloc[:, 1].values.astype(np.float32)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tasks, datasets, _ = cfg["loader"](featurizer="Raw", splitter=None)
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
    print(f"[prep:{task}] saved {len(keep_s)} mols to {csv_path}")
    return keep_s, keep_l


def scaffold_split(smiles, labels, seed=42):
    fps = np.zeros((len(smiles), 1))
    w = np.ones((len(smiles), 1))
    y = labels.reshape(-1, 1)
    dataset = dc.data.NumpyDataset(X=fps, y=y, w=w, ids=np.array(smiles))
    return ScaffoldSplitter().train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
    )


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIG))
    args = parser.parse_args()
    task = args.task
    cfg = TASK_CONFIG[task]

    out_root = REPO / f"unimol_out_{task}"
    sdf_dir = out_root / "sdf"
    splits_dir = out_root / "splits"
    manifest_csv = out_root / "manifest.csv"
    sdf_name = f"{task}_for_unimol.sdf"
    sdf_path = sdf_dir / sdf_name

    out_root.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    smiles, labels = load_smiles_labels(task, cfg)
    train_ds, val_ds, test_ds = scaffold_split(smiles, labels, seed=42)

    s2split = {}
    for s in train_ds.ids:
        s2split[str(s)] = "train"
    for s in val_ds.ids:
        s2split[str(s)] = "valid"
    for s in test_ds.ids:
        s2split[str(s)] = "test"

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
    manifest.to_csv(manifest_csv, index=False)
    manifest.to_csv(splits_dir / f"{task}_split_seed42.csv", index=False)

    n_total = len(manifest)
    n_ok = int(manifest["embed_ok"].sum())
    print(f"[prep:{task}] {n_total} mols total, {n_ok} embedded, {n_embed_fail} skipped")
    print(f"[prep:{task}] SDF written: {sdf_path}")
    print(f"[prep:{task}] Manifest:    {manifest_csv}")
    print(
        f"[prep:{task}] Splits in manifest: "
        f"train={int((manifest['split']=='train').sum())} "
        f"valid={int((manifest['split']=='valid').sum())} "
        f"test={int((manifest['split']=='test').sum())}"
    )


if __name__ == "__main__":
    main()
