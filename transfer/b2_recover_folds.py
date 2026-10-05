"""B2 robust fold-CSV recovery WITHOUT DataHub (which crashes on dropped SMILES).

The deep model's split lives in down_task_<task>_unifold/processed/{train,valid,test}.pt.
Each Data carries mol_id = row index into the post-clean fold ordering, where the
post-clean ordering = the prepared CSV (unimol raw smiles_unimol.csv) with illegal
SMILES removed in place. Uni-Mol's only filter is `Chem.MolFromSmiles(smi) is None`
(see unimol_tools datareader.check_smiles), so we reproduce the 2039-row ordering
with RDKit alone -- deterministic, CPU, no DataHub.

We then validate the recovery against the .pt labels (label[mol_id] must equal the
.pt y for every molecule in every split) and emit a data_with_folds CSV with
fold_id 0 = test (the deep model's test fold), 1 = everything else (val+train),
which is exactly what baselines_matched.py consumes (fid != 0 -> train).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

REPO = Path.home() / "zhoutianyang/SpecMol/SpecMol-Zip"

CASES = {
    "bbbp": dict(
        prepared=REPO / "unimol_out_bbbp/raw/smiles_unimol.csv",
        proc=REPO / "down_task_bbbp_unifold/processed",
        prefix="bbbp",
        out_csv=REPO / "unimol_out_bbbp/splits/data_with_folds_scaffold_k10_seed42.csv",
    ),
    "bace": dict(
        prepared=REPO / "unimol_out/raw/smiles_unimol.csv",
        proc=REPO / "down_task_bace_unifold/processed",
        prefix="bace",
        out_csv=REPO / "unimol_out/splits/data_with_folds_scaffold_k10_seed42.csv",
    ),
}


def load_split(proc, prefix, split):
    o = torch.load(proc / f"{prefix}_{split}.pt", map_location="cpu")
    d = o[0] if isinstance(o, tuple) else o
    mid = d.mol_id.view(-1).tolist()
    y = d.y.view(len(mid), -1)[:, 0].tolist()  # first label col (single-label bbbp/bace)
    return [int(m) for m in mid], [float(v) for v in y]


def recover(name, cfg):
    prepared, proc, pre, out_csv = cfg["prepared"], cfg["proc"], cfg["prefix"], cfg["out_csv"]
    if not prepared.exists():
        print(f"[{name}] PREPARED CSV MISSING: {prepared}")
        return False
    df = pd.read_csv(prepared)
    smi_col = "smiles" if "smiles" in df.columns else df.columns[0]
    lab_col = "label" if "label" in df.columns else df.columns[1]
    raw_smiles = df[smi_col].astype(str).tolist()
    raw_label = df[lab_col].tolist()

    # Reproduce Uni-Mol's clean: keep rows whose SMILES parse, order preserved.
    keep_smiles, keep_label = [], []
    n_dropped = 0
    for s, l in zip(raw_smiles, raw_label):
        if Chem.MolFromSmiles(s) is not None:
            keep_smiles.append(s)
            keep_label.append(l)
        else:
            n_dropped += 1
    n = len(keep_smiles)
    print(f"[{name}] prepared={len(raw_smiles)} dropped={n_dropped} kept={n}")

    # Gather splits + assign fold_id (0=test, 1=val, 2=train).
    fold_of = {}
    all_ids = []
    pt_label = {}
    for split, fid in [("test", 0), ("valid", 1), ("train", 2)]:
        ids, ys = load_split(proc, pre, split)
        for m, y in zip(ids, ys):
            fold_of[m] = fid
            pt_label[m] = y
        all_ids.extend(ids)
        print(f"[{name}]   {split}: n={len(ids)} mol_id[min,max]=[{min(ids)},{max(ids)}]")

    # Validation 1: union of mol_ids exactly covers 0..n-1
    uniq = sorted(set(all_ids))
    cover_ok = (len(uniq) == n and uniq[0] == 0 and uniq[-1] == n - 1 and len(all_ids) == n)
    print(f"[{name}] coverage: n_pt_union={len(uniq)} kept={n} contiguous_0..n-1={cover_ok}")

    # Validation 2: recovered label[mol_id] == .pt y for every molecule
    mism = 0
    for m in uniq:
        if m < n and abs(float(keep_label[m]) - pt_label[m]) > 1e-6:
            mism += 1
    label_ok = (mism == 0)
    print(f"[{name}] label cross-check: mismatches={mism} ({'OK' if label_ok else 'FAIL'})")

    if not (cover_ok and label_ok):
        print(f"[{name}] RECOVERY FAILED -- not writing CSV")
        return False

    # Emit data_with_folds CSV (all n rows, fold_id per the deep split).
    fold_ids = [fold_of[i] for i in range(n)]
    out = pd.DataFrame({"smiles": keep_smiles, "label": keep_label, "fold_id": fold_ids})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    sizes = dict(zip(*np.unique(fold_ids, return_counts=True)))
    print(f"[{name}] WROTE {out_csv}  (rows={n}, fold sizes={sizes})")
    print(f"[{name}] PASS")
    return True


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    allok = True
    for nm in names:
        allok &= recover(nm, CASES[nm])
        print()
    print("ALL_PASS" if allok else "SOME_FAIL")
    sys.exit(0 if allok else 1)
