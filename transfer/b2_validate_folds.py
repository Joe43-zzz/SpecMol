"""B2 validation: confirm the regenerated fold CSV reproduces the EXACT
split baked into the deep model's down_task_*_unifold/processed/*.pt.

The .pt files store data.mol_id = row index into the fold CSV's smiles list
(see make_*_from_unimol.py: data.mol_id = int(mol_index)). If the regenerated
data_with_folds CSV is row-consistent with the original, then:
  - the set of mol_ids in *_test.pt  == rows where fold_id == 0
  - the set of mol_ids in *_valid.pt == rows where fold_id == 1
  - the set of mol_ids in *_train.pt == rows where fold_id in 2..9
A pass proves mol_id -> smiles recovery is correct and RF/Uni-Mol baselines on
fold0 are scored on the identical molecules the deep model tested on.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO = Path.home() / "zhoutianyang/SpecMol/SpecMol-Zip"

CASES = {
    "bbbp": dict(
        proc=REPO / "down_task_bbbp_unifold/processed",
        prefix="bbbp",
        csv=REPO / "unimol_out_bbbp/splits/data_with_folds_scaffold_k10_seed42.csv",
    ),
    "bace": dict(
        proc=REPO / "down_task_bace_unifold/processed",
        prefix="bace",
        csv=REPO / "unimol_out/splits/data_with_folds_scaffold_k10_seed42.csv",
    ),
}


def mol_ids_from_pt(pt_path):
    obj = torch.load(pt_path, map_location="cpu")
    data = obj[0] if isinstance(obj, tuple) else obj
    mid = data.mol_id
    if torch.is_tensor(mid):
        mid = mid.view(-1).tolist()
    return sorted(int(x) for x in mid)


def check(name, cfg):
    proc, pre, csv = cfg["proc"], cfg["prefix"], cfg["csv"]
    if not csv.exists():
        print(f"[{name}] FOLD CSV MISSING: {csv}")
        return False
    df = pd.read_csv(csv)
    fid = df["fold_id"].astype(int).to_numpy()
    nrows = len(df)
    rows_fold0 = set(np.where(fid == 0)[0].tolist())
    rows_fold1 = set(np.where(fid == 1)[0].tolist())
    rows_train = set(np.where(fid >= 2)[0].tolist())

    ok = True
    print(f"[{name}] csv rows={nrows}  fold sizes={dict(zip(*np.unique(fid, return_counts=True)))}")
    for split, expect in [("test", rows_fold0), ("valid", rows_fold1), ("train", rows_train)]:
        pt = proc / f"{pre}_{split}.pt"
        if not pt.exists():
            print(f"  [{split}] .pt MISSING: {pt}")
            ok = False
            continue
        ids = set(mol_ids_from_pt(pt))
        max_id = max(ids) if ids else -1
        match = ids == expect
        ok = ok and match
        extra = ids - expect
        missing = expect - ids
        print(f"  [{split}] n_pt={len(ids)} n_csv_fold={len(expect)} match={match} "
              f"max_mol_id={max_id} (<{nrows}? {max_id < nrows}) "
              + ("" if match else f"extra={len(extra)} missing={len(missing)}"))
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    allok = True
    for n in names:
        allok &= check(n, CASES[n])
        print()
    print("ALL_PASS" if allok else "SOME_FAIL")
    sys.exit(0 if allok else 1)
