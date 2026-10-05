"""Data-pipeline audit: duplicate molecules, train/test leakage, and scaffold
train/test overlap across the matched MoleculeNet splits used in this study.

CPU-only, no GPU, no Uni-Mol weights. Reuses baselines_matched.DATASETS /
load_split so the splits are the EXACT ones the deep models and matched
baselines used. Quantifies the data-quality / data-preparation side of the
pipeline (a data-engineering contribution, independent of any model).

Pre-registered reading (decision aid, not a gate):
  - n_redundant_copies / dup_fraction: exact-canonical duplicate molecules.
  - n_train_test_leaked_smiles: identical molecule present in BOTH train and
    test -> direct leakage; any value > 0 is reportable.
  - frac_test_molecules_scaffold_in_train: how "out-of-distribution" the
    scaffold split really is (1.0 = every test scaffold already seen in train).

Output: audit_dedup_leakage.json + console summary.
Run (HPC, specmol env):  python audit_dedup_leakage.py
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from baselines_matched import DATASETS, load_split
from baselines_ml import REPO

RDLogger.DisableLog("rdApp.*")


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def scaffold(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return None


def audit(name, cfg):
    smi, y, is_train, is_test = load_split(cfg)
    n = len(smi)
    can = [canon(s) for s in smi]
    valid = [c for c in can if c is not None]

    # exact duplicates (canonical) over the whole dataset
    cnt = Counter(valid)
    dup_groups = {k: v for k, v in cnt.items() if v > 1}
    n_redundant = sum(v - 1 for v in dup_groups.values())

    # train/test leakage: canonical SMILES present in BOTH train and test
    tr_can = {c for c, t in zip(can, is_train) if t and c}
    te_can = {c for c, t in zip(can, is_test) if t and c}
    leak = sorted(tr_can & te_can)

    # scaffold overlap: how much of the test is structurally "seen" in train
    tr_scaf = {scaffold(s) for s, t in zip(smi, is_train) if t}
    tr_scaf.discard(None)
    te_scaf_list = [scaffold(s) for s, t in zip(smi, is_test) if t]
    te_scaf_list = [x for x in te_scaf_list if x]
    te_scaf = set(te_scaf_list)
    frac_test_scaf_in_train = (len(te_scaf & tr_scaf) / len(te_scaf)) if te_scaf else None
    frac_test_mol_scaf_in_train = (
        float(np.mean([1.0 if x in tr_scaf else 0.0 for x in te_scaf_list]))
        if te_scaf_list else None)

    res = dict(
        dataset=name, n=n, n_valid=len(valid), n_unique=len(set(valid)),
        n_duplicate_molecule_groups=len(dup_groups), n_redundant_copies=n_redundant,
        dup_fraction=round(n_redundant / len(valid), 4) if valid else None,
        n_train=int(is_train.sum()), n_test=int(is_test.sum()),
        n_train_test_leaked_smiles=len(leak), leaked_smiles_examples=leak[:5],
        n_train_scaffolds=len(tr_scaf), n_test_scaffolds=len(te_scaf),
        frac_test_scaffolds_seen_in_train=(round(frac_test_scaf_in_train, 4)
                                           if frac_test_scaf_in_train is not None else None),
        frac_test_molecules_scaffold_in_train=(round(frac_test_mol_scaf_in_train, 4)
                                               if frac_test_mol_scaf_in_train is not None else None),
        split_csv=cfg["split_csv"])
    print(f"[{name}] n={n} uniq={res['n_unique']} dup_groups={len(dup_groups)} "
          f"redundant={n_redundant} (frac={res['dup_fraction']}) leak={len(leak)} "
          f"test_scaf_in_train={res['frac_test_scaffolds_seen_in_train']} "
          f"test_mol_scaf_in_train={res['frac_test_molecules_scaffold_in_train']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out", default=str(REPO / "audit_dedup_leakage.json"))
    a = ap.parse_args()
    results = {}
    for name in a.datasets:
        cfg = DATASETS.get(name)
        if not cfg:
            print(f"skip unknown dataset {name}")
            continue
        if not (REPO / cfg["split_csv"]).exists():
            print(f"[{name}] split CSV missing ({cfg['split_csv']}); skipping")
            continue
        try:
            results[name] = audit(name, cfg)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
        Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
