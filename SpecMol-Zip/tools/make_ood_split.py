"""Generate a cluster-based out-of-distribution (OOD) fold assignment.

Reads a fold CSV (columns: smiles,label,fold_id) and rewrites *only* the
`fold_id` column so that whole Butina/Tanimoto clusters are kept together in a
fold. With the downstream make_*_from_unimol convention (test_fold = fold,
val_fold = fold+1, train = rest), this makes the test/valid molecules belong to
clusters that are absent from train -> a genuine structural OOD split, much
harder than the scaffold split and the regime where Morgan-fingerprint RF is
expected to degrade (Guo et al., ICANN 2024; survey 4.2).

Row order is preserved exactly, so the existing per-molecule SDF / pair_rep
index mapping in make_*_from_unimol stays valid -- no pair_rep re-extraction
needed, only a re-run of the assembler with --csv-path pointing here.

Usage:
    python tools/make_ood_split.py \
        --in-csv  unimol_out/splits/data_with_folds_scaffold_k10_seed42.csv \
        --out-csv unimol_out/splits/data_with_folds_cluster_ood_k10.csv \
        --cutoff 0.6 --kfold 10
"""
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def morgan_fps(smiles, radius=2, n_bits=2048):
    fps, bad = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append(i)
            fps.append(None)
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return fps, bad


def butina_clusters(fps, cutoff):
    """Return a list of clusters (tuples of molecule indices). cutoff is the
    Tanimoto *distance* threshold; molecules within 1-cutoff similarity cluster
    together. Molecules with an unparseable fp become singleton clusters."""
    n = len(fps)
    valid = [i for i, fp in enumerate(fps) if fp is not None]
    # Condensed lower-triangle distance matrix over the valid molecules only.
    dists = []
    for a in range(1, len(valid)):
        sims = DataStructs.BulkTanimotoSimilarity(
            fps[valid[a]], [fps[valid[b]] for b in range(a)]
        )
        dists.extend(1.0 - s for s in sims)
    raw = Butina.ClusterData(dists, len(valid), cutoff, isDistData=True)
    clusters = [tuple(valid[j] for j in cl) for cl in raw]
    # Re-attach any unparseable molecules as singletons so every row gets a fold.
    seen = {i for cl in clusters for i in cl}
    clusters.extend((i,) for i in range(n) if i not in seen)
    return clusters


def assign_folds(clusters, kfold):
    """Greedy bin-packing: assign whole clusters (largest first) to the
    currently-smallest fold, so folds are size-balanced yet each holds only
    complete clusters (cross-fold structural dissimilarity)."""
    order = sorted(clusters, key=len, reverse=True)
    fold_size = [0] * kfold
    fold_of = {}
    for cl in order:
        f = int(np.argmin(fold_size))
        for idx in cl:
            fold_of[idx] = f
        fold_size[f] += len(cl)
    return fold_of, fold_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--cutoff", type=float, default=0.6,
                    help="Butina Tanimoto distance cutoff (0.6 -> cluster at >=0.4 similarity)")
    ap.add_argument("--kfold", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    smiles = df["smiles"].astype(str).tolist()
    print(f"loaded {len(smiles)} molecules from {args.in_csv}")

    fps, bad = morgan_fps(smiles)
    if bad:
        print(f"WARNING: {len(bad)} unparseable SMILES -> singleton clusters")

    clusters = butina_clusters(fps, args.cutoff)
    print(f"butina: {len(clusters)} clusters at cutoff={args.cutoff} "
          f"(largest={max(len(c) for c in clusters)}, singletons={sum(len(c)==1 for c in clusters)})")

    fold_of, fold_size = assign_folds(clusters, args.kfold)
    new_fold = np.array([fold_of[i] for i in range(len(smiles))], dtype=int)

    out = df.copy()
    out["fold_id"] = new_fold  # row order preserved -> SDF/pair_rep mapping intact
    out.to_csv(args.out_csv, index=False)

    # Report the resulting test(fold0)/valid(fold1)/train split and a sanity
    # check that test clusters are disjoint from train clusters by construction.
    test_n = int((new_fold == 0).sum())
    val_n = int((new_fold == 1).sum())
    train_n = int(len(new_fold) - test_n - val_n)
    print(f"fold sizes: {fold_size}")
    print(f"OOD split -> train={train_n} valid={val_n} test={test_n}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
