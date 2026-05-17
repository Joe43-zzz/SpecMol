"""RF + RDKit2D baseline for SpecMol ICDE submission (L1).

Per Deng et al. Nat. Commun. 2023, a 300-tree random forest on Morgan FP
(1024 bits, radius 2) concatenated with RDKit 2D descriptors statistically
ties pretrained graph encoders on BACE/BBBP/FreeSolv. We replicate that
baseline so we can report it as Table I in the paper.

Output: baselines_ml_results.json with mean+/-std over 5 seeds per dataset.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
)

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent

# 11 RDKit 2D descriptors per Deng et al. 2023 (MW, LogP, TPSA-like core set).
RDKIT2D_DESCRIPTORS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumAliphaticRings",
    "HeavyAtomCount",
    "FractionCSP3",
    "RingCount",
]

DATASETS = {
    "bace": {
        "csv": "dataset/bace.csv",
        "smiles_col": "smiles",
        "label_col": "Class",
        "split_col": None,  # CSV's Model col is 80% Test; use seeded 80/20 random
        "task": "classification",
        "unimol_fold_csv": "unimol_out/splits/data_with_folds_scaffold_k10_seed42.csv",
    },
    "bbbp": {
        "csv": "dataset/BBBP.csv",
        "smiles_col": "smiles",
        "label_col": "p_np",
        "split_col": None,  # use random split
        "task": "classification",
    },
    "freesolv": {
        "csv": "dataset/freesolv/raw/smiles.csv",
        "smiles_col": 0,  # no header
        "label_col": 1,
        "split_col": None,
        "task": "regression",
    },
}

SEEDS = [9, 19, 29, 39, 49]
# Deng et al. (Nat. Commun. 2023) 30-split protocol uses seeds 0..29 sequentially.
# Used when --n_seeds 30 is passed on the CLI (default keeps SEEDS for backward
# compat with previously generated JSON files).
SEEDS_DENG30 = list(range(30))


def morgan_fp(mol, n_bits=1024, radius=2):
    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits),
        dtype=np.int8,
    )


def rdkit2d(mol):
    fns = {n: getattr(Descriptors, n) for n in RDKIT2D_DESCRIPTORS}
    return np.array([fns[n](mol) for n in RDKIT2D_DESCRIPTORS], dtype=np.float32)


def featurize(smiles_list):
    """Returns (X, ok) where ok is per-input mask (len == len(smiles_list))."""
    fps, descs, ok = [], [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            ok.append(False)
            continue
        fps.append(morgan_fp(m))
        descs.append(rdkit2d(m))
        ok.append(True)
    X_fp = np.vstack(fps) if fps else np.zeros((0, 1024), dtype=np.int8)
    X_desc = np.vstack(descs) if descs else np.zeros((0, 11), dtype=np.float32)
    X_desc = np.nan_to_num(X_desc, nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([X_fp, X_desc]).astype(np.float32), np.array(ok, dtype=bool)


def load_dataset(name, cfg):
    csv = REPO / cfg["csv"]
    if isinstance(cfg["smiles_col"], int):
        df = pd.read_csv(csv, header=None)
        smi = df.iloc[:, cfg["smiles_col"]].astype(str).tolist()
        y = df.iloc[:, cfg["label_col"]].astype(float).to_numpy()
        split = None
    else:
        df = pd.read_csv(csv)
        smi = df[cfg["smiles_col"]].astype(str).tolist()
        y = df[cfg["label_col"]].astype(float).to_numpy()
        split = (
            df[cfg["split_col"]].astype(str).to_numpy()
            if cfg["split_col"]
            else None
        )
    return smi, y, split


def _murcko(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smi, includeChirality=False)
    except Exception:
        return ""


def scaffold_kfold(smiles_list, seed=42, k=10):
    """Generate k=10 scaffold folds matching Uni-Mol DataHub protocol:
    Bemis-Murcko scaffolds grouped, sorted by descending size, greedy
    assignment to the smallest-occupancy fold. Returns per-sample fold_id.
    """
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        scaffolds.setdefault(_murcko(smi), []).append(i)
    rng = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    groups.sort(key=lambda g: (-len(g), rng.random()))
    n = len(smiles_list)
    folds = [[] for _ in range(k)]
    for g in groups:
        idx = int(np.argmin([len(f) for f in folds]))
        folds[idx].extend(g)
    fold_ids = np.zeros(n, dtype=int)
    for fid, members in enumerate(folds):
        for i in members:
            fold_ids[i] = fid
    return fold_ids


def scaffold_split(smiles_list, seed, frac_train=0.8, frac_test=0.2):
    """DeepChem-style Bemis-Murcko scaffold split. Largest scaffold groups
    go to train; rare scaffolds spread to test. Seed shuffles tie-breaking
    among same-sized groups."""
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        scf = _murcko(smi)
        scaffolds.setdefault(scf, []).append(i)
    rng = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    # sort by descending size, break ties via rng
    groups.sort(key=lambda g: (-len(g), rng.random()))
    n_total = sum(len(g) for g in groups)
    n_train_target = int(frac_train * n_total)
    tr, te = [], []
    for g in groups:
        if len(tr) + len(g) <= n_train_target:
            tr.extend(g)
        else:
            te.extend(g)
    return np.array(tr), np.array(te)


def split_indices(smiles_list, n, split_col, seed, split_mode="scaffold", unimol_fold_ids=None):
    """split_mode: 'scaffold' | 'random' | 'unimol_fold'.

    Uni-Mol fold protocol (matches model split in make_bace_ablation.py):
        fold 0 -> test, fold 1 -> val, folds 2-9 -> train.
        Val is merged into train for RF (RF does not need a validation split).
        Split is FIXED across seeds; only RF's random_state varies.
    """
    rng = np.random.default_rng(seed)
    if split_mode == "unimol_fold":
        assert unimol_fold_ids is not None, "unimol_fold requires fold_ids"
        te = np.flatnonzero(unimol_fold_ids == 0)
        tr = np.flatnonzero(unimol_fold_ids != 0)
        return tr, te
    if split_col is not None:
        tr = np.flatnonzero(split_col == "Train")
        te = np.flatnonzero((split_col == "Test") | (split_col == "Valid"))
        rng.shuffle(tr)
        return tr, te
    if split_mode == "scaffold":
        return scaffold_split(smiles_list, seed)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    return perm[:cut], perm[cut:]


def fit_eval(X_tr, y_tr, X_te, y_te, task, seed):
    if task == "classification":
        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced",
        )
        clf.fit(X_tr, y_tr.astype(int))
        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
        }
    else:
        reg = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
        reg.fit(X_tr, y_tr)
        pred = reg.predict(X_te)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
            "mae": float(mean_absolute_error(y_te, pred)),
        }


def run_dataset(name, split_mode="scaffold", seeds=None):
    if seeds is None:
        seeds = SEEDS
    cfg = DATASETS[name]
    print(f"[{name}] loading {cfg['csv']}  split={split_mode}  n_seeds={len(seeds)}")
    smi, y, split = load_dataset(name, cfg)
    print(f"[{name}] n_raw={len(smi)} positive_rate={float(np.mean(y > 0)) if cfg['task']=='classification' else 'reg'}")

    unimol_fold_ids = None
    if split_mode == "unimol_fold":
        fold_csv = cfg.get("unimol_fold_csv")
        if fold_csv is not None and (REPO / fold_csv).exists():
            fold_df = pd.read_csv(REPO / fold_csv)
            assert "smiles" in fold_df.columns and "fold_id" in fold_df.columns, fold_df.columns.tolist()
            smi_to_fold = dict(zip(fold_df["smiles"].astype(str), fold_df["fold_id"].astype(int)))
            unimol_fold_ids_full = np.array([smi_to_fold.get(s, -1) for s in smi], dtype=int)
            missing = int((unimol_fold_ids_full < 0).sum())
            print(f"[{name}] unimol_fold from CSV: matched {len(smi)-missing}/{len(smi)}; {missing} missing")
        else:
            print(f"[{name}] no CSV; generating k=10 scaffold folds locally (seed=42, protocol-matched)")
            unimol_fold_ids_full = scaffold_kfold(smi, seed=42, k=10)
            unique, cnt = np.unique(unimol_fold_ids_full, return_counts=True)
            print(f"[{name}] fold sizes: {dict(zip(unique.tolist(), cnt.tolist()))}")

    t0 = time.time()
    X, ok = featurize(smi)
    y = y[ok]
    if split is not None:
        split = split[ok]
    if split_mode == "unimol_fold":
        unimol_fold_ids = unimol_fold_ids_full[ok]
        # Drop any rows where fold was missing
        valid = unimol_fold_ids >= 0
        X = X[valid]; y = y[valid]; unimol_fold_ids = unimol_fold_ids[valid]
    smi_kept = [s for s, k in zip(smi, ok) if k]
    print(f"[{name}] featurized {X.shape} in {time.time()-t0:.1f}s; dropped {int((~ok).sum())} invalid SMILES")

    out = {"dataset": name, "task": cfg["task"], "n": int(X.shape[0]), "split_mode": split_mode, "n_seeds": len(seeds), "seeds": {}}
    for s in seeds:
        tr, te = split_indices(smi_kept, X.shape[0], split, s, split_mode=split_mode, unimol_fold_ids=unimol_fold_ids)
        m = fit_eval(X[tr], y[tr], X[te], y[te], cfg["task"], s)
        m["n_train"], m["n_test"] = int(len(tr)), int(len(te))
        out["seeds"][str(s)] = m
        print(f"[{name}] seed={s}  ", "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))

    # Aggregate
    metric_keys = [k for k in out["seeds"][str(seeds[0])] if k not in ("n_train", "n_test")]
    out["mean"] = {k: float(np.mean([out["seeds"][str(s)][k] for s in seeds])) for k in metric_keys}
    out["std"] = {k: float(np.std([out["seeds"][str(s)][k] for s in seeds], ddof=1)) for k in metric_keys}
    print(f"[{name}] MEAN  ", "  ".join(f"{k}={out['mean'][k]:.4f}+/-{out['std'][k]:.4f}" for k in metric_keys))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--split", choices=["scaffold", "random", "unimol_fold"], default="scaffold")
    p.add_argument("--n_seeds", type=int, default=5,
                   help="Number of RF seeds. 5 (default) uses the existing SEEDS=[9,19,29,39,49] "
                        "for backward-compat with stored JSON. 30 switches to Deng2023's seeds=range(30). "
                        "Any other integer uses SEEDS[:n].")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = str(REPO / f"baselines_ml_results_{args.split}.json")

    if args.n_seeds == 30:
        seeds = SEEDS_DENG30
    elif args.n_seeds == 5:
        seeds = SEEDS
    else:
        seeds = SEEDS[:args.n_seeds] if args.n_seeds <= len(SEEDS) else list(range(args.n_seeds))

    results = {}
    for name in args.datasets:
        if name not in DATASETS:
            print(f"skip unknown dataset {name}")
            continue
        results[name] = run_dataset(name, split_mode=args.split, seeds=seeds)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
