"""Chemprop (D-MPNN) baseline for SpecMol ICDE submission (L2).

Standard supervised graph baseline cited in every 2024-26 MoleculeNet
paper. Runs chemprop_train via subprocess on the same Uni-Mol fold
split (FOLD=0 -> test, FOLD=1 -> val, others -> train) used by
baselines_ml.py and the deep V0/V2/T6 models, so the comparison is
matched cell-for-cell with the rest of the paper.

Output: baselines_chemprop_results_unimol_fold.json with mean/std
across SEEDS per dataset.

Usage:
  python SpecMol-Zip/baselines_chemprop.py --datasets bace bbbp freesolv

CPU runtime: ~5-15 min per (dataset, seed). Default 3 seeds x 3
datasets ~ 1-2h on a laptop.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent

# Mirror baselines_ml.py DATASETS exactly so the splits are identical.
DATASETS = {
    "bace": {
        "csv": "dataset/bace.csv",
        "smiles_col": "smiles",
        "label_col": "Class",
        "task": "classification",
        "unimol_fold_csv": "unimol_out/splits/data_with_folds_scaffold_k10_seed42.csv",
        "chemprop_metric": "auc",
        "result_metric": "roc_auc",
    },
    "bbbp": {
        "csv": "dataset/BBBP.csv",
        "smiles_col": "smiles",
        "label_col": "p_np",
        "task": "classification",
        "unimol_fold_csv": None,  # generate scaffold_kfold on-the-fly (seed=42)
        "chemprop_metric": "auc",
        "result_metric": "roc_auc",
    },
    "freesolv": {
        "csv": "dataset/freesolv/raw/smiles.csv",
        "smiles_col": 0,
        "label_col": 1,
        "task": "regression",
        "unimol_fold_csv": None,
        "chemprop_metric": "rmse",
        "result_metric": "rmse",
    },
    "clintox": {
        "csv": "dataset/clintox/raw/smiles.csv",
        "smiles_col": 0,
        "label_col": [1, 2],
        "task": "classification",
        "unimol_fold_csv": None,
        "chemprop_metric": "auc",
        "result_metric": "roc_auc",
    },
    "tox21": {
        "csv": "dataset/tox21/raw/smiles.csv",
        "smiles_col": 0,
        "label_col": list(range(1, 13)),
        "task": "classification",
        "unimol_fold_csv": None,
        "chemprop_metric": "auc",
        "result_metric": "roc_auc",
    },
}

SEEDS = [9, 19, 29]
KFOLD = 10
FOLD_SPLIT_SEED = 42
TEST_FOLD = 0
VAL_FOLD = 1


def _murcko(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_kfold(smiles_list, seed=FOLD_SPLIT_SEED, k=KFOLD):
    """Same scaffold-fold protocol baselines_ml.py uses for BBBP/FreeSolv."""
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        scf = _murcko(smi)
        scaffolds.setdefault(scf, []).append(i)
    rng = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    groups.sort(key=lambda g: (-len(g), rng.random()))
    fold_sizes = [0] * k
    fold_ids = np.zeros(len(smiles_list), dtype=int)
    for g in groups:
        target = int(np.argmin(fold_sizes))
        for idx in g:
            fold_ids[idx] = target
        fold_sizes[target] += len(g)
    return fold_ids


def load_dataset(name):
    cfg = DATASETS[name]
    csv_path = REPO / cfg["csv"]
    label_col = cfg["label_col"]
    multi = isinstance(label_col, list)
    if isinstance(cfg["smiles_col"], int):
        df = pd.read_csv(csv_path, header=None)
        smiles = df[cfg["smiles_col"]].astype(str).tolist()
        labels = df[label_col].values  # 2D if multi, 1D if scalar col
    else:
        df = pd.read_csv(csv_path)
        smiles = df[cfg["smiles_col"]].astype(str).tolist()
        labels = df[label_col].values
    if multi and labels.ndim == 1:
        labels = labels.reshape(-1, 1)  # defensive; pandas usually returns 2D for list cols

    # Build fold_ids
    if cfg["unimol_fold_csv"]:
        fold_csv = REPO / cfg["unimol_fold_csv"]
        if fold_csv.exists():
            fold_df = pd.read_csv(fold_csv)
            smi_to_fold = dict(zip(fold_df["smiles"].astype(str), fold_df["fold_id"].astype(int)))
            fold_ids = np.array([smi_to_fold.get(s, -1) for s in smiles], dtype=int)
        else:
            fold_ids = scaffold_kfold(smiles)
    else:
        fold_ids = scaffold_kfold(smiles)

    # Drop SMILES with missing fold and invalid mols
    ok = []
    for i, s in enumerate(smiles):
        if fold_ids[i] < 0:
            ok.append(False)
            continue
        if Chem.MolFromSmiles(s) is None:
            ok.append(False)
            continue
        ok.append(True)
    ok = np.array(ok, dtype=bool)
    smiles = [s for s, k in zip(smiles, ok) if k]
    labels = labels[ok]
    fold_ids = fold_ids[ok]
    return smiles, labels, fold_ids, cfg


def write_split_csvs(smiles, labels, fold_ids, cfg, work_dir):
    """Write train/val/test CSVs in chemprop's expected (smiles, target1, ...) format.

    Returns (label_names, n_train, n_val, n_test). label_names is a list so the
    chemprop_train --target_columns flag can accept the right number of columns.
    """
    work_dir = Path(work_dir)
    multi = labels.ndim == 2
    if multi:
        n_tasks = labels.shape[1]
        label_names = [f"task{t}" for t in range(n_tasks)]
    else:
        label_names = ["target"]
    test_mask = fold_ids == TEST_FOLD
    val_mask = fold_ids == VAL_FOLD
    train_mask = ~(test_mask | val_mask)
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        cols = {"smiles": [smiles[i] for i in np.flatnonzero(mask)]}
        if multi:
            for t, ln in enumerate(label_names):
                cols[ln] = labels[mask, t]
        else:
            cols[label_names[0]] = labels[mask]
        df = pd.DataFrame(cols)
        df.to_csv(work_dir / f"{name}.csv", index=False)
    return label_names, int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())


SCORE_RE = re.compile(r"Ensemble test (?:auc|rmse|mae)\s*=\s*([0-9.eE+-]+)")
PER_TASK_SCORE_RE = re.compile(r"Overall test (?:auc|rmse|mae)\s*=\s*([0-9.eE+-]+)")


def parse_chemprop_output(stdout, save_dir, metric_key):
    """Pull the macro-mean test score out of chemprop's test_scores.csv (or
    fall back to stdout regex). For multi-task runs chemprop writes one row
    per task; we mean across rows so the returned scalar is the macro metric.
    """
    test_scores = Path(save_dir) / "test_scores.csv"
    if test_scores.exists():
        df = pd.read_csv(test_scores)
        cols = [c for c in df.columns if metric_key.lower() in c.lower()]
        if cols:
            vals = df[cols[0]].astype(float).dropna()
            if len(vals) > 0:
                return float(vals.mean())
    for rx in (PER_TASK_SCORE_RE, SCORE_RE):
        matches = rx.findall(stdout)
        if matches:
            return float(matches[-1])
    return None


def _find_chemprop_train():
    """Locate the chemprop_train entry-point alongside the current Python.

    On Windows it lives at <venv>/Scripts/chemprop_train.exe; on POSIX at
    <venv>/bin/chemprop_train. Resolves once at runtime and caches.
    """
    py = Path(sys.executable)
    bindir = py.parent
    for cand in (bindir / "chemprop_train.exe", bindir / "chemprop_train"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("chemprop_train entry-point not found next to "
                            f"{py}; is chemprop installed in this env?")


CHEMPROP_TRAIN = _find_chemprop_train()


def run_chemprop_one(name, seed, work_dir):
    smiles, labels, fold_ids, cfg = load_dataset(name)
    label_names, n_tr, n_val, n_te = write_split_csvs(smiles, labels, fold_ids, cfg, work_dir)
    save_dir = Path(work_dir) / f"out_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        CHEMPROP_TRAIN,
        "--data_path", str(Path(work_dir) / "train.csv"),
        "--separate_val_path", str(Path(work_dir) / "val.csv"),
        "--separate_test_path", str(Path(work_dir) / "test.csv"),
        "--dataset_type", cfg["task"],
        "--metric", cfg["chemprop_metric"],
        "--save_dir", str(save_dir),
        "--smiles_columns", "smiles",
        "--target_columns", *label_names,
        "--num_folds", "1",
        "--seed", str(seed),
        "--pytorch_seed", str(seed),
        "--epochs", "30",
        "--quiet",
    ]
    print(f"[{name}] seed={seed}  n_train={n_tr} n_val={n_val} n_test={n_te}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[{name}] seed={seed} FAILED ({elapsed:.1f}s)")
        print("STDOUT tail:\n", "\n".join(proc.stdout.splitlines()[-15:]))
        print("STDERR tail:\n", "\n".join(proc.stderr.splitlines()[-15:]))
        return None
    score = parse_chemprop_output(proc.stdout, save_dir, cfg["chemprop_metric"])
    print(f"[{name}] seed={seed}  test_{cfg['chemprop_metric']}={score}  ({elapsed:.1f}s)")
    return {
        "test_metric": score,
        "metric_name": cfg["chemprop_metric"],
        "n_train": n_tr,
        "n_val": n_val,
        "n_test": n_te,
        "elapsed_sec": round(elapsed, 1),
    }


def run_dataset(name, seeds):
    cfg = DATASETS[name]
    out = {"dataset": name, "task": cfg["task"], "split_mode": "unimol_fold_k10_seed42",
           "seeds": {}}
    metric_name = cfg["chemprop_metric"]
    values = []
    for s in seeds:
        with tempfile.TemporaryDirectory(prefix=f"chemprop_{name}_s{s}_") as tmp:
            result = run_chemprop_one(name, s, tmp)
        if result is None or result["test_metric"] is None:
            print(f"[{name}] seed={s} no score, skipping in aggregate")
            continue
        out["seeds"][str(s)] = result
        values.append(result["test_metric"])
    if values:
        out["mean"] = {metric_name: float(np.mean(values))}
        out["std"] = {metric_name: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        print(f"[{name}] MEAN {metric_name}={out['mean'][metric_name]:.4f} +/- {out['std'][metric_name]:.4f} (n={len(values)})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = str(REPO / "baselines_chemprop_results_unimol_fold.json")

    results = {}
    for name in args.datasets:
        if name not in DATASETS:
            print(f"skip unknown dataset {name}")
            continue
        results[name] = run_dataset(name, args.seeds)
        # Write incrementally so partial results survive a crash
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
