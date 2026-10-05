"""Matched-split frozen-feature baselines for the ICDE 2027 EAB study.

Adds two controls the paper needs, both evaluated on the SAME split the SpecMol
model used, so they are protocol-matched to V0/V2-T5/T7:

  (1) RF on Morgan(1024,r2) + 11 RDKit-2D descriptors -- re-run MATCHED for
      esol/lipo/freesolv/clintox. baselines_ml.py had no split CSV wired for
      these and fell back to a locally regenerated k10 fold, so their RF cells
      were not guaranteed to use the model's exact test set.
  (2) RF on the frozen Uni-Mol molecular (CLS) embedding from UniMolRepr --
      the "Uni-Mol direct" control that rules out the reviewer rebuttal
      "the injection is bad, not the 3D signal". All benchmarks.

Both reuse baselines_ml.fit_eval, so the RF config (300 trees, balanced) and the
metrics (roc_auc / macro-AUC / rmse-via-np.sqrt) are IDENTICAL to the existing
bbbp/bace RF cells. 30 RF random-states per cell (Deng2023 n=30 convention).

Split source per dataset (paths verified present on HPC 2026-05-29):
  regression (esol/lipo/freesolv): unimol_out_<task>/splits/<task>_split_seed42.csv
      columns smiles,label,split(train|valid|test). valid->train (RF needs no
      val); test = the model's exact test set.
  classification (bbbp/bace/clintox): a data_with_folds CSV with a fold_id
      column; fold0=test, folds1-9=train (matches the model's fold0-test split).

Run on HPC (specmol env: rdkit + unimol_tools + Uni-Mol weights). Example:
  UNIMOL_CKPT=$HOME/zhoutianyang/SpecMol/SpecMol-Zip/unimol_weights/mol_pre_all_h_220816.pt \
  UNIMOL_DICT=$HOME/miniconda3/envs/specmol/lib/python3.10/site-packages/unimol_tools/weights/mol.dict.txt \
  python baselines_matched.py --out baselines_matched_results.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

# Reuse the EXACT featurizer + RF eval the existing RF cells used, so the new
# numbers are directly comparable to bbbp/bace deng30.
from baselines_ml import featurize as featurize_morgan_rdkit, fit_eval, REPO

RDLogger.DisableLog("rdApp.*")

# split_csv paths are relative to REPO (the SpecMol-Zip dir).
DATASETS = {
    "freesolv": dict(split_csv="unimol_out_freesolv/splits/freesolv_split_seed42.csv",
                     kind="reg_split", task="regression", smiles="smiles", labels=["label"]),
    "esol": dict(split_csv="unimol_out_esol/splits/esol_split_seed42.csv",
                 kind="reg_split", task="regression", smiles="smiles", labels=["label"]),
    "lipo": dict(split_csv="unimol_out_lipo/splits/lipo_split_seed42.csv",
                 kind="reg_split", task="regression", smiles="smiles", labels=["label"]),
    "clintox": dict(split_csv="unimol_out_clintox/splits/data_with_folds_scaffold_k10_seed42.csv",
                    kind="fold", task="classification", smiles="smiles", labels=["label_0", "label_1"]),
    # bbbp/bace: RF-Morgan already matched in deng30; included here mainly for the
    # Uni-Mol-embedding row. Fold-CSV paths are best-effort; missing -> skipped.
    "bbbp": dict(split_csv="unimol_out_bbbp/splits/data_with_folds_scaffold_k10_seed42.csv",
                 kind="fold", task="classification", smiles="smiles", labels=None),
    "bace": dict(split_csv="unimol_out/splits/data_with_folds_scaffold_k10_seed42.csv",
                 kind="fold", task="classification", smiles="smiles", labels=None),
}

_SKIP_COLS = {"smiles", "fold_id", "split", "sdf_index", "embed_ok"}


def _detect_labels(df, cfg):
    if cfg["labels"]:
        return cfg["labels"]
    return [c for c in df.columns if c not in _SKIP_COLS][:1]


def load_split(cfg):
    """Return (smiles, y, is_train, is_test) on the model's exact split."""
    df = pd.read_csv(REPO / cfg["split_csv"])
    smi = df[cfg["smiles"]].astype(str).str.strip().tolist()
    labels = _detect_labels(df, cfg)
    y = df[labels].astype(float).to_numpy()
    y = y.ravel() if y.shape[1] == 1 else y
    if cfg["kind"] == "reg_split":
        sp = df["split"].astype(str).str.lower().to_numpy()
        is_test = sp == "test"
        is_train = (sp == "train") | (sp == "valid")  # valid -> train; RF needs no val
    else:
        fid = df["fold_id"].astype(int).to_numpy()
        is_test = fid == 0
        is_train = fid != 0  # folds 1-9 (incl. the val fold) -> train
    return smi, y, is_train, is_test


def featurize_unimol(smiles):
    """Frozen Uni-Mol molecular (CLS) embedding. Returns (X, ok_all_true)."""
    from unimol_tools import UniMolRepr
    rep = UniMolRepr(data_type="molecule", remove_hs=False)
    out = rep.get_repr(smiles, return_atomic_reprs=False)
    cls = out["cls_repr"] if isinstance(out, dict) else out
    X = np.asarray(cls, dtype=np.float32)
    if X.shape[0] != len(smiles):
        raise RuntimeError(f"UniMolRepr returned {X.shape[0]} rows for {len(smiles)} SMILES")
    # Guard: failed/garbage conformers can yield non-finite embeddings. Drop
    # those rows (the run() caller realigns y/splits via the returned ok mask)
    # rather than silently feeding NaNs into RandomForest.
    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
        print(f"[unimol] WARNING: dropping {int((~ok).sum())}/{len(smiles)} "
              f"molecules with non-finite embeddings (failed conformers)")
    return X[ok], ok


FEATURIZERS = {"morgan_rdkit": featurize_morgan_rdkit, "unimol": featurize_unimol}


def run(name, cfg, feature, seeds):
    smi, y, is_train, is_test = load_split(cfg)
    # Pre-filter to RDKit-valid SMILES so both feature types share an identical
    # molecule set and all index masks stay aligned.
    valid = np.array([Chem.MolFromSmiles(s) is not None for s in smi], dtype=bool)
    smi = [s for s, v in zip(smi, valid) if v]
    y, is_train, is_test = y[valid], is_train[valid], is_test[valid]

    t0 = time.time()
    X, ok = FEATURIZERS[feature](smi)
    if not ok.all():  # defensive; for morgan post-filter ok should be all-True
        y, is_train, is_test = y[ok], is_train[ok], is_test[ok]
    tr = np.flatnonzero(is_train)
    te = np.flatnonzero(is_test)

    out = dict(dataset=name, task=cfg["task"], feature=feature, n=int(X.shape[0]),
               n_train=int(len(tr)), n_test=int(len(te)),
               split_csv=cfg["split_csv"], feat_dim=int(X.shape[1]), seeds={})
    for s in seeds:
        out["seeds"][str(s)] = fit_eval(X[tr], y[tr], X[te], y[te], cfg["task"], s)
    first = out["seeds"][str(seeds[0])]
    keys = [k for k, v in first.items()
            if isinstance(v, (int, float)) and k not in ("n_train", "n_test", "n_tasks")]
    out["mean"] = {k: float(np.nanmean([out["seeds"][str(s)][k] for s in seeds])) for k in keys}
    out["std"] = {k: float(np.nanstd([out["seeds"][str(s)][k] for s in seeds], ddof=1)) for k in keys}
    print(f"[{name}/{feature}] n={out['n']} tr={len(tr)} te={len(te)} dim={out['feat_dim']} "
          f"({time.time()-t0:.0f}s)  " +
          "  ".join(f"{k}={out['mean'][k]:.4f}+/-{out['std'][k]:.4f}" for k in keys))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--features", nargs="+", default=["morgan_rdkit", "unimol"],
                    choices=list(FEATURIZERS))
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--out", default=str(REPO / "baselines_matched_results.json"))
    a = ap.parse_args()
    seeds = list(range(a.n_seeds))

    results = {}
    for name in a.datasets:
        if name not in DATASETS:
            print(f"skip unknown dataset {name}")
            continue
        if not (REPO / DATASETS[name]["split_csv"]).exists():
            print(f"[{name}] split CSV missing ({DATASETS[name]['split_csv']}); skipping")
            continue
        results.setdefault(name, {})
        for feature in a.features:
            try:
                results[name][feature] = run(name, DATASETS[name], feature, seeds)
            except Exception as e:
                print(f"[{name}/{feature}] FAILED: {type(e).__name__}: {e}")
        Path(a.out).write_text(json.dumps(results, indent=2))  # checkpoint per dataset
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
