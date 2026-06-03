"""End-to-end Uni-Mol *finetune* baseline, matched to the SpecMol study splits.

Motivation
----------
The paper's "Uni-Mol" row was a *frozen* Uni-Mol CLS embedding + RandomForest
control (see baselines_matched.py). That is a legitimate probe, but it is NOT the
Uni-Mol model's own performance: the original Uni-Mol paper (Zhou et al., ICLR
2023) reports end-to-end finetune numbers (BACE 0.857, ClinTox 0.919, ESOL 0.788
RMSE, FreeSolv 1.480, Lipo 0.603). Reviewers will read a row literally labelled
"Uni-Mol" as the latter. This script produces the *real* end-to-end finetune
number, on the SAME train/test partition every other cell in the paper uses, so
it is protocol-matched to V0/V2-T5/T7/RF.

Design
------
- Split: imported verbatim from baselines_matched.py (load_split + DATASETS) so
  the train/test molecule sets are IDENTICAL to the RF / Uni-Mol-embedding cells.
  fold0 = test (classification); split=='test' = test (regression). bbbp/bace are
  re-pointed at paper/audits/<ds>_fold_recovered.csv (the recovered matched folds).
- Estimator: unimol_tools.MolTrain (end-to-end finetune of the pretrained Uni-Mol
  encoder) on the train split, then MolPredict on the held-out test split.
- Metric: roc_auc (single-task cls), macro-AUC over tasks (clintox 2 tasks),
  RMSE = sqrt(mean((pred-y)^2)) for regression -- on the SAME label column the
  deep models and RF used, so the RMSE scale is identical (standardized labels).
- Reporting: mean +/- sample std over --n_seeds finetune seeds, written to JSON in
  the same shape as baselines_matched_results.json (dataset -> {mean,std,seeds}).

Run on HPC (specmol env: torch + unimol_tools + Uni-Mol weights), GPU node only:
  python reproduce_unimol_finetune.py --datasets bace clintox bbbp \
      --n_seeds 3 --epochs 50 --out unimol_finetune_results.json

NOTE: unimol_tools downloads/uses its own pretrained weights (model 'unimolv1',
all-H). That is the same checkpoint family as unimol_weights/mol_pre_all_h_*.pt.
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score

# Reuse the EXACT matched-split loader the RF / frozen-Uni-Mol cells used, so the
# finetune baseline lands on the identical train/test partition.
from baselines_matched import DATASETS as _BASE_DATASETS, load_split, REPO

RDLogger.DisableLog("rdApp.*")

# Re-point bbbp/bace at the recovered matched-fold CSVs (B2). These carry
# columns smiles,label,fold_id -- identical fold semantics to the other cls sets.
DATASETS = {k: dict(v) for k, v in _BASE_DATASETS.items()}
DATASETS["bbbp"] = dict(DATASETS["bbbp"], split_csv="paper/audits/bbbp_fold_recovered.csv",
                        labels=["label"])
DATASETS["bace"] = dict(DATASETS["bace"], split_csv="paper/audits/bace_fold_recovered.csv",
                        labels=["label"])


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def _macro_auc(y_true, y_score):
    """Macro AUC over columns (multi-task). Skips degenerate (single-class) tasks."""
    y_true = np.asarray(y_true, float)
    y_score = np.asarray(y_score, float)
    if y_true.ndim == 1:
        return float(roc_auc_score(y_true, y_score))
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if len(np.unique(col[~np.isnan(col)])) < 2:
            continue
        mask = ~np.isnan(col)
        aucs.append(roc_auc_score(col[mask], y_score[mask, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _target_cols(y):
    """Column names unimol_tools auto-detects via its 'TARGET' prefix."""
    if y.ndim == 1 or y.shape[1] == 1:
        return ["TARGET"]
    return [f"TARGET_{j}" for j in range(y.shape[1])]


def _write_csv(path, smi, y=None):
    """unimol_tools' most portable input is a SMILES csv. Train CSV carries the
    TARGET column(s); test CSV is SMILES-only."""
    df = pd.DataFrame({"SMILES": smi})
    if y is not None:
        cols = _target_cols(y)
        if y.ndim == 1:
            df[cols[0]] = y
        else:
            for j, c in enumerate(cols):
                df[c] = y[:, j]
    df.to_csv(path, index=False)


def _unimol_task(task, y):
    if task == "regression":
        return "regression"
    return "multilabel_classification" if (y.ndim == 2 and y.shape[1] > 1) else "classification"


def _set_global_seed(seed):
    """MolTrain exposes no seed arg, so vary the global RNGs to perturb the
    internal random CV split / init across seeds. Belt-and-suspenders; the
    caller also warns if per-seed outputs come back identical."""
    import os
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:  # if the installed unimol_tools exposes a seed setter, use it too
        from unimol_tools.utils import set_seed as _us
        _us(seed)
    except Exception:
        pass


def finetune_eval(smi_tr, y_tr, smi_te, y_te, task, seed, epochs, lr, batch_size):
    """One end-to-end Uni-Mol finetune + eval on the held-out test split."""
    from unimol_tools import MolTrain, MolPredict

    _set_global_seed(seed)
    umtask = _unimol_task(task, y_tr)
    workdir = Path(tempfile.mkdtemp(prefix=f"unimol_ft_{seed}_"))
    try:
        train_csv = workdir / "train.csv"
        test_csv = workdir / "test.csv"
        _write_csv(train_csv, smi_tr, y_tr)
        _write_csv(test_csv, smi_te)

        clf = MolTrain(
            task=umtask,
            data_type="molecule",
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            kfold=5,                 # internal CV for early-stop / checkpoint select
            metrics="auc" if umtask != "regression" else "mse",
            save_path=str(workdir / "exp"),
            remove_hs=False,         # match the all-H pretrained checkpoint
            smiles_col="SMILES",
            target_cols=_target_cols(y_tr),
        )
        clf.fit(data=str(train_csv))

        pred = MolPredict(load_model=str(workdir / "exp")).predict(data=str(test_csv))
        pred = np.asarray(pred, float)

        if task == "regression":
            return {"rmse": _rmse(y_te, pred.ravel() if pred.ndim > 1 else pred)}
        if y_te.ndim == 2 and y_te.shape[1] > 1:
            return {"roc_auc": _macro_auc(y_te, pred)}
        return {"roc_auc": float(roc_auc_score(y_te, pred.ravel() if pred.ndim > 1 else pred))}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run(name, cfg, seeds, epochs, lr, batch_size):
    smi, y, is_train, is_test = load_split(cfg)
    valid = np.array([Chem.MolFromSmiles(s) is not None for s in smi], dtype=bool)
    smi = [s for s, v in zip(smi, valid) if v]
    y, is_train, is_test = y[valid], is_train[valid], is_test[valid]

    tr = np.flatnonzero(is_train)
    te = np.flatnonzero(is_test)
    smi_tr = [smi[i] for i in tr]
    smi_te = [smi[i] for i in te]
    y_tr, y_te = y[tr], y[te]

    out = dict(dataset=name, task=cfg["task"], feature="unimol_finetune",
               n=len(smi), n_train=len(tr), n_test=len(te),
               split_csv=cfg["split_csv"], seeds={})
    t0 = time.time()
    for s in seeds:
        out["seeds"][str(s)] = finetune_eval(smi_tr, y_tr, smi_te, y_te,
                                              cfg["task"], s, epochs, lr, batch_size)
    keys = list(out["seeds"][str(seeds[0])].keys())
    out["mean"] = {k: float(np.nanmean([out["seeds"][str(s)][k] for s in seeds])) for k in keys}
    out["std"] = {k: float(np.nanstd([out["seeds"][str(s)][k] for s in seeds], ddof=1))
                  for k in keys}
    # Guard: MolTrain has no seed arg. If every seed returned an identical metric
    # the std is a fake 0 (no real seed variance) -- flag it loudly.
    if len(seeds) > 1:
        for k in keys:
            vals = [out["seeds"][str(s)][k] for s in seeds]
            if len(set(np.round(vals, 8))) == 1:
                print(f"[{name}] WARNING: {k} identical across all {len(seeds)} seeds "
                      f"({vals[0]:.4f}); MolTrain seed control may be ineffective -- "
                      f"std is not a real seed std.")
    print(f"[{name}] tr={len(tr)} te={len(te)} ({time.time()-t0:.0f}s)  " +
          "  ".join(f"{k}={out['mean'][k]:.4f}+/-{out['std'][k]:.4f}" for k in keys))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n_seeds", type=int, default=3,
                    help="finetune seeds; 3 matches the deep-variant protocol {9,19,29}")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="explicit seed list; overrides --n_seeds (default [9,19,29] for n=3)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=str(REPO / "unimol_finetune_results.json"))
    a = ap.parse_args()

    if a.seeds is not None:
        seeds = a.seeds
    elif a.n_seeds == 3:
        seeds = [9, 19, 29]          # match the deep-variant training seeds
    else:
        seeds = list(range(a.n_seeds))

    results = {}
    out_path = Path(a.out)
    if out_path.exists():                      # resume / accumulate
        results = json.loads(out_path.read_text())
    for name in a.datasets:
        if name not in DATASETS:
            print(f"skip unknown dataset {name}")
            continue
        if not (REPO / DATASETS[name]["split_csv"]).exists():
            print(f"[{name}] split CSV missing ({DATASETS[name]['split_csv']}); skipping")
            continue
        try:
            results[name] = run(name, DATASETS[name], seeds, a.epochs, a.lr, a.batch_size)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
        out_path.write_text(json.dumps(results, indent=2))   # checkpoint per dataset
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
