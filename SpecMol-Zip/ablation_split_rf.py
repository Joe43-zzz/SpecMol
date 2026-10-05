"""Scaffold-vs-random split ablation (data-pipeline effect; CPU / RF only).

For each dataset we take the SAME molecules, labels, featurizer, and model
(matched RF = Morgan + RDKit-2D via baselines_ml.fit_eval) and evaluate under
two evaluation protocols:
  (a) the study's scaffold split (fold-0 = test), the protocol the deep models
      and matched baselines use;
  (b) a random split of IDENTICAL test size (seeded), holding everything else
      fixed.
The protocol-induced delta = metric(random) - metric(scaffold) isolates how much
the *split choice alone* moves measured performance, with data + features +
model fixed -- a controlled data-engineering ablation.

Pre-registered reading: |delta| above ~0.03 (AUC) / ~0.05 (RMSE) means the split
protocol alone moves the number on the order of the deep-vs-classical gap, i.e.
protocol choices account for a substantial fraction of reported differences;
near-zero means the metric is split-robust on that endpoint.

Output: ablation_split_rf.json + console summary.
Run (HPC, specmol env):  python ablation_split_rf.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from baselines_matched import DATASETS, load_split
from baselines_ml import featurize as featurize_morgan_rdkit, fit_eval, REPO

RDLogger.DisableLog("rdApp.*")


def _agg(d, seeds):
    keys = [k for k, v in d[str(seeds[0])].items()
            if isinstance(v, (int, float)) and k not in ("n_train", "n_test", "n_tasks")]
    return {k: (float(np.nanmean([d[str(s)][k] for s in seeds])),
                float(np.nanstd([d[str(s)][k] for s in seeds], ddof=1))) for k in keys}


def run(name, cfg, seeds):
    smi, y, is_train, is_test = load_split(cfg)
    valid = np.array([Chem.MolFromSmiles(s) is not None for s in smi], dtype=bool)
    smi = [s for s, v in zip(smi, valid) if v]
    y, is_train, is_test = y[valid], is_train[valid], is_test[valid]

    t0 = time.time()
    X, ok = featurize_morgan_rdkit(smi)
    if not ok.all():
        y, is_train, is_test = y[ok], is_train[ok], is_test[ok]
    n = int(X.shape[0])
    n_test = int(is_test.sum())

    # (a) scaffold protocol: fixed split, RF random_state varies across seeds.
    scaf_tr, scaf_te = np.flatnonzero(is_train), np.flatnonzero(is_test)
    scaf = {str(s): fit_eval(X[scaf_tr], y[scaf_tr], X[scaf_te], y[scaf_te], cfg["task"], s)
            for s in seeds}

    # (b) random protocol: per-seed random test of identical size.
    rand = {}
    for s in seeds:
        rng = np.random.RandomState(1000 + s)
        perm = rng.permutation(n)
        te, tr = perm[:n_test], perm[n_test:]
        rand[str(s)] = fit_eval(X[tr], y[tr], X[te], y[te], cfg["task"], s)

    sa, ra = _agg(scaf, seeds), _agg(rand, seeds)
    metric = "rmse" if cfg["task"] == "regression" else ("roc_auc" if "roc_auc" in sa else list(sa)[0])
    sm, rm = sa[metric], ra[metric]
    res = dict(dataset=name, task=cfg["task"], n=n, n_test=n_test, metric=metric,
               scaffold_mean=sm[0], scaffold_std=sm[1],
               random_mean=rm[0], random_std=rm[1],
               delta_random_minus_scaffold=float(rm[0] - sm[0]),
               n_seeds=len(seeds), scaffold_all=sa, random_all=ra,
               split_csv=cfg["split_csv"])
    print(f"[{name}] {metric}: scaffold={sm[0]:.4f}+/-{sm[1]:.4f}  "
          f"random={rm[0]:.4f}+/-{rm[1]:.4f}  delta(rand-scaf)={rm[0]-sm[0]:+.4f}  "
          f"({time.time()-t0:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "ablation_split_rf.json"))
    a = ap.parse_args()
    seeds = list(range(a.n_seeds))
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
            results[name] = run(name, cfg, seeds)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
        Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
