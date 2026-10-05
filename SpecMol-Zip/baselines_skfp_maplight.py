"""Tier-1 CPU baselines: scikit-fingerprints sweep + MapLight-style fusion.

Hardens the paper's "simple-baseline ceiling" claim beyond the single
Morgan(1024)+RDKit2D RF cell:

  (1) skfp fingerprint sweep + RF  -- ECFP / AtomPair / TopologicalTorsion /
      MACCS / RDKit2D, each into the IDENTICAL RF (300 trees, balanced) and the
      IDENTICAL metrics used by every other RF cell (reuses baselines_ml.fit_eval).
      Shows the fingerprint-saturation ceiling is not a one-fingerprint artifact.

  (2) MapLight-style fusion + CatBoost -- ECFP(2048,r2) + Avalon(512) + RDKit2D +
      ErG(rdkit) concatenated into a gradient-boosted tree. This is the recipe
      that tops TDC ADMET and is one of only ~3 models that passed the 2026 TDC
      reproducibility audit (bioRxiv 10.64898/2026.02.26.708193). It is the direct
      competitor to SpecMol's fingerprint+GNN fusion design, so the paper needs it.

Both reuse baselines_matched.load_split, so every cell is evaluated on the SAME
split SpecMol's V0/V2-T5/T7 used -> protocol-matched, directly comparable.

CPU-only (no GPU touched). Cap parallelism so a shared box is not starved:
  LOKY_MAX_CPU_COUNT=16 OMP_NUM_THREADS=2 nice -n 19 \
  python baselines_skfp_maplight.py --out baselines_skfp_maplight_results.json

Splits present on thk 2026-06-04: freesolv / bbbp / bace. esol/lipo/clintox
split CSVs are missing (recover from .pt like B2 before running those).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdReducedGraphs

# Reuse the EXACT split loader + RF eval + dataset registry the matched RF cells use.
from baselines_matched import DATASETS, load_split, REPO
from baselines_ml import fit_eval

RDLogger.DisableLog("rdApp.*")

# Cap intra-op threads of any BLAS the fingerprints/CatBoost pull in.
_NJOBS = int(os.environ.get("BASELINE_NJOBS", "16"))


def _finite_mask(X):
    return np.isfinite(X).all(axis=1)


def _skfp(transformer):
    """Wrap a skfp transformer as a (smiles)->(X, ok) featurizer."""
    def f(smiles):
        X = np.asarray(transformer.transform(smiles), dtype=np.float32)
        ok = _finite_mask(X)
        if not ok.all():
            print(f"[skfp] dropping {int((~ok).sum())}/{len(smiles)} non-finite rows")
        return X[ok], ok
    return f


def make_skfp_featurizers():
    from skfp.fingerprints import (
        ECFPFingerprint, AtomPairFingerprint, TopologicalTorsionFingerprint,
        MACCSFingerprint, RDKit2DDescriptorsFingerprint,
    )
    return {
        "ecfp":   _skfp(ECFPFingerprint(fp_size=2048, radius=2, n_jobs=_NJOBS)),
        "atompair": _skfp(AtomPairFingerprint(fp_size=2048, n_jobs=_NJOBS)),
        "toptorsion": _skfp(TopologicalTorsionFingerprint(fp_size=2048, n_jobs=_NJOBS)),
        "maccs":  _skfp(MACCSFingerprint(n_jobs=_NJOBS)),
        "rdkit2d": _skfp(RDKit2DDescriptorsFingerprint(n_jobs=_NJOBS)),
    }


def featurize_maplight(smiles):
    """MapLight-style concat: ECFP(2048,r2) + Avalon(512) + RDKit2D + ErG(315)."""
    from skfp.fingerprints import ECFPFingerprint, AvalonFingerprint, RDKit2DDescriptorsFingerprint
    ecfp = np.asarray(ECFPFingerprint(fp_size=2048, radius=2, n_jobs=_NJOBS).transform(smiles), dtype=np.float32)
    aval = np.asarray(AvalonFingerprint(fp_size=512, n_jobs=_NJOBS).transform(smiles), dtype=np.float32)
    rd2d = np.asarray(RDKit2DDescriptorsFingerprint(n_jobs=_NJOBS).transform(smiles), dtype=np.float32)
    erg = np.asarray([rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
    X = np.hstack([ecfp, aval, rd2d, erg]).astype(np.float32)
    ok = _finite_mask(X)
    if not ok.all():
        print(f"[maplight] dropping {int((~ok).sum())}/{len(smiles)} non-finite rows")
    return X[ok], ok


def fit_eval_catboost(X_tr, y_tr, X_te, y_te, task, seed):
    """CatBoost analogue of baselines_ml.fit_eval -- IDENTICAL metric keys."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        mean_squared_error, mean_absolute_error,
    )
    common = dict(iterations=500, depth=6, learning_rate=0.05,
                  random_seed=seed, thread_count=_NJOBS, verbose=False)
    if task == "classification" and y_tr.ndim == 2:
        from catboost import CatBoostClassifier
        n_tasks = y_tr.shape[1]
        roc, pr, f1 = [], [], []
        for t in range(n_tasks):
            ytr, yte = y_tr[:, t].astype(int), y_te[:, t].astype(int)
            if len(np.unique(ytr)) < 2:
                roc.append(float("nan")); pr.append(float("nan")); f1.append(float("nan")); continue
            m = CatBoostClassifier(auto_class_weights="Balanced", **common); m.fit(X_tr, ytr)
            proba = m.predict_proba(X_te)[:, 1]; pred = (proba >= 0.5).astype(int)
            if len(np.unique(yte)) < 2:
                roc.append(float("nan")); pr.append(float("nan"))
            else:
                roc.append(float(roc_auc_score(yte, proba))); pr.append(float(average_precision_score(yte, proba)))
            f1.append(float(f1_score(yte, pred, zero_division=0)))
        macro = lambda xs: float(np.nanmean(xs)) if any(not np.isnan(v) for v in xs) else float("nan")
        return {"roc_auc": macro(roc), "pr_auc": macro(pr), "f1": macro(f1),
                "per_task_roc_auc": roc, "per_task_pr_auc": pr, "per_task_f1": f1,
                "n_tasks": n_tasks, "n_valid_roc_tasks": int(sum(not np.isnan(v) for v in roc))}
    if task == "classification":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(auto_class_weights="Balanced", **common); m.fit(X_tr, y_tr.astype(int))
        proba = m.predict_proba(X_te)[:, 1]; pred = (proba >= 0.5).astype(int)
        return {"roc_auc": float(roc_auc_score(y_te, proba)),
                "pr_auc": float(average_precision_score(y_te, proba)),
                "f1": float(f1_score(y_te, pred, zero_division=0))}
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(**common); m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    return {"rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
            "mae": float(mean_absolute_error(y_te, pred))}


def run(name, cfg, feat_name, featurizer, model, seeds):
    smi, y, is_train, is_test = load_split(cfg)
    valid = np.array([Chem.MolFromSmiles(s) is not None for s in smi], dtype=bool)
    smi = [s for s, v in zip(smi, valid) if v]
    y, is_train, is_test = y[valid], is_train[valid], is_test[valid]

    t0 = time.time()
    X, ok = featurizer(smi)
    if not ok.all():
        y, is_train, is_test = y[ok], is_train[ok], is_test[ok]
    tr, te = np.flatnonzero(is_train), np.flatnonzero(is_test)
    evalfn = fit_eval_catboost if model == "catboost" else fit_eval

    out = dict(dataset=name, task=cfg["task"], feature=feat_name, model=model,
               n=int(X.shape[0]), n_train=int(len(tr)), n_test=int(len(te)),
               split_csv=cfg["split_csv"], feat_dim=int(X.shape[1]), seeds={})
    for s in seeds:
        out["seeds"][str(s)] = evalfn(X[tr], y[tr], X[te], y[te], cfg["task"], s)
    first = out["seeds"][str(seeds[0])]
    keys = [k for k, v in first.items()
            if isinstance(v, (int, float)) and k not in ("n_train", "n_test", "n_tasks")]
    out["mean"] = {k: float(np.nanmean([out["seeds"][str(s)][k] for s in seeds])) for k in keys}
    out["std"] = {k: float(np.nanstd([out["seeds"][str(s)][k] for s in seeds], ddof=1)) for k in keys}
    print(f"[{name}/{feat_name}/{model}] n={out['n']} tr={len(tr)} te={len(te)} "
          f"dim={out['feat_dim']} ({time.time()-t0:.0f}s)  " +
          "  ".join(f"{k}={out['mean'][k]:.4f}+/-{out['std'][k]:.4f}" for k in keys))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--skip_maplight", action="store_true")
    ap.add_argument("--out", default=str(REPO / "baselines_skfp_maplight_results.json"))
    a = ap.parse_args()
    seeds = list(range(a.n_seeds))

    skfp_feats = make_skfp_featurizers()
    results = {}
    for name in a.datasets:
        if name not in DATASETS:
            print(f"skip unknown dataset {name}"); continue
        if not (REPO / DATASETS[name]["split_csv"]).exists():
            print(f"[{name}] split CSV missing ({DATASETS[name]['split_csv']}); skipping"); continue
        cfg = DATASETS[name]
        results.setdefault(name, {})
        # (1) skfp sweep + RF
        for feat_name, featurizer in skfp_feats.items():
            try:
                results[name][f"rf_{feat_name}"] = run(name, cfg, feat_name, featurizer, "rf", seeds)
            except Exception as e:
                print(f"[{name}/{feat_name}/rf] FAILED: {type(e).__name__}: {e}")
        # (2) MapLight fusion + CatBoost
        if not a.skip_maplight:
            try:
                results[name]["catboost_maplight"] = run(name, cfg, "maplight", featurize_maplight, "catboost", seeds)
            except Exception as e:
                print(f"[{name}/maplight/catboost] FAILED: {type(e).__name__}: {e}")
        Path(a.out).write_text(json.dumps(results, indent=2))  # checkpoint per dataset
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
