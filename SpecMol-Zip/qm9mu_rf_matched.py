"""Matched RF baseline for QM9-mu (gate-b), per code-review fixes:
  - matched split: deep pilot's EXACT split (<manifest_dir>/manifest.csv),
    same embed_ok molecules, train=train, test=test.
  - two RFs: weak (11 composition descriptors, ADMET/QM7-consistent) AND
    strong (full RDKit descriptor list, incl. partial-charge descriptors that
    CAN correlate with dipole) -- so "RF collapses on mu" is not an artifact of
    a charge-blind 11-descriptor baseline.

REPRO HYGIENE (2026-06-03): the manifest dir is now a REQUIRED argument.
There is NO silent default to unimol_out_qm9mu, because a no-arg run computes
the MMFF cell (n_test=1473, strong RF 0.925) and that value CONTRADICTS the
TRUE-DFT cell (n_test=1256, strong RF 0.845): the DFT embed_ok subset is a
strict (smaller) subset of the MMFF one, so the two RFs are NOT the same number
"by construction". See verify_g9_manifest_identity.py and
paper/STAGE2_QM9_DFT_DESIGN_2026-06-02.md. To reproduce a specific cell:
    python qm9mu_rf_matched.py unimol_out_qm9mu      # MMFF cell  (weak 0.986 / strong 0.925)
    python qm9mu_rf_matched.py unimol_out_qm9mu_dft  # TRUE-DFT   (weak 0.939 / strong 0.845)

The deep arms run on the SAME manifest/test split; this script no longer hard-codes
any deep-reference RMSE string (those drift per cell) -- pass --deep_ref to annotate
the printout if desired, but it is purely cosmetic and never written to the JSON.

Output: writes a named results JSON (default <manifest_dir>_rf_matched.json, or
--out <path>) with the per-RF means so paper/make_tables.py can consume the cell
instead of a hand-keyed value.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parent

WEAK11 = ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
          "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
          "HeavyAtomCount", "FractionCSP3", "RingCount"]
STRONG = [n for n, _ in Descriptors._descList]  # full RDKit ~210 incl MaxPartialCharge etc.

SEEDS = [9, 19, 29]


def morgan(m):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), dtype=np.int8)


def desc(m, names):
    out = []
    for n in names:
        fn = getattr(Descriptors, n)
        try:
            v = float(fn(m))
        except Exception:
            v = 0.0
        out.append(v)
    return np.array(out, dtype=np.float32)


def feat(smis, names):
    M = [Chem.MolFromSmiles(s) for s in smis]
    Xfp = np.vstack([morgan(m) for m in M])
    Xd = np.nan_to_num(np.vstack([desc(m, names) for m in M]),
                       nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([Xfp, Xd]).astype(np.float32)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Matched geometry-blind RF ceiling for QM9-mu on a deep "
                    "pilot's EXACT split. The manifest dir is REQUIRED -- there "
                    "is no MMFF default, because MMFF and DFT use different "
                    "embed_ok subsets and therefore give DIFFERENT RF numbers.")
    p.add_argument("manifest_dir",
                   help="manifest dir (path or name under repo root), e.g. "
                        "unimol_out_qm9mu (MMFF) or unimol_out_qm9mu_dft (DFT); "
                        "must contain manifest.csv with the deep pilot's split.")
    p.add_argument("--out", default=None,
                   help="output JSON path (default <manifest_dir>_rf_matched.json "
                        "under repo root).")
    p.add_argument("--deep_ref", default=None,
                   help="optional cosmetic deep-RMSE annotation for the printout "
                        "(e.g. 'V0=0.880 T8=0.882'); never written to the JSON.")
    return p.parse_args(argv)


def resolve_manifest_dir(manifest_dir: str) -> Path:
    d = Path(manifest_dir)
    if not d.is_absolute():
        # accept either a repo-relative path or a bare dir name under the repo.
        cand = REPO / manifest_dir
        d = cand if (cand / "manifest.csv").exists() or cand.exists() else d
    return d


def main(argv=None):
    args = parse_args(argv)
    mdir = resolve_manifest_dir(args.manifest_dir)
    manifest = mdir / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"[error] manifest not found: {manifest}\n"
                 f"        pass the deep pilot's manifest dir explicitly, e.g.\n"
                 f"        python {Path(__file__).name} unimol_out_qm9mu_dft")

    out_path = (Path(args.out) if args.out
                else REPO / f"{mdir.name}_rf_matched.json")

    man = pd.read_csv(manifest)
    man = man[man["embed_ok"] == True]  # noqa: E712  -- match deep's embed_ok subset
    tr = man[man["split"] == "train"]
    te = man[man["split"] == "test"]
    ytr = tr["label"].values.astype(np.float32)
    yte = te["label"].values.astype(np.float32)
    n_train, n_test = int(len(tr)), int(len(te))
    print(f"matched split [{mdir.name}]: n_train={n_train} n_test={n_test} "
          f"(deep used same embed_ok subset)")
    if args.deep_ref:
        print(f"DEEP ref (test RMSE, cosmetic): {args.deep_ref}")

    rf_cells = {}
    for tag, names in [("weak11", WEAK11), ("strongFull", STRONG)]:
        Xtr = feat(tr["smiles"].tolist(), names)
        Xte = feat(te["smiles"].tolist(), names)
        rmses, maes = [], []
        for seed in SEEDS:
            rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
            rf.fit(Xtr, ytr)
            p = rf.predict(Xte)
            rmses.append(float(np.sqrt(mean_squared_error(yte, p))))
            maes.append(float(mean_absolute_error(yte, p)))
        cell = {
            "nfeat": int(Xtr.shape[1]),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "rmse_seeds": rmses,
        }
        rf_cells[tag] = cell
        print(f"[qm9mu RF {tag}] nfeat={cell['nfeat']} "
              f"RMSE={cell['rmse_mean']:.4f}+/-{cell['rmse_std']:.4f} "
              f"MAE={cell['mae_mean']:.4f}+/-{cell['mae_std']:.4f}")

    results = {
        "manifest_dir": mdir.name,
        "manifest_path": str(manifest),
        "n_train": n_train,
        "n_test": n_test,
        "seeds": SEEDS,
        "weak_rmse": rf_cells["weak11"]["rmse_mean"],
        "weak_rmse_std": rf_cells["weak11"]["rmse_std"],
        "strong_rmse": rf_cells["strongFull"]["rmse_mean"],
        "strong_rmse_std": rf_cells["strongFull"]["rmse_std"],
        "detail": rf_cells,
    }
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_path}")
    print("RF_QM9MU_DONE")


if __name__ == "__main__":
    main()
