"""Screen all 12 QM9 targets to find the GEOMETRY battlefield.

For each target, compare:
  - weak-RF  : 11 composition descriptors (MolWt/LogP/TPSA/ring counts...)
  - strong-RF: full ~210 RDKit descriptors (incl partial-charge: MaxPartialCharge...)
z-scored per target so RMSEs are comparable across targets (dummy=mean -> ~1.0).

Readout per target:
  charge_gain = weak_rmse - strong_rmse  (how much charge/full descriptors add over
                composition). SMALL => charge/descriptors irrelevant.
  skill       = 1 - strong_rmse/dummy    (how much the BEST descriptor-RF explains).
                LOW => descriptors FAIL -> a geometry/3D model has room.

GEOMETRY CANDIDATE = small charge_gain (charge doesn't help) AND low skill
(descriptors barely beat the mean). On mu we expect the opposite (charge wins).

Usage: python qm9_target_screen.py [n_seeds=3] [n_trees=150]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parent

TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv",
           "u0", "u298", "h298", "g298"]
WEAK11 = ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
          "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
          "HeavyAtomCount", "FractionCSP3", "RingCount"]
STRONG = Descriptors._descList  # list of (name, fn), ~210 incl MaxPartialCharge
STRONG_NAMES = [n for n, _ in STRONG]

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_TREES = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def featurize(smiles):
    X = np.zeros((len(smiles), len(STRONG)), dtype=np.float32)
    ok = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        ok[i] = True
        for j, (_, fn) in enumerate(STRONG):
            try:
                X[i, j] = fn(m)
            except Exception:
                X[i, j] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), ok


def main():
    qm9 = pd.read_csv(REPO / "unimol_out_qm9mu" / "qm9.csv").drop_duplicates("smiles")
    man = pd.read_csv(REPO / "unimol_out_qm9mu" / "manifest.csv").drop_duplicates("smiles")
    df = man.merge(qm9[["smiles"] + TARGETS], on="smiles", how="left").dropna(subset=TARGETS)
    df = df.reset_index(drop=True)
    smi = df["smiles"].tolist()
    split = df["split"].to_numpy()
    print(f"merged {len(df)} molecules (manifest subset x qm9 12 targets); "
          f"n_seeds={N_SEEDS} n_trees={N_TREES}")

    t0 = time.time()
    Xfull, ok = featurize(smi)
    print(f"featurized {Xfull.shape} in {time.time()-t0:.0f}s; {int((~ok).sum())} unparsable")
    name_to_idx = {n: i for i, n in enumerate(STRONG_NAMES)}
    weak_idx = [name_to_idx[n] for n in WEAK11 if n in name_to_idx]
    Xw = Xfull[:, weak_idx]
    tr = (split != "test") & ok
    te = (split == "test") & ok
    print(f"train={int(tr.sum())} test={int(te.sum())} weak_feats={Xw.shape[1]} strong_feats={Xfull.shape[1]}")

    def rf_rmse(X, yz):
        rs = []
        for s in range(N_SEEDS):
            r = RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=s)
            r.fit(X[tr], yz[tr])
            p = r.predict(X[te])
            rs.append(float(np.sqrt(np.mean((yz[te] - p) ** 2))))
        return float(np.mean(rs))

    rows = []
    for tgt in TARGETS:
        y = df[tgt].to_numpy(dtype=np.float64)
        ymu, ysd = y[tr].mean(), y[tr].std() + 1e-12
        yz = (y - ymu) / ysd
        dummy = float(np.sqrt(np.mean((yz[te] - 0.0) ** 2)))
        w, s = rf_rmse(Xw, yz), rf_rmse(Xfull, yz)
        rows.append((tgt, dummy, w, s, w - s, 1 - s / dummy))
        print(f"  {tgt:6s} dummy={dummy:.3f} weakRF={w:.3f} strongRF={s:.3f} "
              f"charge_gain={w - s:+.3f} skill={1 - s / dummy:.3f}", flush=True)

    print("=" * 74)
    print(f"{'target':6s} {'dummy':>6s} {'weakRF':>7s} {'strongRF':>8s} {'chargeD':>8s} {'skill':>6s}   (sorted: descriptors-fail first)")
    for r in sorted(rows, key=lambda r: (r[5], r[4])):
        flag = "  <-- GEOMETRY CANDIDATE" if (r[5] < 0.5 and r[4] < 0.1) else ""
        print(f"{r[0]:6s} {r[1]:6.3f} {r[2]:7.3f} {r[3]:8.3f} {r[4]:+8.3f} {r[5]:6.3f}{flag}")
    print("GEOMETRY CANDIDATE = skill<0.5 (descriptor-RF barely beats mean) AND chargeD<0.1 "
          "(charge/full descriptors add little over composition) -> 3D's real opportunity.")
    print("QM9_SCREEN_DONE")


if __name__ == "__main__":
    main()
