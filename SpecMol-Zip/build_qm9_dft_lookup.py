"""build_qm9_dft_lookup.py -- Stage-2 (QM9 DFT cell) LOCAL de-risk + lookup builder.

Goal: prove we can obtain TRUE GDB-9 DFT coordinates for the pilot's 15k QM9
molecules and align them, before committing any HPC compute. We download the raw
molnet QM9 (gdb9.sdf + gdb9.sdf.csv) and parse gdb9.sdf DIRECTLY with RDKit so we
control atom order ourselves -- sidestepping the PyG QM9 process() atom-order bug
(issues #697/#10560).

IDENTITY BRIDGE (why mol_id, not SMILES): the pilot's manifest SMILES are
SMILES-derived; the gdb9 SDF mol's connectivity is perceived from 3D geometry, so
their canonical SMILES disagree for ~79% (aromatic/conjugated). So we bridge on
the exact gdb `mol_id`: manifest SMILES --(canon(pilot qm9.csv smiles)->mol_id)-->
mol_id --(gdb9.sdf.csv row)--> gdb9 SDF mol+DFT conformer. Verified 15000/15000.

Validation checks (all must look healthy before trusting Stage-2):
  1. COVERAGE   -- fraction of manifest molecules mapped to a gdb9 DFT mol.
  2. IDENTITY   -- |manifest mu label - gdb9 mu| small for matched molecules.
  3. ALIGNMENT  -- every covalent bond length in the gdb9 DFT conformer is
                   physical (0.9-2.0 A); garbage/misaligned geometry fails this.

Outputs: data/qm9/qm9_dft_lookup.pkl  ({manifest_smiles -> molblock(+DFT conf)+mu+mol_id},
HPC-consumable) and data/qm9/qm9_dft_alignment_report.json.

Run: D:\\SpecMol-Zip\\.venv\\Scripts\\python.exe build_qm9_dft_lookup.py
(.venv has rdkit 2023.3.2 + pandas; no PyG needed.) Requires the pilot's
unimol_out_qm9mu/{manifest.csv,qm9.csv} (pull from HPC).
"""
import json
import pickle
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent
RAW = REPO / "data" / "qm9" / "raw"
QM9_ZIP_URL = ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
               "molnet_publish/qm9.zip")
UNCHAR_URL = "https://ndownloader.figshare.com/files/3195404"
MANIFEST = REPO / "unimol_out_qm9mu" / "manifest.csv"
PILOT_CSV = REPO / "unimol_out_qm9mu" / "qm9.csv"
OUT_DIR = REPO / "data" / "qm9"
OUT_CACHE = OUT_DIR / "qm9_dft_lookup.pkl"
OUT_REPORT = OUT_DIR / "qm9_dft_alignment_report.json"

BOND_LO, BOND_HI = 0.9, 2.0   # physical covalent range incl. X-H; >2.0 = misaligned


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    sdf, csv, unc = RAW / "gdb9.sdf", RAW / "gdb9.sdf.csv", RAW / "uncharacterized.txt"
    if not (sdf.exists() and csv.exists()):
        zp = RAW / "qm9.zip"
        if not zp.exists():
            print(f"[dl] {QM9_ZIP_URL} ...", flush=True)
            urllib.request.urlretrieve(QM9_ZIP_URL, zp)
        print("[dl] extracting qm9.zip ...", flush=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(RAW)
    if not unc.exists():
        try:
            urllib.request.urlretrieve(UNCHAR_URL, unc)
        except Exception as e:
            print(f"[dl] uncharacterized.txt failed ({e}); continuing without it")
    return sdf, csv, unc


def find_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def bond_violations(mol):
    """Count covalent bonds whose conformer length is nonphysical (misalignment)."""
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    bad, total = 0, 0
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        d = ((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2 + (pi.z - pj.z) ** 2) ** 0.5
        total += 1
        if d < BOND_LO or d > BOND_HI:
            bad += 1
    return bad, total


def build_canon_to_molid(pilot_csv):
    """canon(pilot qm9.csv smiles) -> mol_id, reproducing the manifest's canon."""
    q = pd.read_csv(pilot_csv)
    canon2id = {}
    for mid, s in zip(q["mol_id"], q["smiles"]):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            canon2id[Chem.MolToSmiles(m)] = str(mid)
    return canon2id


def main():
    if not PILOT_CSV.exists():
        raise SystemExit(f"missing {PILOT_CSV} -- pull from HPC "
                         f"(unimol_out_qm9mu/qm9.csv); it carries the smiles<->mol_id bridge")
    sdf_path, csv_path, _ = download()

    meta = pd.read_csv(csv_path)               # gdb9.sdf.csv: mol_id + mu, row==SDF order
    mu_col = find_col(meta.columns, "mu")
    id_col = find_col(meta.columns, "mol_id", "gdb", "name")
    molid_to_row = {str(mid): i for i, mid in enumerate(meta[id_col])}
    print(f"[csv] gdb9 rows={len(meta)} mu_col={mu_col!r} id_col={id_col!r}")

    canon2id = build_canon_to_molid(PILOT_CSV)
    print(f"[bridge] canon-smiles -> mol_id entries = {len(canon2id)}")

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    print(f"[sdf] molecules in gdb9.sdf = {len(supplier)}")

    man = pd.read_csv(MANIFEST)
    matched, no_molid, no_row, no_mol, nosan, noconf = 0, [], [], 0, 0, 0
    mu_absdiff, bad_align, align_total, align_fail = [], 0, 0, []
    cache = {}
    for _, r in man.iterrows():
        s = str(r["smiles"])
        mid = canon2id.get(s)
        if mid is None:
            no_molid.append(s); continue
        ridx = molid_to_row.get(str(mid))
        if ridx is None:
            no_row.append(s); continue
        mol = supplier[ridx]
        if mol is None:
            no_mol += 1; continue
        try:
            mm = Chem.Mol(mol); Chem.SanitizeMol(mm)
        except Exception:
            nosan += 1; continue
        if mm.GetNumConformers() == 0:
            noconf += 1; continue
        matched += 1
        gmu = float(meta.iloc[ridx][mu_col])
        mu_absdiff.append(abs(float(r["label"]) - gmu))
        bv = bond_violations(mm)
        if bv is not None:
            bad_align += bv[0]; align_total += bv[1]
            if bv[0] > 0:
                align_fail.append({"smiles": s, "mol_id": mid, "bad": bv[0], "n": bv[1]})
        cache[s] = {"molblock": Chem.MolToMolBlock(mm, kekulize=False),
                    "mu": gmu, "mol_id": mid}

    n_man = len(man)
    mu_arr = np.array(mu_absdiff) if mu_absdiff else np.array([np.nan])
    report = {
        "manifest_rows": int(n_man),
        "coverage": {
            "matched": matched,
            "coverage_frac": round(matched / n_man, 4),
            "drops": {"no_molid": len(no_molid), "no_gdb_row": len(no_row),
                      "sdf_mol_none": no_mol, "unsanitizable": nosan, "no_conformer": noconf},
            "no_molid_examples": no_molid[:10],
        },
        "identity_mu_check": {
            "n_compared": int(len(mu_absdiff)),
            "mean_abs_diff": float(np.nanmean(mu_arr)),
            "max_abs_diff": float(np.nanmax(mu_arr)),
            "frac_within_0.01": float(np.mean(mu_arr < 0.01)) if mu_absdiff else None,
            "note": "manifest label (deepchem mu) vs gdb9.sdf.csv mu; ~0 confirms same molecule",
        },
        "alignment_bondlen_gate": {
            "total_bonds_checked": align_total,
            "nonphysical_bonds": bad_align,
            "violation_frac": round(bad_align / align_total, 8) if align_total else None,
            "mols_with_any_violation": len(align_fail),
            "examples": align_fail[:10],
            "range_angstrom": [BOND_LO, BOND_HI],
            "note": "DFT conformer bonds; ~0 violations => coords real & atom-aligned",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    with open(OUT_CACHE, "wb") as f:
        pickle.dump(cache, f)

    print("\n===== STAGE-2 DFT DE-RISK REPORT =====")
    print(f"coverage      : {matched}/{n_man} = {matched/n_man:.2%} "
          f"(drops: no_molid={len(no_molid)} no_row={len(no_row)} "
          f"none={no_mol} nosan={nosan} noconf={noconf})")
    if mu_absdiff:
        print(f"identity (mu) : mean|d|={np.nanmean(mu_arr):.3g} max|d|={np.nanmax(mu_arr):.3g} "
              f"frac<0.01={np.mean(mu_arr<0.01):.3f}")
    if align_total:
        print(f"alignment gate: {bad_align}/{align_total} nonphysical bonds "
              f"({bad_align/align_total:.4%}), {len(align_fail)} mols flagged")
    print(f"cache         : {OUT_CACHE}  ({len(cache)} mols)")
    print(f"report        : {OUT_REPORT}")
    verdict = (matched / n_man > 0.98
               and (not mu_absdiff or np.nanmean(mu_arr) < 0.05)
               and (not align_total or bad_align / align_total < 0.001))
    print(f"VERDICT       : {'PASS -- approach sound, safe to build Stage-2' if verdict else 'INVESTIGATE -- see report'}")


if __name__ == "__main__":
    main()
