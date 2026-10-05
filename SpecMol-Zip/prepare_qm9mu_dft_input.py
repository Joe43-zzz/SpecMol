"""prepare_qm9mu_dft_input.py -- Stage-2 DFT cell SDF/manifest builder (Solution B).

Writes the Uni-Mol input for the TRUE-DFT-geometry QM9-mu cell. Reuses the EXACT
pilot molecule set + split (unimol_out_qm9mu/manifest.csv) and swaps ONLY the 3D
coordinates: each molecule's SDF entry carries true GDB-9 DFT coordinates,
reordered onto the canonical AddHs(MolFromSmiles) atom order (Solution B). The 2D
graph (built later from SMILES) is therefore IDENTICAL to the MMFF cell -- only
the geometry Uni-Mol sees differs, so the DFT-vs-MMFF contrast is clean.

DFT coords come from the validated cache data/qm9/qm9_dft_solB_cache.pkl
({smiles -> canonical-order mol_h MolBlock w/ DFT conformer}), built+gated by
build_qm9_dft_lookup.py + verify_dft_reorder.py (0 nonphysical bonds). The ~16%
of molecules where the published GDB-9 SDF connectivity disagrees with the SMILES
are absent from the cache and conservatively dropped (embed_ok=False).

Outputs: unimol_out_qm9mu_dft/{sdf/qm9mu_for_unimol.sdf, manifest.csv, splits/}.
"""
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

REPO = Path(__file__).resolve().parent
SRC_MANIFEST = REPO / "unimol_out_qm9mu" / "manifest.csv"
SOLB_CACHE = REPO / "data" / "qm9" / "qm9_dft_solB_cache.pkl"
OUT_ROOT = REPO / "unimol_out_qm9mu_dft"
SDF_DIR = OUT_ROOT / "sdf"
SPLITS_DIR = OUT_ROOT / "splits"
MANIFEST_CSV = OUT_ROOT / "manifest.csv"
SDF_NAME = "qm9mu_for_unimol.sdf"


def main():
    if not SRC_MANIFEST.exists():
        raise SystemExit(f"missing {SRC_MANIFEST} (the MMFF pilot manifest defining the molecule set + split)")
    if not SOLB_CACHE.exists():
        raise SystemExit(f"missing {SOLB_CACHE} -- run build_qm9_dft_lookup.py then verify_dft_reorder.py")

    man = pd.read_csv(SRC_MANIFEST)
    with open(SOLB_CACHE, "rb") as f:
        cache = pickle.load(f)                  # {smiles -> molblock(str)}; version-portable
    SDF_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(SDF_DIR / SDF_NAME))
    rows, sdf_index, n_drop, n_bad = [], 0, 0, 0
    for r in man.itertuples():
        s = str(r.smiles)
        rec = {"smiles": s, "label": float(r.label), "split": r.split}
        mb = cache.get(s)
        mol = Chem.MolFromMolBlock(mb, removeHs=False, sanitize=True) if mb is not None else None
        if mol is None or mol.GetNumConformers() == 0:
            n_drop += (mb is None); n_bad += (mb is not None and mol is None)
            rows.append({**rec, "sdf_index": -1, "embed_ok": False})
            continue
        mol.SetProp("_Name", s)
        writer.write(mol)
        rows.append({**rec, "sdf_index": sdf_index, "embed_ok": True})
        sdf_index += 1
    writer.close()

    out = pd.DataFrame(rows)
    out.to_csv(MANIFEST_CSV, index=False)
    out.to_csv(SPLITS_DIR / "qm9mu_split_seed42.csv", index=False)
    print(f"QM9mu-DFT: {len(out)} total, {int(out['embed_ok'].sum())} embedded (DFT), "
          f"{n_drop} not-in-cache, {n_bad} molblock-parse-fail")
    for sp in ("train", "valid", "test"):
        sub = out[(out["split"] == sp) & out["embed_ok"]]
        print(f"  {sp}: {len(sub)}")


if __name__ == "__main__":
    main()
