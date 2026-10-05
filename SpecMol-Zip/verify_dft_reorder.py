"""verify_dft_reorder.py -- validate the Solution-B atom map for Stage-2.

Solution B keeps the graph IDENTICAL to the MMFF cell (built from SMILES,
canonical order, SMILES aromaticity) and changes ONLY the coordinates Uni-Mol
sees: we reorder the gdb9 DFT coords onto the canonical AddHs(MolFromSmiles)
atom order. That reorder needs an atom map between two RDKit mols of the SAME
molecule that may differ in bond perception (gdb9 SDF Kekule/3D-perceived vs
SMILES aromatic) and H placement.

This script validates the map for every cached molecule BEFORE we build the
pipeline: build mol_h = AddHs(MolFromSmiles(s)); read the gdb9 DFT mol; map
gdb9 atom -> mol_h atom; transfer DFT coords onto mol_h; run the physical
bond-length gate on the RESULT. ~100% pass => Solution B is safe.

Mapping strategy (per molecule, first that works):
  1. canonical-rank map (Chem.CanonicalRankAtoms, breakTies) -- representation
     invariant, deterministic.
  2. generic-bond GetSubstructMatch fallback (bonds made generic so aromatic vs
     Kekule never blocks the match).
Either way the bond-length gate is the final arbiter.

Run: D:\\SpecMol-Zip\\.venv\\Scripts\\python.exe verify_dft_reorder.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem  # noqa: F401  (AddHs lives on Chem; import keeps parity)

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent
CACHE = REPO / "data" / "qm9" / "qm9_dft_lookup.pkl"
MANIFEST = REPO / "unimol_out_qm9mu" / "manifest.csv"
OUT_REPORT = REPO / "data" / "qm9" / "qm9_dft_reorder_report.json"
OUT_SOLB = REPO / "data" / "qm9" / "qm9_dft_solB_cache.pkl"
BOND_LO, BOND_HI = 0.9, 2.0


def canonical_map(gdb9, mol_h):
    """map[k] = mol_h atom index for gdb9 atom k, via canonical ranks; or None."""
    if gdb9.GetNumAtoms() != mol_h.GetNumAtoms():
        return None
    try:
        gr = list(Chem.CanonicalRankAtoms(gdb9, breakTies=True))
        mr = list(Chem.CanonicalRankAtoms(mol_h, breakTies=True))
    except Exception:
        return None
    if sorted(gr) != sorted(mr):
        return None
    m_rank2idx = {r: i for i, r in enumerate(mr)}
    try:
        return [m_rank2idx[gr[k]] for k in range(gdb9.GetNumAtoms())]
    except KeyError:
        return None


def generic_match_map(gdb9, mol_h):
    """Fallback: generic-bond substructure match (gdb9 as query) -> mol_h idx."""
    try:
        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeBondsGeneric = True
        q = Chem.AdjustQueryProperties(Chem.Mol(gdb9), params)
        match = mol_h.GetSubstructMatch(q)  # match[k] = mol_h idx for query atom k
        if len(match) == gdb9.GetNumAtoms():
            return list(match)
    except Exception:
        pass
    return None


def transfer_coords(gdb9, mol_h, amap):
    """Put gdb9 DFT coords onto mol_h in mol_h's atom order; return mol_h w/ conf."""
    gconf = gdb9.GetConformer()
    conf = Chem.Conformer(mol_h.GetNumAtoms())
    for k in range(gdb9.GetNumAtoms()):
        p = gconf.GetAtomPosition(k)
        conf.SetAtomPosition(amap[k], p)
    mh = Chem.Mol(mol_h)
    mh.RemoveAllConformers()
    mh.AddConformer(conf, assignId=True)
    # element check: mapped atom symbols must agree (guards a bad map)
    for k in range(gdb9.GetNumAtoms()):
        if gdb9.GetAtomWithIdx(k).GetAtomicNum() != mh.GetAtomWithIdx(amap[k]).GetAtomicNum():
            return None
    return mh


def bond_violations(mol):
    conf = mol.GetConformer()
    bad = 0
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        d = ((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2 + (pi.z - pj.z) ** 2) ** 0.5
        if d < BOND_LO or d > BOND_HI:
            bad += 1
    return bad, mol.GetNumBonds()


def main():
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    man = pd.read_csv(MANIFEST)

    n = ok = 0
    via_canon = via_generic = 0
    no_map, gate_fail, elem_fail = [], [], []
    tot_bad = tot_bonds = 0
    solb_cache = {}   # smiles -> canonical-order mol_h MolBlock w/ DFT conformer
    for _, r in man.iterrows():
        s = str(r["smiles"])
        rec = cache.get(s)
        if rec is None:
            continue
        n += 1
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            no_map.append(s); continue
        mol_h = Chem.AddHs(mol)
        gdb9 = Chem.MolFromMolBlock(rec["molblock"], removeHs=False, sanitize=True)
        if gdb9 is None:
            no_map.append(s); continue
        amap = generic_match_map(gdb9, mol_h)
        used = "generic"
        if amap is None:
            amap = canonical_map(gdb9, mol_h); used = "canon"
        if amap is None:
            no_map.append(s); continue
        mh = transfer_coords(gdb9, mol_h, amap)
        if mh is None:
            elem_fail.append(s); continue
        bad, nb = bond_violations(mh)
        tot_bad += bad; tot_bonds += nb
        if bad > 0:
            gate_fail.append({"smiles": s, "bad": bad, "n": nb, "via": used}); continue
        ok += 1
        via_canon += (used == "canon")
        via_generic += (used == "generic")
        # canonical-order mol_h carrying DFT coords -> Solution-B cache (HPC-consumable)
        solb_cache[s] = Chem.MolToMolBlock(mh, kekulize=False)

    report = {
        "n_cached_in_manifest": n,
        "reorder_ok": ok,
        "reorder_ok_frac": round(ok / n, 4) if n else None,
        "via_canonical": via_canon,
        "via_generic_fallback": via_generic,
        "no_map": len(no_map),
        "element_mismatch": len(elem_fail),
        "bond_gate_fail": len(gate_fail),
        "total_bad_bonds": tot_bad,
        "total_bonds": tot_bonds,
        "bad_bond_frac": round(tot_bad / tot_bonds, 8) if tot_bonds else None,
        "no_map_examples": no_map[:10],
        "gate_fail_examples": gate_fail[:10],
    }
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    with open(OUT_SOLB, "wb") as f:
        pickle.dump(solb_cache, f)

    print("\n===== SOLUTION-B REORDER VALIDATION =====")
    print(f"cached-in-manifest : {n}")
    print(f"reorder OK (gate)  : {ok} ({ok/n:.2%})  [canon={via_canon} generic={via_generic}]")
    print(f"no_map={len(no_map)}  elem_mismatch={len(elem_fail)}  gate_fail={len(gate_fail)}")
    print(f"bad bonds          : {tot_bad}/{tot_bonds}")
    print(f"solB cache         : {OUT_SOLB}  ({len(solb_cache)} mols, canonical order + DFT coords)")
    print(f"report             : {OUT_REPORT}")
    verdict = (n and ok / n > 0.98 and tot_bad == 0)
    print(f"VERDICT            : {'PASS -- Solution B safe (graph identical to MMFF, only coords change)' if verdict else 'INVESTIGATE'}")


if __name__ == "__main__":
    main()
