"""B4 bug forensic audit: quantify atom-level out-degree-zero rate.

The pre-B4 code in create_data_DC.py:get_edge_features emitted ONE directed
edge [begin, end] per chemical bond, omitting the reverse [end, begin]. In a
symmetric Laplacian (get_laplacian('sym')), an atom that never appears as
`begin` in any bond has out-degree 0, gets D^(-1/2) = 0, and is effectively
disconnected from spectral message passing.

This script reconstructs the pre-B4 directed edge_index from raw SMILES and
computes:
    - per-molecule: count atoms with out-degree 0
    - per-dataset: aggregate fraction of atoms zeroed, fraction of molecules
      with at least one zeroed atom, and a histogram of zeroed atom symbols

Output: paper/audits/b4_atom_dropout.json (and a short text summary on stdout).
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "audits" / "b4_atom_dropout.json"

DATASETS = {
    "bbbp": {
        "csv": REPO / "dataset" / "BBBP.csv",
        "smiles_col": "smiles",
        "has_header": True,
    },
    "bace": {
        "csv": REPO / "dataset" / "bace.csv",
        "smiles_col": "smiles",
        "has_header": True,
    },
    "freesolv": {
        "csv": REPO / "dataset" / "freesolv" / "raw" / "smiles.csv",
        "smiles_col": 0,
        "has_header": False,
    },
}


def iter_smiles(cfg: dict) -> Iterable[str]:
    csv_path: Path = cfg["csv"]
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8") as f:
        if cfg["has_header"]:
            reader = csv.DictReader(f)
            for row in reader:
                s = row.get(cfg["smiles_col"])
                if s:
                    yield s
        else:
            reader = csv.reader(f)
            idx = cfg["smiles_col"]
            for row in reader:
                if row and len(row) > idx:
                    yield row[idx]


def audit_molecule(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None
    out_deg = [0] * n_atoms
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        out_deg[begin] += 1
    zeroed_indices = [i for i, d in enumerate(out_deg) if d == 0]
    zeroed_symbols = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in zeroed_indices)
    return {
        "n_atoms": n_atoms,
        "n_zeroed": len(zeroed_indices),
        "frac_zeroed": len(zeroed_indices) / n_atoms,
        "zeroed_symbols": zeroed_symbols,
    }


def audit_dataset(name: str, cfg: dict) -> dict:
    n_mols = 0
    n_mols_failed = 0
    n_mols_with_zero = 0
    total_atoms = 0
    total_zeroed = 0
    per_mol_frac: list[float] = []
    symbol_hist: Counter = Counter()

    for smi in iter_smiles(cfg):
        res = audit_molecule(smi)
        if res is None:
            n_mols_failed += 1
            continue
        n_mols += 1
        total_atoms += res["n_atoms"]
        total_zeroed += res["n_zeroed"]
        per_mol_frac.append(res["frac_zeroed"])
        if res["n_zeroed"] > 0:
            n_mols_with_zero += 1
        symbol_hist.update(res["zeroed_symbols"])

    if n_mols == 0:
        return {"error": f"no valid SMILES found at {cfg['csv']}"}

    return {
        "dataset": name,
        "csv_path": str(cfg["csv"].relative_to(REPO)),
        "n_molecules_parsed": n_mols,
        "n_molecules_failed_parse": n_mols_failed,
        "n_molecules_with_any_zeroed_atom": n_mols_with_zero,
        "frac_molecules_with_any_zeroed_atom": round(n_mols_with_zero / n_mols, 4),
        "total_atoms": total_atoms,
        "total_atoms_zeroed_outdeg": total_zeroed,
        "global_frac_atoms_zeroed": round(total_zeroed / total_atoms, 4),
        "per_molecule_frac_zeroed_mean": round(statistics.mean(per_mol_frac), 4),
        "per_molecule_frac_zeroed_median": round(statistics.median(per_mol_frac), 4),
        "per_molecule_frac_zeroed_std": round(
            statistics.stdev(per_mol_frac) if len(per_mol_frac) > 1 else 0.0, 4
        ),
        "zeroed_symbol_histogram_top10": symbol_hist.most_common(10),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, cfg in DATASETS.items():
        print(f"[audit] {name}: csv={cfg['csv']}  exists={cfg['csv'].exists()}")
        results[name] = audit_dataset(name, cfg)
        r = results[name]
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(
            f"  n_molecules={r['n_molecules_parsed']:>5}  "
            f"global_frac_atoms_zeroed={r['global_frac_atoms_zeroed']:.4f}  "
            f"frac_mols_with_zeroed={r['frac_molecules_with_any_zeroed_atom']:.4f}  "
            f"top_zeroed_symbols={r['zeroed_symbol_histogram_top10'][:5]}"
        )

    results["_meta"] = {
        "description": "Pre-B4 audit: fraction of heavy atoms with out-degree 0 in "
                       "symmetric Laplacian under unidirectional [begin, end] bond edges.",
        "method": "iterate mol.GetBonds(), record only [GetBeginAtomIdx, GetEndAtomIdx], "
                  "compute per-atom out-degree as count of times the atom appears as begin.",
        "bug_fix_commit": "909966bd (2026-05-13)",
        "fix_diff_lines": "create_data_DC.py:65-70 now appends [end, begin] alongside [begin, end].",
    }

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
