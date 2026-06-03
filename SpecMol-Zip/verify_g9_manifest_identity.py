#!/usr/bin/env python
"""G9 split-alignment verifier (stdlib only -- runs locally OR on HPC).

The DFT geometry result (deep T8 RMSE 0.882 vs charge-RF 0.925 on QM9-mu) is only
valid if the deep model and the RF baseline were evaluated on the IDENTICAL test
molecules. Two manifests exist:
  - unimol_out_qm9mu/manifest.csv      (MMFF conformers)
  - unimol_out_qm9mu_dft/manifest.csv  (true DFT coordinates)
They share the SAME molecules/labels/split assignment (by sdf_index); they differ
ONLY in `embed_ok` -- DFT embedding drops more molecules, so the DFT test set is a
SUBSET of the MMFF test set.

deep-T8(DFT) is built from down_task_qm9mu_unimol_v2_dft = DFT manifest test rows.
For a VALID comparison the charge-RF must ALSO be on the DFT manifest test rows.
`qm9mu_rf_matched.py` DEFAULTS to the MMFF dir, so a default invocation gives the
WRONG (MMFF) test set. The RF job 139003 printed `n_test=1473` -> that is the MMFF
count (DFT test = 1256) -> RED FLAG that RF was on MMFF, not DFT.

Usage:
  python verify_g9_manifest_identity.py
  python verify_g9_manifest_identity.py <mmff_manifest> <dft_manifest>
"""
import csv
import sys


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_true(v):
    return str(v).strip().lower() == "true"


def test_embed_smiles(rows):
    return {r["smiles"] for r in rows if r["split"] == "test" and is_true(r["embed_ok"])}


def summarize(rows, name):
    n = len(rows)
    embed = sum(1 for r in rows if is_true(r["embed_ok"]))
    test_all = sum(1 for r in rows if r["split"] == "test")
    test_emb = len(test_embed_smiles(rows))
    print(f"[{name}] rows={n}  embed_ok={embed}  test(all)={test_all}  test(embed_ok)={test_emb}")
    return {"n": n, "embed": embed, "test_all": test_all, "test_emb": test_emb}


def main():
    mmff = sys.argv[1] if len(sys.argv) > 1 else "unimol_out_qm9mu/manifest.csv"
    dft = sys.argv[2] if len(sys.argv) > 2 else "unimol_out_qm9mu_dft/manifest.csv"
    print("=" * 70)
    print("G9 split-alignment verification")
    print("=" * 70)
    rm = load(mmff)
    rd = load(dft)
    sm = summarize(rm, "MMFF " + mmff)
    sd = summarize(rd, "DFT  " + dft)

    # 1) Same underlying molecules + split assignment across the two manifests?
    key_m = [(r["sdf_index"], r["smiles"], r["split"]) for r in rm]
    key_d = [(r["sdf_index"], r["smiles"], r["split"]) for r in rd]
    same_mol_split = key_m == key_d
    print("-" * 70)
    print(f"same (sdf_index, smiles, split) ordering across manifests: {same_mol_split}")
    if not same_mol_split:
        # fall back to set comparison if order differs
        set_m = set((r["smiles"], r["split"]) for r in rm)
        set_d = set((r["smiles"], r["split"]) for r in rd)
        print(f"  set-level (smiles,split) identical: {set_m == set_d}")

    # 2) DFT test set vs MMFF test set (the molecules each evaluation actually uses)
    tm = test_embed_smiles(rm)
    td = test_embed_smiles(rd)
    inter = tm & td
    print("-" * 70)
    print(f"MMFF test(embed_ok) = {len(tm)}   DFT test(embed_ok) = {len(td)}")
    print(f"  |DFT n MMFF| = {len(inter)}   |DFT only| = {len(td - tm)}   |MMFF only| = {len(tm - td)}")
    dft_subset = td <= tm
    print(f"  DFT test set is a SUBSET of MMFF test set: {dft_subset}  "
          f"(MMFF drops {len(tm - td)} extra mols that failed DFT embedding)")

    # 3) Verdict
    print("=" * 70)
    print("VERDICT")
    print(f"  deep-T8(DFT) test set  = DFT manifest test(embed_ok) = {sd['test_emb']} molecules")
    print(f"  charge-RF MUST be run on this SAME {sd['test_emb']}-molecule set.")
    print(f"  If the RF run reported n_test={sm['test_emb']} (=MMFF), it used the WRONG manifest")
    print(f"    -> re-run: python qm9mu_rf_matched.py unimol_out_qm9mu_dft \"<deep ref>\"")
    print(f"  STILL TO CONFIRM (HPC logs): which manifest the charge-RF job (139003) actually read.")
    print("=" * 70)
    # machine-readable summary line
    print(f"G9_SUMMARY mmff_test={sm['test_emb']} dft_test={sd['test_emb']} "
          f"dft_subset_of_mmff={dft_subset} same_mol_split={same_mol_split}")


if __name__ == "__main__":
    main()
