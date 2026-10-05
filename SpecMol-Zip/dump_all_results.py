"""Exhaustive dump of every experiment result, for cross-checking the paper +
brief. Uses make_tables.py's OWN parsers so the cls/reg numbers are exactly the
canonical paper numbers (recomputed from raw per-seed JSONs), then dumps the
QM / finetune-sweep / audit JSONs and lists every *result* file so nothing is
missed. Read-only; writes nothing.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper"))
import make_tables as mt
load = mt.load


def build_canonical():
    bbbp = mt.parse_bbbp(load("bbbp_all_results.json") or {})
    bace = mt.parse_bace_v0_fp(load("baseline_unifold_results.json") or {})
    bf = load("bace_all_results.json")
    if bf:
        bace.update(mt.parse_bace_full(bf))
    clintox = mt.parse_bace_full(load("clintox_all_results.json") or {})
    freesolv = mt.parse_regression_matrix("freesolv")
    esol = mt.parse_regression_matrix("esol")
    lipo = mt.parse_regression_matrix("lipo")
    tables = dict(bbbp=bbbp, bace=bace, clintox=clintox,
                  freesolv=freesolv, esol=esol, lipo=lipo)

    rf_raw = load("baselines_ml_results_deng30_unimol_fold.json") or load("baselines_ml_results_unimol_fold.json")
    if rf_raw:
        rf = mt.parse_rf_unimol_fold(rf_raw)
        for ds, t in tables.items():
            if ds in rf:
                t["RF"] = rf[ds]
    cp_raw = load("baselines_chemprop_results_unimol_fold.json")
    if cp_raw:
        cp = mt.parse_chemprop_unimol_fold(cp_raw)
        for ds, t in tables.items():
            if ds in cp:
                t["Chemprop"] = cp[ds]
    m_raw = load("baselines_matched_results.json")
    if m_raw:
        rfm, um = mt.parse_matched_baselines(m_raw)
        for ds, t in tables.items():
            if ds in rfm:
                t["RF"] = rfm[ds]
            if ds in um:
                t["Uni-Mol-emb+RF"] = um[ds]
    ft_raw = load("unimol_finetune_results.json")
    if ft_raw:
        ft = mt.parse_unimol_finetune(ft_raw)
        for ds, t in tables.items():
            if ds in ft:
                t["Uni-Mol (FT)"] = ft[ds]
    for t, fn in ((bbbp, "bbbp_t7_bare_results.json"), (bace, "bace_t7_bare_results.json"),
                  (clintox, "clintox_t7_bare_results.json")):
        if "T7" not in t:
            r = load(fn)
            s = mt.parse_t7_collect_format(r) if r else None
            if s:
                t["T7"] = s
    for t, fn in ((freesolv, "freesolv_t7_bare_results.json"), (esol, "esol_t7_bare_results.json"),
                  (lipo, "lipo_t7_bare_results.json")):
        if "T7" not in t:
            r = load(fn)
            if r:
                s = mt._freesolv_per_seed(r)
                if s:
                    t["T7"] = mt._stats(s)
    for t, fn in ((bbbp, "bbbp_t8_bare_results.json"), (bace, "bace_t8_bare_results.json"),
                  (clintox, "clintox_t8_bare_results.json")):
        if "T8" not in t:
            r = load(fn)
            s = mt.parse_t7_collect_format(r) if r else None
            if s:
                t["T8"] = s
    return tables


def fmt(v):
    try:
        return f"{v[0]:.4f} +/- {v[1]:.4f}"
    except Exception:
        return str(v)


print("=" * 78)
print("CANONICAL PAPER CELLS (make_tables parsers, recomputed from raw per-seed JSONs)")
print("=" * 78)
tables = build_canonical()
for name, d in tables.items():
    print(f"\n### {name}")
    for k in ["2D-Only", "V0", "FP-only", "RF", "Chemprop", "Uni-Mol-emb+RF",
              "Uni-Mol (FT)", "SEG", "V2-T5", "PBA", "T7", "APC", "T8", "T6", "APC-logits"]:
        if k in d:
            print(f"  {k:16} {fmt(d[k])}")

print("\n" + "=" * 78)
print("QM / FINETUNE-SWEEP / FEATURIZATION AUDIT JSONs (raw)")
print("=" * 78)
for fn in ["paper/audits/inductive_sweep_2026-06-01.json",
           "paper/audits/qm_pilot_2026-06-02.json",
           "paper/audits/stage2_qm9_dft_2026-06-03.json",
           "paper/audits/b4_atom_dropout.json",
           "paper/audits/spectral_readout.json"]:
    p = ROOT / fn
    if p.exists():
        print(f"\n--- {fn} ---")
        print(json.dumps(json.load(open(p, encoding="utf-8")), indent=1))
    else:
        print(f"\n--- {fn} --- MISSING")

print("\n" + "=" * 78)
print("EVERY *result*/qm*/nullify* JSON top-level keys (catch anything not above)")
print("=" * 78)
seen = set()
for p in sorted(ROOT.glob("*.json")):
    nm = p.name.lower()
    if not any(t in nm for t in ("result", "qm7", "qm9", "nullify", "audit", "finetune", "matched")):
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
        if isinstance(d, dict):
            print(f"  {p.name:52} keys={list(d.keys())}")
        else:
            print(f"  {p.name:52} <{type(d).__name__} len={len(d)}>")
    except Exception as e:
        print(f"  {p.name:52} ERR {type(e).__name__}: {e}")
