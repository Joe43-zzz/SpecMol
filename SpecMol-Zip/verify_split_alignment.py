"""Verify train/test split alignment across processed `.pt` data directories.

Why this exists
---------------
The whole paper rests on V0 / V2-T5 / T6 / T7 being compared on the *same*
molecules. The canonical pipeline (hpc/run_dataset_pipeline.sh) routes
  V0  -> down_task_<task>_unifold   (make_task_from_unimol --mode baseline)
  V2* -> down_task_<task>_v2        (make_task_from_unimol --mode v2)
and BOTH are supposed to read the *same* Uni-Mol fold CSV, differing only in the
presence of pair_repr edges. If they ever read different fold CSVs, every
"V0 vs V2-T5" cell silently compares different test sets and the headline tie is
an artifact. This script settles that mechanically, per dataset, with a
split-source-independent molecular identity.

Molecular identity
-------------------
We hash each molecule's **node-feature matrix `data.x`** (93-dim per atom, in
RDKit canonical atom order from create_data_DC.smile_to_graph). This is the
correct cross-dir identity because:
  * it depends only on the molecule's atoms, NOT on the split source or row
    order, and NOT on bond/edge topology -- so the B4 edge-doubling regen does
    NOT change it. Empirically the full molecule universes of two BACE dirs
    built at different times matched 1512/1512 under this key.
  * `mol_id` is a row index into whatever CSV a dir was built from, so it is the
    WRONG thing to compare across dirs (matching indices != matching molecules).

WHY NOT fps: the fingerprint vector (`data.fps`, CombinedFingerprintsFeaturizer)
is NOT stable across builds -- the same molecule came out with 353 vs 419
non-zero bits in two BACE dirs, giving a spurious 0% overlap. Never use fps as a
cross-dir identity; it is available via --identity fps only for debugging and
prints a warning.

What it reports, per task
-------------------------
  1. pairwise test-set overlap matrix (set intersection of molecular hashes);
  2. split-equivalence groups: dirs whose test sets are identical (up to
     intra-set duplicate molecules) are grouped -- mismatched dirs fall out;
  3. cross-dir leakage: |test(A) ∩ train(B)| for every ordered pair (a non-zero
     here means one dir's test molecules are another dir's training molecules);
  4. a verdict line: ALIGNED if all dirs for the task share one test set,
     else MISALIGNED with the offending groups named.

Usage
-----
  python verify_split_alignment.py                 # all tasks, all down_task* dirs
  python verify_split_alignment.py --task bace      # one task
  python verify_split_alignment.py --root . --out paper/audits/split_alignment.json

Run it on HPC too -- the decisive BACE check (down_task_bace_unifold vs
down_task_bace_v2) needs the unifold dir that only exists there.
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"torch import failed: {exc}", file=sys.stderr)
    raise

REPO = Path(__file__).resolve().parent
SPLITS = ("train", "valid", "test")
# task-name infixes we recognise in "<task>_<split>.pt"; extend as needed.
KNOWN_TASKS = ("bace", "bbbp", "hiv", "clintox", "tox21", "freesolv", "esol", "lipo")


def _seg_hashes(path, identity="x", precision=4):
    """Per-molecule identity hashes for one collated `.pt`.

    identity:
      "x"   -> hash rounded node-feature block (default; build-stable).
      "fps" -> hash rounded fingerprint block (DEBUG ONLY; build-UNSTABLE, warns).
    Returns (hashes, identity_key).
    """
    data, slices = torch.load(path, weights_only=False)
    key = "fps" if identity == "fps" else "x"
    if identity == "fps":
        print(f"  [warn] --identity fps is build-unstable; not a valid cross-dir "
              f"molecular identity. Use only for debugging.", file=sys.stderr)
    if key not in slices or not hasattr(data, key):
        raise KeyError(f"{path} has no '{key}' field for identity hashing")
    arr = getattr(data, key)
    bnd = slices[key].tolist()
    out = []
    for i in range(len(bnd) - 1):
        seg = np.ascontiguousarray(np.round(arr[bnd[i]:bnd[i + 1]].numpy().astype("float64"), precision))
        out.append(hashlib.sha1(seg.tobytes()).hexdigest())
    return out, key


def discover(root):
    """task -> {dir_name -> {split -> Path}} for every down_task*/processed dir."""
    found = defaultdict(lambda: defaultdict(dict))
    for proc in sorted(root.glob("down_task*/processed")):
        dname = proc.parent.name
        for pt in proc.glob("*_*.pt"):
            m = re.match(r"^(?P<task>[a-z0-9]+)_(?P<split>train|valid|test|all)$", pt.stem)
            if not m:
                # also catch seed-tagged like bace_tri_9_test -> skip (not canonical)
                continue
            task, split = m.group("task"), m.group("split")
            if task not in KNOWN_TASKS or split == "all":
                continue
            found[task][dname][split] = pt
    return found


def _overlap(a, b):
    return len(set(a) & set(b))


def analyse_task(task, dirs, identity="x", precision=4):
    """dirs: {dir_name -> {split -> Path}}. Returns a JSON-able report dict."""
    test_hashes, identity_keys, dups = {}, {}, {}
    train_hashes = {}
    for dname, splits in dirs.items():
        if "test" in splits:
            h, k = _seg_hashes(splits["test"], identity, precision)
            test_hashes[dname] = h
            identity_keys[dname] = k
            dups[dname] = len(h) - len(set(h))
        if "train" in splits:
            th, _ = _seg_hashes(splits["train"], identity, precision)
            train_hashes[dname] = th

    names = sorted(test_hashes)
    # pairwise test overlap matrix
    matrix = {a: {b: _overlap(test_hashes[a], test_hashes[b]) for b in names} for a in names}

    # split-equivalence groups: same test set up to duplicates -> same set(hash)
    groups = []
    assigned = set()
    for a in names:
        if a in assigned:
            continue
        g = [a]
        assigned.add(a)
        for b in names:
            if b in assigned:
                continue
            if set(test_hashes[a]) == set(test_hashes[b]):
                g.append(b)
                assigned.add(b)
        groups.append(sorted(g))

    # cross-dir leakage: test(A) molecules appearing in train(B). Only meaningful
    # ACROSS split-equivalence groups -- within a group, train/test are disjoint
    # by construction, so any residual is an fps hash collision (a distinct
    # molecule sharing a fingerprint), reported separately as noise.
    group_of = {d: gi for gi, g in enumerate(groups) for d in g}
    leakage, fps_collisions = {}, {}
    for a in names:
        for b in train_hashes:
            if a == b:
                continue
            n = _overlap(test_hashes[a], train_hashes[b])
            if not n:
                continue
            if group_of.get(a) == group_of.get(b):
                fps_collisions[f"test[{a}] vs train[{b}]"] = n
            else:
                leakage[f"test[{a}] in train[{b}]"] = n

    verdict = "ALIGNED" if len(groups) == 1 else "MISALIGNED"
    return {
        "task": task,
        "dirs": names,
        "n_test": {a: len(test_hashes[a]) for a in names},
        "n_test_unique": {a: len(set(test_hashes[a])) for a in names},
        "intra_set_duplicates": dups,
        "identity_key": identity_keys,
        "overlap_matrix": matrix,
        "split_equivalence_groups": groups,
        "cross_dir_leakage": leakage,
        "intra_group_fps_collisions": fps_collisions,
        "verdict": verdict,
    }


def print_task_report(rep):
    names = rep["dirs"]
    print(f"\n{'='*72}\nTASK: {rep['task']}   ->   {rep['verdict']}")
    if len(names) < 2:
        print(f"  only {len(names)} dir with a test split locally: {names} "
              f"(cannot compare; run where both V0 + V2 dirs exist)")
    keyset = set(rep["identity_key"].values())
    print(f"  identity: {', '.join(sorted(keyset))}")
    dups = {k: v for k, v in rep["intra_set_duplicates"].items() if v}
    if dups:
        print(f"  NOTE intra-set duplicate molecules (fps-identical): {dups}")
    # matrix
    if len(names) >= 2:
        w = max(12, max(len(n) for n in names) + 1)
        print("  test-set overlap matrix (|A ∩ B|):")
        print("    " + "".join(f"{n[-w+1:]:>{w}}" for n in names))
        for a in names:
            row = "".join(f"{rep['overlap_matrix'][a][b]:>{w}}" for b in names)
            print(f"    {a[-w+1:]:>{w-1}} {row}")
    print("  split-equivalence groups (same test molecules):")
    for i, g in enumerate(rep["split_equivalence_groups"]):
        n = rep["n_test_unique"][g[0]]
        print(f"    group {i+1} (n_unique≈{n}): {g}")
    if rep["cross_dir_leakage"]:
        print("  CROSS-GROUP LEAKAGE (test molecules found in a DIFFERENT-split dir's train):")
        for k, v in rep["cross_dir_leakage"].items():
            print(f"    {k}: {v}")
    if rep.get("intra_group_fps_collisions"):
        tot = sum(rep["intra_group_fps_collisions"].values())
        print(f"  (ignore: {tot} same-group fps hash-collision hits across "
              f"{len(rep['intra_group_fps_collisions'])} pairs -- not real leakage)")
    if rep["verdict"] == "MISALIGNED":
        print("  >>> MISALIGNED: these dirs do NOT share one test set. Any V0-vs-V2")
        print("      comparison spanning two different groups is on DIFFERENT test")
        print("      sets and is NOT a valid like-for-like comparison.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO),
                    help="repo root containing down_task*/ (default: script dir)")
    ap.add_argument("--task", default=None, help="restrict to one task (e.g. bace)")
    ap.add_argument("--out", default=None,
                    help="write JSON report (default: paper/audits/split_alignment_<date>.json)")
    ap.add_argument("--identity", choices=["x", "fps"], default="x",
                    help="per-molecule identity key (default x = build-stable node features)")
    ap.add_argument("--precision", type=int, default=4, help="rounding decimals for the hash")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    found = discover(root)
    if a.task:
        found = {a.task: found.get(a.task, {})}
    if not any(found.values()):
        print(f"No processed down_task*/ dirs with <task>_test.pt under {root}")
        return

    reports = {}
    for task in sorted(found):
        if not found[task]:
            continue
        rep = analyse_task(task, found[task], a.identity, a.precision)
        reports[task] = rep
        print_task_report(rep)

    misaligned = [t for t, r in reports.items() if r["verdict"] == "MISALIGNED"]
    print(f"\n{'='*72}\nSUMMARY: {len(reports)} task(s) checked; "
          f"MISALIGNED: {misaligned or 'none'}")

    out = a.out
    if out is None:
        stamp = datetime.date.today().isoformat()
        out = root / "paper" / "audits" / f"split_alignment_{stamp}.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
