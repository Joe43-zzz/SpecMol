"""Compute n=9 BBBP V0/V2-T5/T6 statistics from the extended HPC logs.

Currently the paper Table 1 BBBP column uses n=3 training seeds (9, 19, 29)
because that's all that was run pre-2026-05-18. The HPC extension submitted
in the next-phase roadmap added seeds 39, 49, 59, 69, 79, 89 for V2-T5 (and
in a follow-up commit, for V0 and T6 too).

This script:
  - Reads per-seed mean AUC for the 3 original + 6 extension seeds
    (per-seed = mean over 3 eval_seeds within that training seed)
  - Reports n=3 (original), n=6 (extension), n=9 (combined) for each variant
  - Computes matched-seed deltas at n=3 vs n=9 to show whether the original
    "+2.6 AUC on BBBP" claim holds at scale

Run after HPC extension completes and after the per-restart numbers are
pasted in below from `grep "Best Test Auc" hpc/logs/bbbp/v0_seed*_*.log`
output (or pulled from collect_results.py output JSON).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# --- V2-T5 extension data (committed; pulled from HPC 2026-05-18) ----
# Per-training-seed mean across 3 eval restarts.
V2T5_EXT_PER_SEED = {
    39: 0.7979,
    49: 0.8170,
    59: 0.8154,
    69: 0.7810,
    79: 0.8383,
    89: 0.8407,
}

# --- V0 / T6 extension placeholders ---------------------------------
# Filled by reading the new logs once they exist on HPC. Until then we
# fall back to the existing bbbp_all_results.json (n=3 only).
V0_EXT_PER_SEED: dict[int, float] = {}
T6_EXT_PER_SEED: dict[int, float] = {}


def load_original_n3():
    path = REPO / "bbbp_all_results.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for label, src_key in [("V0", "v0_baseline"), ("V2-T5", "v2_t5_static_pair"),
                            ("T6", "t6_dynamic_pair_node")]:
        seeds = d[src_key]["results"]
        per_seed_means = []
        for sk, sv in seeds.items():
            if "mean" in sv:
                per_seed_means.append(float(sv["mean"]))
            elif "best_test_auc" in sv:
                per_seed_means.append(float(sv["best_test_auc"]))
        out[label] = per_seed_means
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO / "paper" / "bbbp_n9_summary.md"))
    args = p.parse_args()

    orig = load_original_n3()
    print("Original n=3 per-seed means (from bbbp_all_results.json):")
    for k, v in orig.items():
        print(f"  {k}: {v}  mean={np.mean(v):.4f} std={np.std(v, ddof=1):.4f}")

    v2t5_all = orig["V2-T5"] + list(V2T5_EXT_PER_SEED.values())
    v0_all = orig["V0"] + list(V0_EXT_PER_SEED.values())
    t6_all = orig["T6"] + list(T6_EXT_PER_SEED.values())

    print("\nV2-T5 BBBP at scale:")
    print(f"  n=3 original: mean={np.mean(orig['V2-T5']):.4f} std={np.std(orig['V2-T5'], ddof=1):.4f}")
    if V2T5_EXT_PER_SEED:
        ext = list(V2T5_EXT_PER_SEED.values())
        print(f"  n=6 extension: mean={np.mean(ext):.4f} std={np.std(ext, ddof=1):.4f}")
    print(f"  n={len(v2t5_all)} combined: mean={np.mean(v2t5_all):.4f} std={np.std(v2t5_all, ddof=1):.4f}")

    lines = ["# BBBP n=9 extension summary", ""]
    lines.append("| Variant | n=3 mean | n=3 std | n=6 ext mean | n=6 ext std | n combined | combined mean | combined std |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, ext_dict, all_seeds in [
        ("V2-T5", V2T5_EXT_PER_SEED, v2t5_all),
        ("V0", V0_EXT_PER_SEED, v0_all),
        ("T6", T6_EXT_PER_SEED, t6_all),
    ]:
        n3_mean = np.mean(orig[label])
        n3_std = np.std(orig[label], ddof=1)
        n6_mean = np.mean(list(ext_dict.values())) if ext_dict else None
        n6_std = np.std(list(ext_dict.values()), ddof=1) if ext_dict and len(ext_dict) > 1 else None
        nc_mean = np.mean(all_seeds)
        nc_std = np.std(all_seeds, ddof=1)
        lines.append(f"| {label} | {n3_mean:.4f} | {n3_std:.4f} | "
                     f"{n6_mean:.4f if n6_mean is not None else 'pending'} | "
                     f"{n6_std:.4f if n6_std is not None else 'pending'} | "
                     f"{len(all_seeds)} | {nc_mean:.4f} | {nc_std:.4f} |")

    lines.append("")
    if V2T5_EXT_PER_SEED and not V0_EXT_PER_SEED:
        lines.append(("> Note: V0 and T6 6-seed extensions still pending. The V2-T5 delta vs V0 "
                      f"shown above compares V2-T5 (n={len(v2t5_all)}) against V0 (n=3); a fair "
                      "matched-seed comparison requires V0/T6 extensions to land first."))

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
