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

# --- V0 / T6 extension data (committed; pulled from HPC 2026-05-19) -
# Per-training-seed mean across 3 eval restarts. Logs:
#   v0_seed{39..89}_135{834..839}.log,
#   t6_seed{39..89}_135{840..845}.log
V0_EXT_PER_SEED = {
    39: (0.8332 + 0.8424 + 0.8379) / 3,  # 0.8378
    49: (0.8459 + 0.8460 + 0.8444) / 3,  # 0.8454
    59: (0.7871 + 0.7826 + 0.7815) / 3,  # 0.7837
    69: (0.8069 + 0.8031 + 0.8024) / 3,  # 0.8041
    79: (0.8489 + 0.8453 + 0.8489) / 3,  # 0.8477
    89: (0.8214 + 0.8229 + 0.8314) / 3,  # 0.8252
}
T6_EXT_PER_SEED = {
    39: (0.8261 + 0.8312 + 0.8255) / 3,
    49: (0.8345 + 0.8343 + 0.8354) / 3,
    59: (0.8087 + 0.8086 + 0.8101) / 3,
    69: (0.8422 + 0.8422 + 0.8425) / 3,
    79: (0.8313 + 0.8322 + 0.8320) / 3,
    89: (0.8098 + 0.8131 + 0.8076) / 3,
}

# --- V2-T5 random-pair ablation (3 matched seeds, 3 eval restarts) ---
# Logs: v2_randp_seed{9,19,29}_135{849,850,851}.log
RANDP_PER_SEED = {
    9:  (0.8491 + 0.8464 + 0.8481) / 3,  # 0.8479
    19: (0.8149 + 0.8135 + 0.8126) / 3,  # 0.8137
    29: (0.8268 + 0.8321 + 0.8256) / 3,  # 0.8282
}

# --- T7 extension data (FUTURE) ---
# Populated only if outcome A triggers T7 n=9 scale-up. Until then, leaving
# this empty causes the script to silently skip T7 from the n=9 table.
T7_EXT_PER_SEED: dict[int, float] = {}


def load_original_n3():
    path = REPO / "bbbp_all_results.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for label, src_key in [("V0", "v0_baseline"), ("V2-T5", "v2_t5_static_pair"),
                            ("T6", "t6_dynamic_pair_node"), ("T7", "t7")]:
        block = d.get(src_key)
        if not block:
            # T7 expected absent until user merges bare-T7 results manually.
            continue
        seeds = block.get("results", {})
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
    t7_all = orig.get("T7", []) + list(T7_EXT_PER_SEED.values())

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
        ("T7", T7_EXT_PER_SEED, t7_all),
    ]:
        if label not in orig:
            # T7 row skipped if no n=3 data merged in yet.
            continue
        n3_mean = np.mean(orig[label])
        n3_std = np.std(orig[label], ddof=1) if len(orig[label]) > 1 else 0.0
        ext_vals = list(ext_dict.values())
        n6_str = f"{np.mean(ext_vals):.4f}" if ext_vals else "pending"
        n6_std_str = f"{np.std(ext_vals, ddof=1):.4f}" if len(ext_vals) > 1 else "pending"
        nc_mean = np.mean(all_seeds)
        nc_std = np.std(all_seeds, ddof=1) if len(all_seeds) > 1 else 0.0
        lines.append(f"| {label} | {n3_mean:.4f} | {n3_std:.4f} | "
                     f"{n6_str} | {n6_std_str} | "
                     f"{len(all_seeds)} | {nc_mean:.4f} | {nc_std:.4f} |")

    lines.append("")
    # Random-pair matched-seed comparison (seeds 9,19,29 only)
    if RANDP_PER_SEED:
        v2_matched = orig["V2-T5"]  # [seed9, seed19, seed29] means
        rp_matched = [RANDP_PER_SEED[9], RANDP_PER_SEED[19], RANDP_PER_SEED[29]]
        deltas = [v - r for v, r in zip(v2_matched, rp_matched)]
        wins = sum(1 for d in deltas if d > 0)
        lines.append("## Random-pair ablation (matched seeds 9/19/29)")
        lines.append("")
        lines.append(f"- V2-T5 (real Uni-Mol pair): {v2_matched} → mean {np.mean(v2_matched):.4f}")
        lines.append(f"- V2-T5 (random Gaussian pair): {[f'{x:.4f}' for x in rp_matched]} → mean {np.mean(rp_matched):.4f}")
        lines.append(f"- per-seed deltas (real − random): {[f'{d:+.4f}' for d in deltas]}")
        lines.append(f"- mean delta: {np.mean(deltas):+.4f}")
        lines.append(f"- sign test: {wins}/3 seeds favor real pair (binomial one-sided p≈{0.5 ** wins if wins == 3 else 'n/a'})")
        lines.append("")
    # Honest n=9 summary
    lines.append("## n=9 BBBP grand summary")
    lines.append("")
    lines.append("| Variant | n=9 mean | n=9 std |")
    lines.append("|---|---|---|")
    lines.append(f"| V0 | {np.mean(v0_all):.4f} | {np.std(v0_all, ddof=1):.4f} |")
    lines.append(f"| V2-T5 | {np.mean(v2t5_all):.4f} | {np.std(v2t5_all, ddof=1):.4f} |")
    lines.append(f"| T6 | {np.mean(t6_all):.4f} | {np.std(t6_all, ddof=1):.4f} |")
    if t7_all:
        t7_std = np.std(t7_all, ddof=1) if len(t7_all) > 1 else 0.0
        lines.append(f"| T7 | {np.mean(t7_all):.4f} | {t7_std:.4f} |")
    if RANDP_PER_SEED:
        rp_all = list(RANDP_PER_SEED.values())
        lines.append(f"| V2-T5 (random pair, n=3) | {np.mean(rp_all):.4f} | {np.std(rp_all, ddof=1):.4f} |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
