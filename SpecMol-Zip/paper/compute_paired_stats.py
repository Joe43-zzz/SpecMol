"""Per-seed deltas and sign-test p-values for the variant comparisons reported
in the paper.

Usage:
    python paper/compute_paired_stats.py

The script reads the same result JSONs that paper/make_tables.py consumes, then
emits a compact table of matched-seed (variant_A - variant_B) differences plus
the two-sided sign-test p-value at n=3. Output is both stdout-friendly and
written to paper/paired_stats.json for later cross-referencing in the prose.

Rationale: with only three pretraining seeds, the headline +X.X-AUC mean can be
the average of two large wins and a loss on the same seed, which a single mean
hides. We surface every such per-seed delta explicitly. The sign test is the
appropriate non-parametric test at n=3 (Wilcoxon ties up too quickly), and we
report it strictly for transparency rather than as a basis for any "significant
at p<0.05" claim.
"""

from __future__ import annotations

import json
import statistics
from math import comb
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "paired_stats.json"


def _load(name: str) -> Optional[dict]:
    path = REPO / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _bbbp_per_seed_mean(block: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for seed_key, entry in block.get("results", {}).items():
        if "mean" in entry:
            out[seed_key] = float(entry["mean"])
        elif "best_test_auc" in entry:
            out[seed_key] = float(entry["best_test_auc"])
    return out


def _bace_per_seed_mean(block: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for seed_key, entry in block.get("results_per_seed", {}).items():
        if "mean_test_auc" in entry:
            out[seed_key] = float(entry["mean_test_auc"])
        elif "mean_test_metric" in entry:
            # hpc/collect_results.py output uses generic "metric" naming
            out[seed_key] = float(entry["mean_test_metric"])
        elif "eval_splits" in entry:
            vals = [
                float(r.get("test_auc", r.get("test_metric", 0.0)))
                for r in entry["eval_splits"].values()
                if "test_auc" in r or "test_metric" in r
            ]
            if vals:
                out[seed_key] = statistics.mean(vals)
    return out


def _collect_t7_per_seed_mean(data: dict) -> dict[str, float]:
    """Extract per-seed means from hpc/collect_results.py output for the t7 variant.

    Used when bare-T7 results live in a separate JSON (bbbp_t7_bare_results.json
    or bace_t7_bare_results.json) rather than being merged into the canonical
    results file. Returns {} if t7 not present.
    """
    if not data:
        return {}
    variants = data.get("variants", {})
    t7 = variants.get("t7")
    if not t7:
        return {}
    # collect_results.py output uses "results_per_seed" + "mean_test_metric"
    return _bace_per_seed_mean(t7)


def _freesolv_per_seed(block: dict) -> dict[str, float]:
    return {
        sk: float(v["best_test_rmse"])
        for sk, v in block.get("results_per_seed", {}).items()
        if "best_test_rmse" in v
    }


def sign_test_two_sided(deltas: list[float]) -> float:
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    cdf = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cdf)


def _align(a: dict[str, float], b: dict[str, float]) -> list[tuple[str, float, float]]:
    # Tolerate "seed9" vs "seed_9" naming inconsistency.
    def normalize(k: str) -> str:
        return k.replace("_", "")

    a_norm = {normalize(k): (k, v) for k, v in a.items()}
    b_norm = {normalize(k): (k, v) for k, v in b.items()}
    aligned = []
    for nk in sorted(set(a_norm) & set(b_norm)):
        ak, av = a_norm[nk]
        _, bv = b_norm[nk]
        aligned.append((ak, av, bv))
    return aligned


def _compare(
    label: str,
    a_label: str,
    a: dict[str, float],
    b_label: str,
    b: dict[str, float],
    higher_is_better: bool = True,
) -> dict:
    pairs = _align(a, b)
    if not pairs:
        return {"label": label, "note": "no matched seeds", "n": 0}
    # delta = "a improves over b". For AUC this is a - b; for RMSE it is b - a.
    deltas = [(av - bv) if higher_is_better else (bv - av) for _, av, bv in pairs]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    p = sign_test_two_sided(deltas)
    return {
        "label": label,
        "a": a_label,
        "b": b_label,
        "metric_direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "per_seed": [
            {"seed": sk, "a": av, "b": bv, "delta": d}
            for (sk, av, bv), d in zip(pairs, deltas)
        ],
        "mean_delta": statistics.mean(deltas),
        "wins_for_a": wins,
        "losses_for_a": losses,
        "ties": len(deltas) - wins - losses,
        "n": len(deltas),
        "sign_test_two_sided_p": p,
    }


def _fmt_row(c: dict) -> str:
    if c.get("n", 0) == 0:
        return f"  [skip] {c['label']}: {c['note']}"
    per_seed = ", ".join(
        f"{p['seed']}: {p['delta']:+.4f}" for p in c["per_seed"]
    )
    return (
        f"  {c['label']:32s}  mean={c['mean_delta']:+.4f}  "
        f"W/L/T={c['wins_for_a']}/{c['losses_for_a']}/{c['ties']}  "
        f"sign-p={c['sign_test_two_sided_p']:.3f}  ({per_seed})"
    )


def main() -> None:
    bbbp = _load("bbbp_all_results.json") or {}
    bace = _load("baseline_unifold_results.json") or {}
    fs_base = _load("freesolv_baseline_results.json") or {}
    fs_v2 = _load("freesolv_v2t5_results.json") or {}

    v0_bbbp = _bbbp_per_seed_mean(bbbp.get("v0_baseline", {}))
    v2_bbbp = _bbbp_per_seed_mean(bbbp.get("v2_t5_static_pair", {}))
    t6_bbbp = _bbbp_per_seed_mean(bbbp.get("t6_dynamic_pair_node", {}))

    # Bare-T7 from S1: prefer separate file (collector format); fall back to
    # legacy "t7" key in bbbp_all_results.json if user merged manually.
    t7_bbbp_raw = _load("bbbp_t7_bare_results.json")
    t7_bbbp = _collect_t7_per_seed_mean(t7_bbbp_raw) if t7_bbbp_raw else {}
    if not t7_bbbp:
        t7_bbbp = _bbbp_per_seed_mean(bbbp.get("t7", {}))

    v0_bace = _bace_per_seed_mean(bace.get("baseline", {}))
    fp_bace = _bace_per_seed_mean(bace.get("fp_only", {}))

    v0_fs = _freesolv_per_seed(fs_base)
    v2_fs = _freesolv_per_seed(fs_v2)

    comparisons = [
        _compare("BBBP V2-T5 vs V0", "V2-T5", v2_bbbp, "V0", v0_bbbp, higher_is_better=True),
        _compare("BBBP T6 vs V0",    "T6",    t6_bbbp, "V0", v0_bbbp, higher_is_better=True),
        _compare("BBBP V2-T5 vs T6", "V2-T5", v2_bbbp, "T6", t6_bbbp, higher_is_better=True),
        _compare("BBBP T7 vs V2-T5", "T7", t7_bbbp, "V2-T5", v2_bbbp, higher_is_better=True),
        _compare("BBBP T7 vs T6",    "T7", t7_bbbp, "T6",    t6_bbbp, higher_is_better=True),
        _compare("BBBP T7 vs V0",    "T7", t7_bbbp, "V0",    v0_bbbp, higher_is_better=True),
        _compare("BACE FP-only vs V0", "FP-only", fp_bace, "V0", v0_bace, higher_is_better=True),
        _compare("FreeSolv V2-T5 vs V0", "V2-T5", v2_fs, "V0", v0_fs, higher_is_better=False),
    ]

    print("Paired-seed comparisons (mean delta of A over B; sign-test two-sided p):")
    for c in comparisons:
        print(_fmt_row(c))

    OUT.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(f"\n[ok] wrote {OUT.name}")


if __name__ == "__main__":
    main()
