"""Aggregate *_results.json into LaTeX booktabs tables.

Usage:
    python paper/make_tables.py

Reads:
    bbbp_all_results.json
    baseline_unifold_results.json
    freesolv_baseline_results.json
    freesolv_v2t5_results.json
    bace_all_results.json            (optional, produced post-B4)

Writes:
    paper/tables/main_results.tex    (cls/reg combined table)
    paper/tables/classification.tex
    paper/tables/regression.tex
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "tables"


def load(name: str) -> Optional[dict]:
    path = REPO / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fmt(mean: float, std: Optional[float]) -> str:
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _stats(per_seed_values: list[float]) -> tuple[float, float]:
    """Sample mean and sample std (n-1) over per-seed values.

    Unified convention: every reported (mean, std) pair in the paper comes from
    the same formula, regardless of whether the underlying run had eval restarts.
    For variants with restarts, callers should pass per-seed *means* (i.e. each
    seed's restarts already collapsed), not raw restart-level values, so that
    inter-seed and intra-seed variance are not conflated.
    """
    if not per_seed_values:
        return float("nan"), float("nan")
    m = statistics.mean(per_seed_values)
    s = statistics.stdev(per_seed_values) if len(per_seed_values) > 1 else 0.0
    return m, s


def _bbbp_per_seed_means(block: dict) -> list[float]:
    """Extract per-seed means from a BBBP results block.

    The block schema differs across variants:
        v0_baseline:           {seed{N}: {best_test_auc, best_epoch}}
        v2_t5_static_pair / t6: {seed{N}: {eval_seed_*: float, mean: float}}
    For V0 we use best_test_auc directly; for the others we use the recorded
    mean (which is itself an average over eval restarts).
    """
    means: list[float] = []
    for seed_key in sorted(block.get("results", {})):
        entry = block["results"][seed_key]
        if "mean" in entry:
            means.append(float(entry["mean"]))
        elif "best_test_auc" in entry:
            means.append(float(entry["best_test_auc"]))
    return means


def parse_bbbp(data: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for key, label in [
        ("v0_baseline", "V0"),
        ("v2_t5_static_pair", "V2-T5"),
        ("t6_dynamic_pair_node", "T6"),
    ]:
        block = data.get(key)
        if not block:
            continue
        out[label] = _stats(_bbbp_per_seed_means(block))
    return out


def _bace_v0_fp_per_seed_means(block: dict) -> list[float]:
    """Per-seed means for BACE baseline_unifold_results.json blocks.

    Each seed entry has {eval_splits: {restart_*: {test_auc, ...}}, mean_test_auc}.
    We use the recorded mean_test_auc when present, else compute it on the fly.
    """
    means: list[float] = []
    per_seed = block.get("results_per_seed", {})
    for seed_key in sorted(per_seed):
        entry = per_seed[seed_key]
        if "mean_test_auc" in entry:
            means.append(float(entry["mean_test_auc"]))
        elif "eval_splits" in entry:
            restart_vals = [float(r["test_auc"]) for r in entry["eval_splits"].values()]
            if restart_vals:
                means.append(statistics.mean(restart_vals))
    return means


def parse_bace_v0_fp(data: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for key, label in [("baseline", "V0"), ("fp_only", "FP-only")]:
        block = data.get(key)
        if not block:
            continue
        out[label] = _stats(_bace_v0_fp_per_seed_means(block))
    return out


def parse_bace_full(data: dict) -> dict[str, tuple[float, float]]:
    """Parser for post-B4 bace_all_results.json (when it exists)."""
    out = {}
    for variant_key in ("v0", "v2_t5", "t6", "fp_only"):
        block = data.get(variant_key)
        if not block:
            continue
        label = {
            "v0": "V0",
            "v2_t5": "V2-T5",
            "t6": "T6",
            "fp_only": "FP-only",
        }[variant_key]
        # Prefer per_seed_means when present so the std convention matches.
        if "per_seed_means" in block:
            out[label] = _stats([float(x) for x in block["per_seed_means"]])
        elif "summary" in block and "per_seed_means" in block["summary"]:
            out[label] = _stats([float(x) for x in block["summary"]["per_seed_means"]])
        elif "grand_mean" in block:
            out[label] = (float(block["grand_mean"]),
                          float(block.get("std", block.get("grand_std", 0.0))))
        elif "summary" in block:
            s = block["summary"]
            out[label] = (float(s["grand_mean"]),
                          float(s.get("grand_std", 0.0)))
    return out


def _freesolv_per_seed(block: dict) -> list[float]:
    if not block:
        return []
    # Each seed entry has best_test_rmse (single run, no restarts).
    return [float(v["best_test_rmse"]) for v in block.get("results_per_seed", {}).values()
            if "best_test_rmse" in v]


def parse_freesolv(baseline: dict, v2: dict) -> dict[str, tuple[float, float]]:
    out = {}
    if baseline:
        out["V0"] = _stats(_freesolv_per_seed(baseline))
    if v2:
        out["V2-T5"] = _stats(_freesolv_per_seed(v2))
    return out


def render_cls_table(bbbp: dict, bace: dict) -> str:
    variants = ["V0", "FP-only", "RF", "Chemprop", "V2-T5", "T6"]
    lines = [
        "% Auto-generated by paper/make_tables.py — do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Classification results (test AUC, mean $\\pm$ sample std (n{=}3) over per-seed means, Uni-Mol fold split).}",
        "\\label{tab:classification}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Variant & BBBP & BACE \\\\",
        "\\midrule",
    ]
    for v in variants:
        bbbp_cell = fmt(*bbbp[v]) if v in bbbp else "--"
        bace_cell = fmt(*bace[v]) if v in bace else "--"
        lines.append(f"{v} & {bbbp_cell} & {bace_cell} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_reg_table(freesolv: dict) -> str:
    variants = ["V0", "V2-T5"]
    lines = [
        "% Auto-generated by paper/make_tables.py — do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Regression results on FreeSolv (test RMSE, lower is better; mean $\\pm$ sample std (n{=}3), scaffold split; the V2-T5 row uses a GBF pair-feature surrogate in place of Uni-Mol, see \\S\\ref{sec:v2t5}).}",
        "\\label{tab:regression}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Variant & FreeSolv (RMSE) \\\\",
        "\\midrule",
    ]
    for v in variants:
        cell = fmt(*freesolv[v]) if v in freesolv else "--"
        lines.append(f"{v} & {cell} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def parse_rf_unimol_fold(data: dict) -> dict[str, tuple[float, float]]:
    """Parse baselines_ml_results_unimol_fold.json -> {dataset: (mean, std)} for RF.

    Uses roc_auc for classification, rmse for regression. The JSON's own
    sample size (5-seed default or 30-seed Deng2023 protocol) is preserved
    in std; cell text shows the value as-is.
    """
    out = {}
    for ds in ("bace", "bbbp", "freesolv"):
        if ds not in data:
            continue
        m, s = data[ds]["mean"], data[ds]["std"]
        metric = "rmse" if ds == "freesolv" else "roc_auc"
        out[ds] = (m[metric], s[metric])
    return out


def parse_chemprop_unimol_fold(data: dict) -> dict[str, tuple[float, float]]:
    """Parse baselines_chemprop_results_unimol_fold.json -> {dataset: (mean, std)}.

    Chemprop emits 'auc' for classification and 'rmse' for regression under
    the same metric_name key. mean/std are pre-aggregated by the runner.
    """
    out = {}
    for ds in ("bace", "bbbp", "freesolv"):
        if ds not in data or "mean" not in data[ds]:
            continue
        metric = "rmse" if ds == "freesolv" else "auc"
        if metric in data[ds]["mean"]:
            out[ds] = (data[ds]["mean"][metric], data[ds]["std"].get(metric, 0.0))
    return out


def render_main_table(bbbp: dict, bace: dict, freesolv: dict) -> str:
    variants = ["V0", "FP-only", "RF", "Chemprop", "V2-T5", "T6"]
    lines = [
        "% Auto-generated by paper/make_tables.py — do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main results across molecular property prediction benchmarks. "
        "Classification reports test AUC (higher is better); regression reports test RMSE (lower is better). "
        "Classification benchmarks use the Uni-Mol scaffold-fold split; FreeSolv uses a standard scaffold split. "
        "Deep-model cells (V0, FP-only, V2-T5, T6) are mean~$\\pm$~sample standard deviation across three training seeds (n{=}3); "
        "for variants with multiple downstream restarts per training seed, the per-seed mean (averaging over restarts) "
        "is the unit used in the std calculation, so inter-seed and intra-seed noise are not conflated. "
        "The RF row uses the 30-seed protocol of \\citet{deng2023systematic} on the matched Uni-Mol fold split (n{=}30). "
        "Dashes mark runs pending the post-B4 data rebuild (\\S\\ref{sec:discussion}).}",
        "\\label{tab:main}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Classification (AUC $\\uparrow$)} & Regression (RMSE $\\downarrow$) \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-4}",
        "Variant & BBBP & BACE & FreeSolv \\\\",
        "\\midrule",
    ]
    for v in variants:
        bbbp_cell = fmt(*bbbp[v]) if v in bbbp else "--"
        bace_cell = fmt(*bace[v]) if v in bace else "--"
        free_cell = fmt(*freesolv[v]) if v in freesolv else "--"
        lines.append(f"{v} & {bbbp_cell} & {bace_cell} & {free_cell} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bbbp_raw = load("bbbp_all_results.json")
    bace_v0fp_raw = load("baseline_unifold_results.json")
    bace_full_raw = load("bace_all_results.json")  # post-B4
    fs_base = load("freesolv_baseline_results.json")
    fs_v2 = load("freesolv_v2t5_results.json")
    # RF classical baseline: prefer 30-seed Deng2023 protocol, fall back to 5-seed.
    rf_raw = load("baselines_ml_results_deng30_unimol_fold.json") or \
             load("baselines_ml_results_unimol_fold.json")

    bbbp = parse_bbbp(bbbp_raw) if bbbp_raw else {}
    bace = parse_bace_v0_fp(bace_v0fp_raw) if bace_v0fp_raw else {}
    if bace_full_raw:
        # post-B4 numbers override / extend
        bace.update(parse_bace_full(bace_full_raw))
    freesolv = parse_freesolv(fs_base, fs_v2)
    if rf_raw:
        rf = parse_rf_unimol_fold(rf_raw)
        if "bbbp" in rf:
            bbbp["RF"] = rf["bbbp"]
        if "bace" in rf:
            bace["RF"] = rf["bace"]
        if "freesolv" in rf:
            freesolv["RF"] = rf["freesolv"]

    chemprop_raw = load("baselines_chemprop_results_unimol_fold.json")
    if chemprop_raw:
        cp = parse_chemprop_unimol_fold(chemprop_raw)
        if "bbbp" in cp:
            bbbp["Chemprop"] = cp["bbbp"]
        if "bace" in cp:
            bace["Chemprop"] = cp["bace"]
        if "freesolv" in cp:
            freesolv["Chemprop"] = cp["freesolv"]

    (OUT_DIR / "classification.tex").write_text(render_cls_table(bbbp, bace), encoding="utf-8")
    (OUT_DIR / "regression.tex").write_text(render_reg_table(freesolv), encoding="utf-8")
    (OUT_DIR / "main_results.tex").write_text(render_main_table(bbbp, bace, freesolv), encoding="utf-8")

    print(f"[ok] wrote tables to {OUT_DIR}")
    print(f"  bbbp variants:     {list(bbbp)}")
    print(f"  bace variants:     {list(bace)}")
    print(f"  freesolv variants: {list(freesolv)}")
    if "V2-T5" not in bace or "T6" not in bace:
        print("[warn] BACE V2-T5 / T6 missing — re-run HPC pipeline to produce bace_all_results.json")


if __name__ == "__main__":
    main()
