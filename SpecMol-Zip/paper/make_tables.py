"""Aggregate *_results.json into LaTeX booktabs tables.

Usage:
    python paper/make_tables.py

Reads (sprint extension 2026-05-28 — supports 7-dataset matrix):
    Classification:
      bbbp_all_results.json
      bace_all_results.json
      clintox_all_results.json             (sprint addition)
      tox21_all_results.json               (sprint addition)
      baseline_unifold_results.json
      bbbp_t7_bare_results.json            (optional)
      bace_t7_bare_results.json            (optional)
    Regression:
      freesolv_baseline_results.json
      freesolv_v2t5_results.json (or freesolv_unimol_v2t5_results.json preferred)
      esol_v2t5_results.json               (sprint addition)
      lipo_v2t5_results.json               (sprint addition)
      freesolv_t7_bare_results.json        (optional)
      esol_t7_bare_results.json            (optional)
      lipo_t7_bare_results.json            (optional)
    Baselines:
      baselines_chemprop_results_unimol_fold.json
      baselines_ml_results_deng30_unimol_fold.json (preferred, 30-seed)
      baselines_ml_results_unimol_fold.json        (fallback, 5-seed)

Writes:
    paper/tables/main_results.tex    (legacy 3-dataset combined table — kept for back-compat)
    paper/tables/classification.tex  (extended to BBBP/BACE/ClinTox/Tox21)
    paper/tables/regression.tex      (extended to FreeSolv/ESOL/Lipo)
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
        ("v0_baseline", "2D-Only"),
        ("v2_t5_static_pair", "SEG"),
        ("t6_dynamic_pair_node", "T6"),
        ("t7", "PBA"),  # bare-T7 if merged into bbbp_all_results.json (legacy schema)
    ]:
        block = data.get(key)
        if not block:
            continue
        out[label] = _stats(_bbbp_per_seed_means(block))
    return out


def parse_t7_collect_format(data: dict) -> Optional[tuple[float, float]]:
    """Parse hpc/collect_results.py output for the t7 variant.

    Used when bare-T7 results land in a separate file (bbbp_t7_bare_results.json
    or bace_t7_bare_results.json) rather than being merged into the canonical
    bbbp_all_results.json / bace_all_results.json.

    Expected schema (matches hpc/collect_results.py main() output):
        {"variants": {"t7": {"summary": {"per_seed_means": [...]}}}}
    """
    if not data:
        return None
    variants = data.get("variants", {})
    t7 = variants.get("t7")
    if not t7:
        return None
    summary = t7.get("summary", {})
    per_seed = summary.get("per_seed_means", [])
    if not per_seed:
        return None
    return _stats([float(x) for x in per_seed])


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
    for key, label in [("baseline", "2D-Only"), ("fp_only", "FP-only")]:
        block = data.get(key)
        if not block:
            continue
        out[label] = _stats(_bace_v0_fp_per_seed_means(block))
    return out


def parse_bace_full(data: dict) -> dict[str, tuple[float, float]]:
    """Parser for post-B4 bace_all_results.json produced by hpc/collect_results.py.

    Layout: {"variants": {"v0": {"summary": {...}, "results_per_seed": ...}, "v2t5": ..., "t6": ...}}
    Each variant's summary has per_seed_means + grand_mean + grand_std.
    Older layouts (variants at top level, no `variants` wrapper) also accepted
    for backward compat.
    """
    out = {}
    # Either nested under "variants" (new collector layout) or at top level.
    src = data.get("variants", data)
    # Accept both "v2t5" (collector) and "v2_t5" (older legacy).
    key_aliases = {
        "v0": "2D-Only",
        "v2t5": "SEG",
        "v2_t5": "SEG",
        "t6": "T6",
        "t7": "PBA",  # bare-T7 from S1 (collector output, same schema as v2t5)
        "t8": "APC",
        "t8_logits": "APC-logits",
        "fp_only": "FP-only",
    }
    for variant_key, label in key_aliases.items():
        block = src.get(variant_key)
        if not block:
            continue
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
        out["2D-Only"] = _stats(_freesolv_per_seed(baseline))
    if v2:
        out["SEG"] = _stats(_freesolv_per_seed(v2))
    return out


def parse_regression_matrix(task: str) -> dict[str, tuple[float, float]]:
    """Build {variant_label: (mean, std)} for one regression task from the
    per-variant aggregated JSONs produced by aggregate_regression_seeds.py.

    Reads <task>_v0_results.json / <task>_t6_results.json /
    <task>_t7_bare_results.json, plus the V2-T5 cell from
    freesolv_unimol_v2t5_results.json (FreeSolv) or <task>_v2t5_results.json
    (ESOL/Lipo). Each (mean, std) is recomputed from the per-seed
    best_test_rmse values via _stats so the std convention matches every other
    cell in the paper. Variants with no result file are simply omitted (the
    table renders them as "--").
    """
    v2_name = ("freesolv_unimol_v2t5_results.json" if task == "freesolv"
               else f"{task}_v2t5_results.json")
    sources = [
        ("2D-Only", f"{task}_v0_results.json"),
        ("SEG", v2_name),
        ("T6", f"{task}_t6_results.json"),
        ("PBA", f"{task}_t7_bare_results.json"),
        ("APC", f"{task}_t8_results.json"),
        ("APC-logits", f"{task}_t8_logits_results.json"),
    ]
    out: dict[str, tuple[float, float]] = {}
    for label, fname in sources:
        block = load(fname)
        seeds = _freesolv_per_seed(block) if block else []
        if seeds:
            out[label] = _stats(seeds)
    return out


def render_cls_table(bbbp: dict, bace: dict,
                     clintox: dict | None = None,
                     tox21: dict | None = None) -> str:
    """Sprint-extended classification table.

    Includes ClinTox/Tox21 columns when data is present; falls back to the
    original 2-column layout otherwise.
    """
    variants = ["2D-Only", "FP-only", "RF", "Chemprop", "Uni-Mol-emb$+$RF",
                "Uni-Mol (FT)", "SEG", "PBA", "APC", "APC-logits"]
    clintox = clintox or {}
    tox21 = tox21 or {}
    has_ctox = bool(clintox)
    has_tox21 = bool(tox21)

    columns = ["BBBP", "BACE"]
    column_data = [bbbp, bace]
    if has_ctox:
        columns.append("ClinTox")
        column_data.append(clintox)
    if has_tox21:
        columns.append("Tox21")
        column_data.append(tox21)

    col_spec = "l" + "c" * len(columns)
    header = " & ".join(["Variant"] + columns) + " \\\\"

    has_ft = any("Uni-Mol (FT)" in d for d in column_data)
    ft_clause = (" ``Uni-Mol (FT)'' is the end-to-end finetuned Uni-Mol on the "
                 "identical split (MolTrain exposes no seed control: cells with "
                 "std${\\approx}0.000$ are deterministic single-split internal-"
                 "5-fold-CV point estimates, \\emph{not} a zero-variance claim; "
                 "only BBBP and ClinTox showed seed variance). The two "
                 "``$\\pm$'' semantics differ: deep-variant cells "
                 "(2D-Only/SEG/PBA) report \\emph{seed} variance, whereas "
                 "Uni-Mol~(FT) cells report training-set-resampling "
                 "sensitivity (internal 5-fold CV), so the two columns' error "
                 "bars are not directly comparable." if has_ft else "")
    caption = (
        "\\caption{Classification test AUC, mean $\\pm$ sample std over per-seed "
        "means under matched Uni-Mol scaffold-fold split. BBBP 2D-Only/SEG use "
        "$n{=}9$ training seeds; BBBP PBA and BACE/ClinTox deep cells use "
        "$n{=}3$; RF is a Morgan$+$RDKit-2D random forest (the descriptor recipe "
        "of \\citet{deng2023systematic}, 30 restarts) recomputed on the model's "
        "\\emph{exact} test split. ClinTox cells report macro-averaged AUC across "
        "its two constituent tasks. The ``Uni-Mol-emb$+$RF'' row is a frozen "
        "Uni-Mol embedding with an RF head (a probe, \\emph{not} Uni-Mol's own "
        "performance)." + ft_clause + "}"
    )

    lines = [
        "% Auto-generated by paper/make_tables.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        caption,
        "\\label{tab:classification}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for v in variants:
        if all(v not in d for d in column_data):
            continue
        cells = []
        for d in column_data:
            cells.append(fmt(*d[v]) if v in d else "--")
        lines.append(f"{v} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_reg_table(freesolv: dict,
                     esol: dict | None = None,
                     lipo: dict | None = None) -> str:
    """Sprint-extended regression table.

    Includes ESOL/Lipo columns when data is present; FreeSolv-only otherwise.
    """
    variants = ["2D-Only", "RF", "Chemprop", "Uni-Mol-emb$+$RF", "Uni-Mol (FT)",
                "SEG", "PBA", "APC", "APC-logits"]
    esol = esol or {}
    lipo = lipo or {}
    has_esol = bool(esol)
    has_lipo = bool(lipo)

    columns = ["FreeSolv (RMSE)"]
    column_data = [freesolv]
    if has_esol:
        columns.append("ESOL (RMSE)")
        column_data.append(esol)
    if has_lipo:
        columns.append("Lipo (RMSE)")
        column_data.append(lipo)

    col_spec = "l" + "c" * len(columns)
    header = " & ".join(["Variant"] + columns) + " \\\\"

    has_ft = any("Uni-Mol (FT)" in d for d in column_data)
    ft_clause = (" ``Uni-Mol (FT)'' is the end-to-end Uni-Mol finetune; its "
                 "std${\\approx}0.000$ cells are deterministic single-split "
                 "internal-5-fold-CV point estimates (MolTrain has no seed "
                 "control), not a zero-variance claim. The two ``$\\pm$'' "
                 "semantics differ: deep-variant cells (2D-Only/SEG/PBA) "
                 "report \\emph{seed} variance, whereas Uni-Mol~(FT) cells "
                 "report training-set-resampling sensitivity (internal 5-fold "
                 "CV), so the two columns' error bars are not directly "
                 "comparable." if has_ft else "")
    caption = (
        "\\caption{Regression test RMSE (lower is better; mean $\\pm$ sample std "
        "over per-seed means under matched scaffold splits). FreeSolv 2D-Only/SEG use "
        "$n{=}9$ seeds; ESOL/Lipo deep cells and PBA use $n{=}3$; RF is a "
        "Morgan$+$RDKit-2D random forest (Deng2023 recipe, 30 restarts) on the "
        "model's \\emph{exact} test fold. "
        "all SEG cells (FreeSolv/ESOL/Lipo) use Uni-Mol pair representations; the "
        "FreeSolv PBA cell alone uses an RDKit-GBF pair surrogate (\\S\\ref{sec:v2t5}). RMSE is on the standardized "
        "label scale (std${\\approx}1$ for all three regression sets; FreeSolv is thus not in kcal/mol---its raw "
        "scale is $\\sigma{\\approx}3.8$ kcal/mol, so a standardized RMSE of $0.7$ is ${\\approx}2.7$ kcal/mol and "
        "is not comparable to the kcal/mol literature). The ``Uni-Mol-emb$+$RF'' row is the frozen-embedding "
        "probe." + ft_clause + "}"
    )

    lines = [
        "% Auto-generated by paper/make_tables.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        caption,
        "\\label{tab:regression}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for v in variants:
        if all(v not in d for d in column_data):
            continue
        cells = []
        for d in column_data:
            cells.append(fmt(*d[v]) if v in d else "--")
        lines.append(f"{v} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


CLASSIFICATION_DS = ("bbbp", "bace", "clintox", "tox21")
REGRESSION_DS = ("freesolv", "esol", "lipo")


def parse_rf_unimol_fold(data: dict) -> dict[str, tuple[float, float]]:
    """Parse baselines_ml_results_unimol_fold.json -> {dataset: (mean, std)} for RF.

    Uses roc_auc for classification, rmse for regression. The JSON's own
    sample size (5-seed default or 30-seed Deng2023 protocol) is preserved
    in std; cell text shows the value as-is.
    """
    out = {}
    for ds in CLASSIFICATION_DS + REGRESSION_DS:
        if ds not in data:
            continue
        m, s = data[ds]["mean"], data[ds]["std"]
        metric = "rmse" if ds in REGRESSION_DS else "roc_auc"
        if metric in m:
            out[ds] = (m[metric], s[metric])
    return out


def parse_chemprop_unimol_fold(data: dict) -> dict[str, tuple[float, float]]:
    """Parse baselines_chemprop_results_unimol_fold.json -> {dataset: (mean, std)}.

    Chemprop emits 'auc' for classification and 'rmse' for regression under
    the same metric_name key. mean/std are pre-aggregated by the runner.
    """
    out = {}
    for ds in CLASSIFICATION_DS + REGRESSION_DS:
        if ds not in data or "mean" not in data[ds]:
            continue
        metric = "rmse" if ds in REGRESSION_DS else "auc"
        if metric in data[ds]["mean"]:
            out[ds] = (data[ds]["mean"][metric], data[ds]["std"].get(metric, 0.0))
    return out


def parse_matched_baselines(data: dict) -> tuple[dict, dict]:
    """Parse baselines_matched_results.json -> (rf_matched, unimol) dicts.

    Schema: {dataset: {"morgan_rdkit": {mean:{...}, std:{...}}, "unimol": {...}}}.
    Both produced by baselines_matched.py on the model's *exact* split, so the
    'morgan_rdkit' RF here supersedes the deng30 RF for datasets whose deng30
    cell was computed on a regenerated (unmatched) split. The 'unimol' feature
    is the frozen-Uni-Mol-embedding + RF control. Uses rmse for regression,
    roc_auc (macro for multi-label) for classification.
    """
    rf_out: dict[str, tuple[float, float]] = {}
    unimol_out: dict[str, tuple[float, float]] = {}
    for ds, feats in data.items():
        metric = "rmse" if ds in REGRESSION_DS else "roc_auc"
        for feat, dst in (("morgan_rdkit", rf_out), ("unimol", unimol_out)):
            blk = feats.get(feat) if isinstance(feats, dict) else None
            if blk and "mean" in blk and metric in blk["mean"]:
                dst[ds] = (blk["mean"][metric], blk["std"].get(metric, 0.0))
    return rf_out, unimol_out


def parse_unimol_finetune(data: dict) -> dict[str, tuple[float, float]]:
    """Parse unimol_finetune_results.json -> {dataset: (mean, std)}.

    Schema (reproduce_unimol_finetune.py): {dataset: {..., "mean": {roc_auc|rmse: ..},
    "std": {..}}}. The end-to-end Uni-Mol finetune on the model's exact split; uses
    rmse for regression, roc_auc (macro over tasks for multi-label) for classification.
    """
    out: dict[str, tuple[float, float]] = {}
    for ds, blk in data.items():
        if not isinstance(blk, dict) or "mean" not in blk:
            continue
        metric = "rmse" if ds in REGRESSION_DS else "roc_auc"
        if metric in blk["mean"]:
            out[ds] = (blk["mean"][metric], blk.get("std", {}).get(metric, 0.0))
    return out


def render_main_table(bbbp: dict, bace: dict, freesolv: dict) -> str:
    variants = ["2D-Only", "FP-only", "RF", "Chemprop", "SEG", "T6", "PBA"]
    lines = [
        "% Auto-generated by paper/make_tables.py — do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main results. Classification reports AUC ($\\uparrow$); regression reports RMSE ($\\downarrow$). "
        "Deep cells are mean~$\\pm$~sample std over training seeds: BBBP and FreeSolv V0/V2-T5 use $n{=}9$, "
        "BBBP T7 and BACE/ClinTox/ESOL/Lipo deep cells use $n{=}3$, and RF uses the 30-seed protocol of \\citet{deng2023systematic}. "
        "the FreeSolv T7 cell alone uses an RDKit-GBF pair surrogate; all V2-T5 cells use Uni-Mol pair representations.}",
        "\\label{tab:main}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Classification (AUC $\\uparrow$)} & Regression (RMSE $\\downarrow$) \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-4}",
        "Variant & BBBP & BACE & FreeSolv \\\\",
        "\\midrule",
    ]
    for v in variants:
        if v not in bbbp and v not in bace and v not in freesolv:
            # Skip row entirely if no data for this variant in any column.
            continue
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
    clintox_full_raw = load("clintox_all_results.json")  # sprint addition
    tox21_full_raw = load("tox21_all_results.json")      # sprint addition
    # RF classical baseline: prefer 30-seed Deng2023 protocol, fall back to 5-seed.
    rf_raw = load("baselines_ml_results_deng30_unimol_fold.json") or \
             load("baselines_ml_results_unimol_fold.json")

    bbbp = parse_bbbp(bbbp_raw) if bbbp_raw else {}
    bace = parse_bace_v0_fp(bace_v0fp_raw) if bace_v0fp_raw else {}
    if bace_full_raw:
        # post-B4 numbers override / extend
        bace.update(parse_bace_full(bace_full_raw))
    clintox = parse_bace_full(clintox_full_raw) if clintox_full_raw else {}
    tox21 = parse_bace_full(tox21_full_raw) if tox21_full_raw else {}
    # Regression matrix (V0/V2-T5/T6/T7) from the per-variant aggregated JSONs
    # produced by paper/aggregate_regression_seeds.py.
    freesolv = parse_regression_matrix("freesolv")
    esol = parse_regression_matrix("esol")
    lipo = parse_regression_matrix("lipo")

    if rf_raw:
        rf = parse_rf_unimol_fold(rf_raw)
        for ds_key, table in (("bbbp", bbbp), ("bace", bace),
                              ("clintox", clintox), ("tox21", tox21),
                              ("freesolv", freesolv), ("esol", esol), ("lipo", lipo)):
            if ds_key in rf:
                table["RF"] = rf[ds_key]

    chemprop_raw = load("baselines_chemprop_results_unimol_fold.json")
    if chemprop_raw:
        cp = parse_chemprop_unimol_fold(chemprop_raw)
        for ds_key, table in (("bbbp", bbbp), ("bace", bace),
                              ("clintox", clintox), ("tox21", tox21),
                              ("freesolv", freesolv), ("esol", esol), ("lipo", lipo)):
            if ds_key in cp:
                table["Chemprop"] = cp[ds_key]

    # Matched-split baselines (baselines_matched.py): matched RF overrides the
    # deng30 RF cell, and the frozen-Uni-Mol-embedding control adds a Uni-Mol row.
    matched_raw = load("baselines_matched_results.json")
    if matched_raw:
        rf_m, unimol_m = parse_matched_baselines(matched_raw)
        for ds_key, table in (("bbbp", bbbp), ("bace", bace),
                              ("clintox", clintox),
                              ("freesolv", freesolv), ("esol", esol), ("lipo", lipo)):
            if ds_key in rf_m:
                table["RF"] = rf_m[ds_key]
            if ds_key in unimol_m:
                table["Uni-Mol-emb$+$RF"] = unimol_m[ds_key]

    # End-to-end Uni-Mol FINETUNE (reproduce_unimol_finetune.py). Distinct from
    # the frozen-embedding probe above: this is the same encoder's own supervised
    # number on the identical split, so the table never conflates the two.
    ft_raw = load("unimol_finetune_results.json")
    if ft_raw:
        ft = parse_unimol_finetune(ft_raw)
        for ds_key, table in (("bbbp", bbbp), ("bace", bace),
                              ("clintox", clintox),
                              ("freesolv", freesolv), ("esol", esol), ("lipo", lipo)):
            if ds_key in ft:
                table["Uni-Mol (FT)"] = ft[ds_key]

    # Bare-T7 (HPC collector output, separate file per task).
    for table, json_name in (
        (bbbp, "bbbp_t7_bare_results.json"),
        (bace, "bace_t7_bare_results.json"),
        (clintox, "clintox_t7_bare_results.json"),
        (tox21, "tox21_t7_bare_results.json"),
    ):
        if "PBA" in table:
            continue
        t7_raw = load(json_name)
        t7_stats = parse_t7_collect_format(t7_raw) if t7_raw else None
        if t7_stats is not None:
            table["PBA"] = t7_stats

    for table, json_name in (
        (freesolv, "freesolv_t7_bare_results.json"),
        (esol, "esol_t7_bare_results.json"),
        (lipo, "lipo_t7_bare_results.json"),
    ):
        if "PBA" in table:
            continue
        t7_raw = load(json_name)
        if t7_raw:
            t7_seeds = _freesolv_per_seed(t7_raw)
            if t7_seeds:
                table["PBA"] = _stats(t7_seeds)

    # Bare-T8 (HPC collector output, classification), same schema as bare-T7.
    for table, json_name in (
        (bbbp, "bbbp_t8_bare_results.json"),
        (bace, "bace_t8_bare_results.json"),
        (clintox, "clintox_t8_bare_results.json"),
    ):
        if "APC" in table:
            continue
        t8_raw = load(json_name)
        t8_stats = parse_t7_collect_format(t8_raw) if t8_raw else None
        if t8_stats is not None:
            table["APC"] = t8_stats

    # Tox21 is deferred from the 6-dataset matrix (Uni-Mol pair extraction
    # OOMs on its 7,831 molecules); we do not render a Tox21 column even
    # though the 30-seed RF file still carries a Tox21 cell.
    (OUT_DIR / "classification.tex").write_text(
        render_cls_table(bbbp, bace, clintox=clintox),
        encoding="utf-8",
    )
    (OUT_DIR / "regression.tex").write_text(
        render_reg_table(freesolv, esol=esol, lipo=lipo),
        encoding="utf-8",
    )
    # Legacy combined 3-dataset table (kept for back-compat with main.tex until rewrite).
    (OUT_DIR / "main_results.tex").write_text(
        render_main_table(bbbp, bace, freesolv), encoding="utf-8"
    )

    print(f"[ok] wrote tables to {OUT_DIR}")
    for name, t in (("bbbp", bbbp), ("bace", bace), ("clintox", clintox),
                    ("tox21", tox21), ("freesolv", freesolv), ("esol", esol), ("lipo", lipo)):
        print(f"  {name:<9}: {list(t)}")
    if "SEG" not in bace or "T6" not in bace:
        print("[warn] BACE V2-T5 / T6 missing -- re-run HPC pipeline.")


if __name__ == "__main__":
    main()
