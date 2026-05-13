"""Summarize fair BACE V0/V1/V2 results into a reviewable report."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "bace_fair_results_report.md"
SUMMARY_PATH = ROOT / "bace_fair_results_summary.json"


def load_json(name):
    path = ROOT / name
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def summary_from_values(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_cells": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "values": [round(float(v), 4) for v in arr.tolist()],
    }


def nested_aucs(result, seeds=(9, 19, 29)):
    aucs = []
    missing = []
    per_seed = result.get("results_per_seed", {}) if result else {}
    for seed in seeds:
        key = f"seed_{seed}"
        if key not in per_seed:
            missing.append(key)
            continue
        eval_splits = per_seed[key].get("eval_splits", {})
        for split in (9, 19, 29):
            split_key = f"split_{split}"
            restart_key = f"restart_{split}"
            item = eval_splits.get(split_key, eval_splits.get(restart_key))
            if item is None:
                missing.append(f"{key}/{split_key}")
                continue
            aucs.append(float(item["test_auc"]))
    return aucs, missing


def add_row(rows, key, label, data_root, split_source, topology, signal, result, section=None):
    if section:
        src = section
        summary = src.get("summary")
        values = summary.get("all_9_test_aucs", []) if summary else []
        stats = summary_from_values(values)
        missing = [] if stats and stats["n_cells"] == 9 else ["incomplete summary"]
    else:
        values, missing = nested_aucs(result)
        stats = summary_from_values(values)

    rows.append({
        "key": key,
        "label": label,
        "data_root": data_root,
        "split_source": split_source,
        "topology": topology,
        "signal": signal,
        "n_cells": stats["n_cells"] if stats else 0,
        "mean": stats["mean"] if stats else None,
        "std": stats["std"] if stats else None,
        "missing": missing,
    })


def fmt_metric(row):
    if row["mean"] is None:
        return "pending"
    return f'{row["mean"]:.4f} +/- {row["std"]:.4f}'


def main():
    baseline = load_json("baseline_unifold_results.json")
    v1_static = load_json("v1_static_bace_results.json")
    v2_t5 = load_json("v2_t5_bace_results.json")
    nullify = load_json("v2_t5_nullify_bace_results.json")

    rows = []
    add_row(
        rows,
        "v0_unifold",
        "V0 2D baseline",
        "down_task_unifold",
        "Uni-Mol scaffold k10 seed42; fold0=test fold1=valid",
        "chemical bond edge_index",
        "none",
        baseline,
        section=baseline.get("baseline") if baseline else None,
    )
    add_row(
        rows,
        "fp_only",
        "FP-only",
        "down_task_unifold",
        "Uni-Mol scaffold k10 seed42; fold0=test fold1=valid",
        "no GNN graph path in downstream head",
        "fingerprints only",
        baseline,
        section=baseline.get("fp_only") if baseline else None,
    )
    add_row(
        rows,
        "v1_static",
        "V1 static Uni-Mol",
        "down_task_v2",
        "Uni-Mol scaffold k10 seed42; fold0=test fold1=valid",
        "chemical bond edge_index",
        "fixed softplus(mean(pair_repr_edge)) edge_weight",
        v1_static,
    )
    add_row(
        rows,
        "v2_t5",
        "V2-T5 learned edge mapper",
        "down_task_v2",
        "Uni-Mol scaffold k10 seed42; fold0=test fold1=valid",
        "chemical bond edge_index",
        "pair_repr_edge -> learned PairToEdgeWeight",
        v2_t5,
    )
    add_row(
        rows,
        "v2_t5_nullify",
        "V2-T5-nullify",
        "down_task_v2",
        "Uni-Mol scaffold k10 seed42; fold0=test fold1=valid",
        "chemical bond edge_index",
        "pair_repr path present, edge_weight forced to 1",
        nullify,
    )

    complete_rows = [row for row in rows if row["n_cells"] == 9]
    interpretation = []
    fp = next(row for row in rows if row["key"] == "fp_only")
    v2 = next(row for row in rows if row["key"] == "v2_t5")
    null = next(row for row in rows if row["key"] == "v2_t5_nullify")
    v1 = next(row for row in rows if row["key"] == "v1_static")

    if v2["mean"] is not None and fp["mean"] is not None:
        if v2["mean"] < fp["mean"]:
            interpretation.append("V2-T5 is below FP-only, so current evidence does not show the full FP+GNN predictor beating fingerprints alone.")
        else:
            interpretation.append("V2-T5 beats FP-only under this protocol.")
    if null["n_cells"] < 9:
        interpretation.append("V2-T5-nullify is still incomplete; do not use it for final conclusions yet.")
    elif v2["mean"] is not None and null["mean"] is not None:
        if null["mean"] > v2["mean"]:
            interpretation.append("Nullify beats non-nullified V2-T5, suggesting the learned edge weights may hurt rather than help.")
        else:
            interpretation.append("Non-nullified V2-T5 beats nullify, supporting a useful learned pair signal.")
    if v1["n_cells"] < 9:
        interpretation.append("V1 static Uni-Mol is pending or incomplete; static 3D signal has not been cleanly established.")

    report_lines = [
        "# BACE Fair V0/V1/V2 Results",
        "",
        "Protocol: BACE only; Uni-Mol scaffold k10 seed42 split; fold 0 test, fold 1 valid; pretrain seeds 9/19/29; eval restarts 9/19/29; ROC-AUC.",
        "",
        "| Variant | Data root | Topology | 3D / pair signal | Cells | ROC-AUC | Status |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        status = "complete" if row["n_cells"] == 9 and not row["missing"] else "incomplete"
        if row["missing"]:
            status += f" ({len(row['missing'])} missing)"
        report_lines.append(
            f"| {row['label']} | `{row['data_root']}` | {row['topology']} | "
            f"{row['signal']} | {row['n_cells']} | {fmt_metric(row)} | {status} |"
        )

    report_lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    report_lines.extend(f"- {item}" for item in interpretation)
    report_lines.extend([
        "",
        "## V1 Static Weight Definition",
        "",
        "- Distance source: no hand-crafted distance is used; weights come from Uni-Mol `encoder_pair_rep` exported as `pair_repr_edge`.",
        "- Scalarization: `edge_weight = softplus(mean(pair_repr_edge[i,j,:]))` for each chemical bond edge.",
        "- Normalization: none.",
        "- Bond type scaling: none.",
        "- Missing conformer fallback: none in this runner; `down_task_v2` must already contain `pair_repr_edge`.",
        "",
    ])

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps({"rows": rows, "interpretation": interpretation}, indent=2), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    for row in rows:
        print(f"{row['label']}: n={row['n_cells']} auc={fmt_metric(row)}")


if __name__ == "__main__":
    main()
