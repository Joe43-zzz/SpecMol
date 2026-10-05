"""Plot the V2-T5 trained-gate distribution across seeds for the paper
supplementary. Reads mlp_phi_stats/bace_v2t5_seed*_b5_audit.json and
emits a single PDF showing the post-sigmoid edge-weight histogram per
seed on the same axes.

Usage:
    python paper/figure_gate_distribution.py
    python paper/figure_gate_distribution.py --out paper/figures/gate_distribution_bace_v2t5.pdf

The figure is the visual companion to the post-hoc audit paragraph in
main.tex (V2-T5 method section): it makes the "unimodal right-skewed,
mode ~0.06, tail < 0.27" claim glanceable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]


def _read(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats-dir", default=str(REPO / "mlp_phi_stats"))
    p.add_argument("--patterns", nargs="+",
                   default=["bace_v2t5_seed*_b5_audit.json", "bace_v2t5_seed*_b1_audit.json"],
                   help="One or more glob patterns. Each pattern's matches are drawn "
                        "as a group on the same axes, color-coded by pattern.")
    # back-compat: --pattern singular maps to a one-pattern list
    p.add_argument("--pattern", default=None,
                   help="Deprecated alias for --patterns (single pattern).")
    p.add_argument("--out", default=str(REPO / "paper" / "figures" / "gate_distribution_bace_v2t5.pdf"))
    p.add_argument("--title", default="V2-T5 trained per-bond gate distribution on BACE (b=+5 init)")
    args = p.parse_args()

    patterns = [args.pattern] if args.pattern else list(args.patterns)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6))
    # Pattern → palette. b=+5 group blue-ish, b=+1 group red-ish.
    group_palettes = [
        ["tab:blue", "tab:cyan", "navy"],
        ["tab:red", "tab:orange", "darkred"],
        ["tab:green", "olive", "darkgreen"],
    ]
    init_markers = set()
    for grp_i, pattern in enumerate(patterns):
        paths = sorted(Path(args.stats_dir).glob(pattern))
        if not paths:
            print(f"[warn] no JSONs matching {pattern} under {args.stats_dir}")
            continue
        palette = group_palettes[grp_i % len(group_palettes)]
        for i, path in enumerate(paths):
            payload = _read(path)
            gate = payload["edge_weight_post_sigmoid_nonloop"]
            counts = gate["histogram"]["counts"]
            edges = gate["histogram"]["edges"]
            centers = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
            widths = [b - a for a, b in zip(edges, edges[1:])]
            total = float(sum(counts)) or 1.0
            densities = [c / total for c in counts]
            tag = path.stem.split("_")[2] if len(path.stem.split("_")) > 2 else path.stem
            bias_init_tag = "b5" if "_b5_" in path.stem else "b1" if "_b1_" in path.stem else "?"
            bias_final = payload.get("global_bias_b", float("nan"))
            color = palette[i % len(palette)]
            ax.bar(centers, densities, width=widths, alpha=0.45,
                   color=color, edgecolor=color,
                   label=f"{tag} {bias_init_tag} (mean={gate['mean']:.3f}, b={bias_final:.2f})")
        # Add init marker once per pattern group
        if "_b5_" in pattern and "b5" not in init_markers:
            ax.axvline(0.993, color="tab:blue", linestyle=":", linewidth=1,
                       label=r"init $\sigma(+5)\approx 0.993$")
            init_markers.add("b5")
        if "_b1_" in pattern and "b1" not in init_markers:
            ax.axvline(0.731, color="tab:red", linestyle=":", linewidth=1,
                       label=r"init $\sigma(+1)\approx 0.731$")
            init_markers.add("b1")
    ax.set_xlabel(r"per-bond gate $g_{ij} = \sigma(\mathrm{MLP}_\phi(P_{ij}) + b)$")
    ax.set_ylabel("fraction of bonds in batch")
    ax.set_title(args.title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=150)
    print(f"[ok] wrote {out_path}")
    print(f"[ok] wrote {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
