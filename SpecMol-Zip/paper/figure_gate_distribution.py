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
    p.add_argument("--pattern", default="bace_v2t5_seed*_b5_audit.json")
    p.add_argument("--out", default=str(REPO / "paper" / "figures" / "gate_distribution_bace_v2t5.pdf"))
    p.add_argument("--title", default="V2-T5 trained per-bond gate distribution on BACE (b=+5 init)")
    args = p.parse_args()

    paths = sorted(Path(args.stats_dir).glob(args.pattern))
    if not paths:
        raise SystemExit(f"no JSONs matching {args.pattern} under {args.stats_dir}")

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, path in enumerate(paths):
        payload = _read(path)
        gate = payload["edge_weight_post_sigmoid_nonloop"]
        counts = gate["histogram"]["counts"]
        edges = gate["histogram"]["edges"]
        centers = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
        widths = [b - a for a, b in zip(edges, edges[1:])]
        # Normalize to density-like form (count / total) for cross-seed comparability
        total = float(sum(counts)) or 1.0
        densities = [c / total for c in counts]
        # Tag from filename, e.g. 'seed9'
        tag = path.stem.split("_")[2] if len(path.stem.split("_")) > 2 else path.stem
        bias = payload.get("global_bias_b", float("nan"))
        ax.bar(centers, densities, width=widths, alpha=0.45,
               color=colors[i % len(colors)],
               label=f"{tag} (mean={gate['mean']:.3f}, bias={bias:.2f})",
               edgecolor=colors[i % len(colors)])

    # Mark the initial gate value at the right for context (sigma(+5)~=0.993)
    ax.axvline(0.993, color="black", linestyle=":", linewidth=1,
               label=r"init $\sigma(+5)\approx 0.993$")
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
