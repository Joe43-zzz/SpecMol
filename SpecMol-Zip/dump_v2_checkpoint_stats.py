"""Standalone saturation diagnostic for an existing V2 / T6 / T7 checkpoint.

Loads a .pkl produced by `main_pretrain.py --use_v2 ...`, attaches the
forward hooks from `dump_mlp_phi_stats`, runs one inference pass over a
pretraining dataset, and writes `mlp_phi_stats/<tag>.json`. CPU is fine
(no gradients, single pass).

Why this exists: re-running `main_pretrain.py` just to dump the gate
distribution takes ~4h on HPC. With this script we can audit every
already-trained V2-T5 / T6 / T7 checkpoint we have for the bias-saturation
hypothesis in seconds.

Optional `--per-edge-out <path.npz>` additionally saves, for every
chemical-bond edge in the pretraining pass, a (gate, edge_attr) pair
so a local analysis script can correlate gate values with bond type /
stereo / chemistry. Both .npz outputs are typically <5 MB per dataset.

Usage:
  python SpecMol-Zip/dump_v2_checkpoint_stats.py \\
    --checkpoint pkl/best_spec_model_bace_seed9_job134628_*.pkl \\
    --data-path down_task_bace_v2 \\
    --task bace --tag bace_v2t5_seed9_audit \\
    --per-edge-out mlp_phi_stats/bace_v2t5_seed9_per_edge.npz

Variant hyperparams are read from --variant (default v2t5). For T6/T7
checkpoints pass --variant t6 / t7. The model is constructed with the
same hyperparams used by `hpc/run_dataset_training.sbatch` and the
default `main_pretrain.py` argparse.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import DataLoader

REPO = Path(__file__).resolve().parent

# Reuse the production helpers verbatim.
sys.path.insert(0, str(REPO))
from main_pretrain import dump_mlp_phi_stats  # type: ignore
from train_utils import load_model_state_dict  # type: ignore
from utils_fp_downstream import TestbedDataset  # type: ignore


def build_v2_model(variant, hid_dim, K, dprate, dropout, is_bns, act_fn, type_, bias_init,
                    pair_dim, pair_hidden_dim, proj_dim,
                    t6_safe_frozen_zero, t6_safe_delta_init_std,
                    t7_num_heads, t7_head_dim, t7_dropout, t7_init_std):
    from model_gnn_pre_v2 import LH_Direct_V2  # local import (heavy)
    t6 = variant == "t6"
    t6_safe = variant == "t6safe"
    t7 = variant == "t7"
    return LH_Direct_V2(
        in_dim=93,
        hid_dim=hid_dim,
        K=K,
        dprate=dprate,
        dropout=dropout,
        is_bns=is_bns,
        act_fn=act_fn,
        type=type_,
        pair_dim=pair_dim,
        pair_hidden_dim=pair_hidden_dim,
        proj_dim=proj_dim,
        t6=t6,
        t6_safe=t6_safe,
        t6_safe_frozen_zero=t6_safe_frozen_zero,
        t6_safe_delta_init_std=t6_safe_delta_init_std,
        t7=t7,
        t7_num_heads=t7_num_heads,
        t7_head_dim=t7_head_dim,
        t7_dropout=t7_dropout,
        t7_init_std=t7_init_std,
        bias_init=bias_init,
    )


class _StubArgs:
    """Minimal duck-typed object whose attributes dump_mlp_phi_stats reads.

    dump_mlp_phi_stats consults args.t6, args.t6_safe, args.t7 to label the
    variant in the output JSON. Nothing else.
    """
    def __init__(self, variant):
        self.t6 = variant == "t6"
        self.t6_safe = variant == "t6safe"
        self.t7 = variant == "t7"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best_spec_model_*.pkl")
    p.add_argument("--data-path", required=True, help="root for TestbedDataset, e.g. down_task_bace_v2")
    p.add_argument("--task", required=True, help="bace | bbbp | ...")
    p.add_argument("--variant", default="v2t5", choices=["v2t5", "t6", "t6safe", "t7"])
    p.add_argument("--tag", default=None, help="output filename base (defaults to checkpoint stem)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    # Hyperparams to reconstruct LH_Direct_V2; defaults match main_pretrain.py argparse
    p.add_argument("--hid-dim", type=int, default=512)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--dprate", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--is-bns", action="store_true")
    p.add_argument("--act-fn", default="relu")
    p.add_argument("--type", default="tri", dest="type_")
    p.add_argument("--bias-init", type=float, default=5.0,
                   help="Must match the training-time bias_init for the loaded checkpoint")
    p.add_argument("--pair-dim", type=int, default=64)
    p.add_argument("--pair-hidden-dim", type=int, default=32)
    p.add_argument("--proj-dim", type=int, default=32)
    p.add_argument("--t6-safe-frozen-zero", action="store_true")
    p.add_argument("--t6-safe-delta-init-std", type=float, default=1e-3)
    p.add_argument("--t7-num-heads", type=int, default=4)
    p.add_argument("--t7-head-dim", type=int, default=32)
    p.add_argument("--t7-dropout", type=float, default=0.0)
    p.add_argument("--t7-init-std", type=float, default=0.02)
    p.add_argument("--random-seed", type=int, default=0, help="for dump_mlp_phi_stats JSON metadata only")
    p.add_argument("--per-edge-out", default=None,
                   help="Optional path to write a .npz with per-edge (gate, edge_attr) pairs for "
                        "local substructure-attribution analysis. Saved as two arrays: "
                        "gate [E], edge_attr [E, F]. Self-loops (gate=1.0) are filtered out.")
    args = p.parse_args()

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    tag = args.tag or ckpt.stem

    print(f"[diag] building V2 model (variant={args.variant}, bias_init={args.bias_init})")
    model = build_v2_model(
        variant=args.variant,
        hid_dim=args.hid_dim, K=args.K, dprate=args.dprate, dropout=args.dropout,
        is_bns=args.is_bns, act_fn=args.act_fn, type_=args.type_,
        bias_init=args.bias_init,
        pair_dim=args.pair_dim, pair_hidden_dim=args.pair_hidden_dim, proj_dim=args.proj_dim,
        t6_safe_frozen_zero=args.t6_safe_frozen_zero,
        t6_safe_delta_init_std=args.t6_safe_delta_init_std,
        t7_num_heads=args.t7_num_heads, t7_head_dim=args.t7_head_dim,
        t7_dropout=args.t7_dropout, t7_init_std=args.t7_init_std,
    ).to(args.device)

    print(f"[diag] loading state_dict from {ckpt}")
    state = load_model_state_dict(str(ckpt), map_location=args.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[diag] WARNING missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[diag] WARNING unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    print(f"[diag] loading dataset from {args.data_path} task={args.task}")
    data = TestbedDataset(root=args.data_path, dataset='all', task=args.task, type=args.type_)

    print("[diag] calling dump_mlp_phi_stats")
    dump_mlp_phi_stats(
        spec_model=model,
        pretrain_data=data,
        batch_size=args.batch_size,
        device=args.device,
        task=args.task,
        random_seed=args.random_seed,
        tag=tag,
        args=_StubArgs(args.variant),
    )
    print(f"[diag] wrote mlp_phi_stats/{tag}.json")

    if args.per_edge_out:
        print(f"[diag] collecting per-edge (gate, edge_attr) pairs for {args.per_edge_out}")
        import numpy as np
        gate_buf = []
        attr_buf = []

        def _gate_hook(_mod, _inp, output):
            gate_buf.append(output.detach().cpu())

        h = model.pair_to_edge_weight.register_forward_hook(_gate_hook)
        attr_batches = []
        try:
            loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    gate_buf_len_before = sum(g.numel() for g in gate_buf)
                    _ = model(batch, args.device)
                    # data.edge_attr is one row per chem-bond edge (matches edge_index)
                    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
                        attr_batches.append(batch.edge_attr.detach().cpu())
        finally:
            h.remove()
        gate = torch.cat([g.flatten() for g in gate_buf], dim=0) if gate_buf else torch.empty(0)
        edge_attr = torch.cat(attr_batches, dim=0) if attr_batches else torch.empty(0, 0)
        # Filter self-loops where pair_to_edge_weight hard-codes 1.0
        # (chem bonds get learned weights, self-loops do not)
        if gate.numel() and edge_attr.numel():
            if gate.numel() == edge_attr.size(0):
                mask = gate != 1.0
                gate = gate[mask]
                edge_attr = edge_attr[mask]
            else:
                print(f"[diag] WARNING gate ({gate.numel()}) vs edge_attr "
                      f"({edge_attr.size(0)}) shape mismatch; saving unfiltered")
        os.makedirs(os.path.dirname(args.per_edge_out) or ".", exist_ok=True)
        np.savez_compressed(
            args.per_edge_out,
            gate=gate.numpy().astype("float32"),
            edge_attr=edge_attr.numpy().astype("float32"),
        )
        print(f"[diag] wrote {args.per_edge_out}: gate {tuple(gate.shape)}, edge_attr {tuple(edge_attr.shape)}")


if __name__ == "__main__":
    sys.exit(main())
