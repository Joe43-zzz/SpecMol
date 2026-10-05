"""E1: T7 saturation probe.

Direct evidence test for adversarial diagnosis:
  - attn.max(-1).values.mean()  → if >0.9, softmax saturated
  - attn entropy                → if collapse to <0.5, attention is degenerate
  - pair_repr.norm() per K-step → if growing >2x per step, positive feedback

Runs only 5 epochs on BACE seed=9. Fast (~30s).

Decision rule:
  - attn.max < 0.5 AND entropy > 1.5  → attention is healthy, current T7 is OK
  - attn.max > 0.9 OR entropy < 0.3   → SATURATED, fixes needed before benchmark
  - pair_repr per-K-step ratio < 1.5  → bounded, no feedback loop
  - pair_repr per-K-step ratio > 2.0  → EXPLODING, need normalization
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.optim as optim
import torch.nn.functional as F

try:
    from utils_fp_downstream import TestbedDataset
except ImportError:
    from torch_geometric.data import InMemoryDataset
    class TestbedDataset(InMemoryDataset):
        def __init__(self, root, dataset='all', task='bace', type='tri', seed=None):
            pt_name = f"{task}_{dataset}.pt"
            pt_path = os.path.join(root, 'processed', pt_name)
            super().__init__(root=None)
            self.data, self.slices = torch.load(pt_path, weights_only=False)
        def _download(self): pass
        def _process(self): pass

from torch_geometric.data import DataLoader
from model_gnn_pre_v2 import LH_Direct_V2
from train_utils import ours_loss, set_seed


def probe_attention(model, batch, device):
    """Forward one batch through T7, capture attn distribution + pair_repr trajectory."""
    pair_attn = model.encoder.prop1.pair_attn

    # We need to hook into pair_attn.forward to grab attn weights and pair_repr at each call
    captured = {"attn_max": [], "attn_entropy": [], "pair_repr_norm": []}

    original_forward = pair_attn.forward

    def hooked_forward(x, pair_repr_edge, pair_edge_index):
        from torch_geometric.utils import softmax, scatter
        N = x.size(0)
        E = pair_edge_index.size(1)
        if E == 0:
            return torch.zeros_like(x), pair_repr_edge.new_zeros(0, pair_attn.pair_dim)
        src = pair_edge_index[0]
        dst = pair_edge_index[1]
        # Apply LayerNorm if module has it (fixed version)
        x_n = pair_attn.x_norm(x) if hasattr(pair_attn, 'x_norm') else x
        pair_n = pair_attn.pair_norm(pair_repr_edge) if hasattr(pair_attn, 'pair_norm') else pair_repr_edge
        Q = pair_attn.q(x_n).view(N, pair_attn.H, pair_attn.d)
        K = pair_attn.k(x_n).view(N, pair_attn.H, pair_attn.d)
        V = pair_attn.v(x_n).view(N, pair_attn.H, pair_attn.d)
        qk = (Q[src] * K[dst]).sum(-1) / (pair_attn.d ** 0.5)
        bias = pair_attn.bias_proj(pair_n)
        logits = qk + bias
        attn = softmax(logits.float(), src).to(logits.dtype)
        # CAPTURE
        with torch.no_grad():
            captured["attn_max"].append(float(attn.max(dim=0).values.mean().item()))
            # entropy: -sum(p log p) per (edge, head). We compute per-source-node group entropy.
            # But simpler: mean over all edges of -p*log(p+1e-12) summed at the per-edge level
            ent = -(attn * (attn + 1e-12).log()).mean().item()
            captured["attn_entropy"].append(float(ent))
            captured["pair_repr_norm"].append(float(pair_repr_edge.norm().item()))
        msg = attn.unsqueeze(-1) * V[dst]
        out = scatter(msg, src, dim=0, dim_size=N, reduce='sum').reshape(N, pair_attn.H * pair_attn.d)
        out = pair_attn.out_proj(out)
        pair_delta = pair_attn.delta_proj(logits)
        return out, pair_delta

    pair_attn.forward = hooked_forward
    try:
        model.eval()
        with torch.no_grad():
            _ = model(batch, device)
    finally:
        pair_attn.forward = original_forward

    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--path', type=str, default='down_task_v2')
    args = parser.parse_args()

    device = args.device
    seed = 9
    set_seed(seed)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print("\nLoading dataset...")
    data = TestbedDataset(root=args.path, dataset='all', task='bace', type='tri')
    print(f"  {len(data)} molecules loaded")

    print("\nCreating T7 model (default settings, no fixes)...")
    model = LH_Direct_V2(
        in_dim=93, hid_dim=512, K=10, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn='relu', type='tri', pair_dim=64, t7=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)

    pair_attn = model.encoder.prop1.pair_attn
    print(f"  bias_proj.weight at init: norm={pair_attn.bias_proj.weight.norm().item():.4f}")
    print(f"  bias output magnitude estimate at init:")
    with torch.no_grad():
        # Sample one batch to estimate bias magnitude vs qk magnitude
        sample_batch = next(iter(DataLoader(data[:32], batch_size=32, shuffle=False))).to(device)
        x_test = sample_batch.x
        pair_test = sample_batch.pair_repr_edge
        # Apply LayerNorm if available (matches actual forward path)
        x_n_test = pair_attn.x_norm(x_test) if hasattr(pair_attn, 'x_norm') else x_test
        pair_n_test = pair_attn.pair_norm(pair_test) if hasattr(pair_attn, 'pair_norm') else pair_test
        Q_test = pair_attn.q(x_n_test).view(-1, pair_attn.H, pair_attn.d)
        K_test = pair_attn.k(x_n_test).view(-1, pair_attn.H, pair_attn.d)
        bias_test = pair_attn.bias_proj(pair_n_test)
        # Use first chem edge or just any pair
        src_test = sample_batch.pair_edge_index[0][:100]
        dst_test = sample_batch.pair_edge_index[1][:100]
        qk_test = (Q_test[src_test] * K_test[dst_test]).sum(-1) / (pair_attn.d ** 0.5)
        print(f"  ATTENTION INIT MAGNITUDES (with LayerNorm if present):")
        print(f"    qk magnitude (mean abs):    {qk_test.abs().mean().item():.4f}")
        print(f"    bias magnitude (mean abs):  {bias_test[:100].abs().mean().item():.4f}")
        print(f"    bias / qk ratio:            {bias_test[:100].abs().mean().item() / max(qk_test.abs().mean().item(), 1e-9):.2f}x   (target: <10x for healthy attention)")
        print(f"    pair_repr norm (raw):       {pair_test.norm().item():.2f}")
        print(f"    pair_repr norm (LN'd):      {pair_n_test.norm().item():.2f}")
        print(f"    pair_repr per-edge norm:    {pair_n_test.norm(dim=-1).mean().item():.4f}")

    # ========== Per-epoch training + probe ==========
    print(f"\n{'Epoch':>5} | {'Loss':>10} | {'attn_max':>10} | {'attn_ent':>10} | {'pr_norm_t0':>12} | {'pr_norm_tK':>12} | {'ratio':>7}")
    print("-" * 90)

    sample_batch = next(iter(DataLoader(data[:32], batch_size=32, shuffle=False))).to(device)
    for epoch in range(args.epochs):
        # Train one epoch
        model.train()
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
        total_loss, total_num = 0.0, 0
        for batch in loader:
            low_x, high_x, spec_x, x_fp = model(batch, device)
            loss = ours_loss(low_x, high_x, spec_x, x_fp)
            total_num += batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / total_num

        # Probe attention with hooked forward
        captured = probe_attention(model, sample_batch, device)
        # Each call to pair_attn happens K=10 times (per high+low pass) × 2 = 20 calls
        attn_max_mean = sum(captured["attn_max"]) / len(captured["attn_max"])
        attn_ent_mean = sum(captured["attn_entropy"]) / len(captured["attn_entropy"])
        pr_t0 = captured["pair_repr_norm"][0]
        pr_tK = captured["pair_repr_norm"][-1]
        ratio = pr_tK / max(pr_t0, 1e-9)

        print(f"{epoch:>5} | {avg_loss:>10.6f} | {attn_max_mean:>10.4f} | {attn_ent_mean:>10.4f} | {pr_t0:>12.2f} | {pr_tK:>12.2f} | {ratio:>7.2f}x")

    # Detailed K-step trajectory at last epoch
    print("\n=== Per-K-step pair_repr evolution (last epoch, one branch) ===")
    captured = probe_attention(model, sample_batch, device)
    for i, (am, ae, pr) in enumerate(zip(captured["attn_max"], captured["attn_entropy"], captured["pair_repr_norm"])):
        print(f"  call {i:>2}: attn_max={am:.4f} entropy={ae:.4f} pair_repr_norm={pr:.2f}")

    # Verdicts
    print("\n--- VERDICT ---")
    final_attn_max = sum(captured["attn_max"]) / len(captured["attn_max"])
    final_attn_ent = sum(captured["attn_entropy"]) / len(captured["attn_entropy"])
    pr_norms = captured["pair_repr_norm"]
    max_ratio = max(pr_norms[i+1]/max(pr_norms[i], 1e-9) for i in range(len(pr_norms)-1)) if len(pr_norms) > 1 else 1.0

    print(f"Final attention max:       {final_attn_max:.4f}  (>0.9 = saturated)")
    print(f"Final attention entropy:   {final_attn_ent:.4f}  (<0.3 = collapsed)")
    print(f"Max per-K-step pr ratio:   {max_ratio:.2f}x      (>2.0 = exploding)")

    saturated = final_attn_max > 0.9
    collapsed = final_attn_ent < 0.3
    exploding = max_ratio > 2.0

    if saturated or collapsed or exploding:
        print("\n>>> FAIL: Adversarial diagnosis CONFIRMED. Apply LayerNorm + fix init before benchmark.")
        if saturated: print(f"    - softmax saturated (attn_max={final_attn_max:.3f})")
        if collapsed: print(f"    - attention entropy collapsed (entropy={final_attn_ent:.3f})")
        if exploding: print(f"    - pair_repr exploding (per-K ratio {max_ratio:.2f}x)")
    else:
        print("\n>>> PASS: Attention is healthy. Adversarial agents over-cautious. Current T7 OK.")


if __name__ == '__main__':
    main()
