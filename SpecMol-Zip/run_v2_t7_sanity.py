"""V2-T7 BACE training sanity check.

50 epochs, seed=9, batch_size=512 (match T5/T6 baseline), GPU.
Reports: loss curve, T7 module weight norms (q/k/v/out_proj/bias_proj/delta_proj),
pair_repr diff after K-step, GPU memory peak.

T7 = sparse pair-biased multi-head attention with bidirectional pair update.
Mutually exclusive with T6.
"""

import argparse
import gc
import math
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

# Try standard import first (dgcl env), fall back to preloaded .pt
try:
    from utils_fp_downstream import TestbedDataset
    USE_PRELOADED = False
except ImportError:
    from torch_geometric.data import InMemoryDataset
    class TestbedDataset(InMemoryDataset):
        """Bypass rdkit import by loading .pt directly."""
        def __init__(self, root, dataset='all', task='bace', type='tri', seed=None):
            pt_name = f"{task}_{dataset}.pt"
            pt_path = os.path.join(root, 'processed', pt_name)
            super().__init__(root=None)
            self.data, self.slices = torch.load(pt_path, weights_only=False)
        def _download(self): pass
        def _process(self): pass
    USE_PRELOADED = True

from torch_geometric.data import DataLoader
from model_gnn_pre_v2 import LH_Direct_V2
from train_utils import ours_loss, set_seed


def get_gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--path', type=str, default='down_task_v2')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--head_dim', type=int, default=32)
    parser.add_argument('--init_std', type=float, default=0.02)
    args = parser.parse_args()

    device = args.device
    seed = 9
    epochs = args.epochs
    batch_size = args.batch_size
    lr = 0.0001
    hid_dim = 512
    K = 10
    path = args.path

    set_seed(seed)

    print(f"Device: {device}")
    print(f"Preloaded dataset: {USE_PRELOADED}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print("\nLoading dataset...")
    data = TestbedDataset(root=path, dataset='all', task='bace', type='tri')
    print(f"  {len(data)} molecules loaded")

    # Snapshot initial pair_repr for diff measurement
    sample_batch = next(iter(DataLoader(data[:32], batch_size=32, shuffle=False)))
    pair_repr_init_snapshot = sample_batch.pair_repr_edge.clone()

    print("\nCreating T7 model...")
    model = LH_Direct_V2(
        in_dim=93, hid_dim=hid_dim, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn='relu', type='tri', pair_dim=64, t7=True,
        t7_num_heads=args.num_heads, t7_head_dim=args.head_dim,
        t7_init_std=args.init_std,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    # Record inits — T7 has 6 trainable matrices in pair_attn
    pair_attn = model.encoder.prop1.pair_attn
    bias_init = model.pair_to_edge_weight.mlp[2].bias.item()
    q_init = pair_attn.q.weight.norm().item()
    k_init = pair_attn.k.weight.norm().item()
    v_init = pair_attn.v.weight.norm().item()
    out_proj_init = pair_attn.out_proj.weight.norm().item()
    bias_proj_init = pair_attn.bias_proj.weight.norm().item()
    delta_proj_init = pair_attn.delta_proj.weight.norm().item()

    print(f"  Bias init:       {bias_init:.4f}")
    print(f"  Q norm init:     {q_init:.6f}")
    print(f"  K norm init:     {k_init:.6f}")
    print(f"  V norm init:     {v_init:.6f}")
    print(f"  out_proj init:   {out_proj_init:.6f} (small, std={args.init_std})")
    print(f"  bias_proj init:  {bias_proj_init:.6f}")
    print(f"  delta_proj init: {delta_proj_init:.6f} (small, std={args.init_std})")

    loss_history = []
    epoch_times = []

    print(f"\n{'Epoch':>5} | {'Loss':>10} | {'Bias':>8} | {'Q':>8} | {'K':>8} | {'V':>8} | {'out_proj':>10} | {'Time(s)':>7}")
    print("-" * 90)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        total_loss, total_num = 0.0, 0

        for batch in loader:
            low_x, high_x, spec_x, x_fp = model(batch, device)
            loss = ours_loss(low_x, high_x, spec_x, x_fp)

            if torch.isnan(loss):
                print(f"\n  *** NaN at epoch {epoch}! ***")
                sys.exit(1)

            total_num += batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dt = time.time() - t0
        epoch_times.append(dt)
        avg_loss = total_loss / total_num
        loss_history.append(avg_loss)

        bias_now = model.pair_to_edge_weight.mlp[2].bias.item()
        q_now = pair_attn.q.weight.norm().item()
        k_now = pair_attn.k.weight.norm().item()
        v_now = pair_attn.v.weight.norm().item()
        op_now = pair_attn.out_proj.weight.norm().item()

        if epoch < 5 or epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{epoch:>5} | {avg_loss:>10.6f} | {bias_now:>8.4f} | {q_now:>8.4f} | {k_now:>8.4f} | {v_now:>8.4f} | {op_now:>10.6f} | {dt:>7.1f}")

    # --- pair_repr diff measurement ---
    model.eval()
    with torch.no_grad():
        sample_batch2 = next(iter(DataLoader(data[:32], batch_size=32, shuffle=False)))
        feat = sample_batch2.x.to(device)
        edge_index = sample_batch2.edge_index.to(device)
        batch_vec = sample_batch2.batch.to(device)
        pr = sample_batch2.pair_repr_edge.to(device).clone()
        pei = sample_batch2.pair_edge_index.to(device)

        ew = model.pair_to_edge_weight(pr, pei, edge_index, batch_vec)
        result = model.encoder.prop1(
            feat, edge_index, ew, highpass=True,
            pair_repr_edge=pr, pair_edge_index=pei,
            batch=batch_vec, pair_to_ew=model.pair_to_edge_weight,
        )
        if isinstance(result, tuple):
            _, pair_repr_after = result
        pair_repr_diff = (pair_repr_after.cpu() - pair_repr_init_snapshot).norm().item()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("T7 SANITY CHECK SUMMARY")
    print("=" * 60)

    bias_final = model.pair_to_edge_weight.mlp[2].bias.item()
    q_final = pair_attn.q.weight.norm().item()
    k_final = pair_attn.k.weight.norm().item()
    v_final = pair_attn.v.weight.norm().item()
    op_final = pair_attn.out_proj.weight.norm().item()
    bp_final = pair_attn.bias_proj.weight.norm().item()
    dp_final = pair_attn.delta_proj.weight.norm().item()

    print(f"\nLoss curve:")
    print(f"  Epoch  0: {loss_history[0]:.6f}")
    for i in [4, 9, 19, 29, 39, 49]:
        if i < len(loss_history):
            print(f"  Epoch {i:>2}: {loss_history[i]:.6f}")
    print(f"  Decreased: {loss_history[-1] < loss_history[0]}")
    print(f"  Any NaN: {any(math.isnan(l) for l in loss_history)}")

    decreasing = sum(1 for i in range(1, len(loss_history)) if loss_history[i] < loss_history[i-1])
    print(f"  Decreasing steps: {decreasing}/{epochs-1}")

    print(f"\nBias:       {bias_init:.4f} -> {bias_final:.4f}")
    print(f"Q norm:     {q_init:.6f} -> {q_final:.6f}")
    print(f"K norm:     {k_init:.6f} -> {k_final:.6f}")
    print(f"V norm:     {v_init:.6f} -> {v_final:.6f}")
    print(f"out_proj:   {out_proj_init:.6f} -> {op_final:.6f}")
    print(f"bias_proj:  {bias_proj_init:.6f} -> {bp_final:.6f}")
    print(f"delta_proj: {delta_proj_init:.6f} -> {dp_final:.6f}")

    print(f"\npair_repr diff (32 mols, post-training K-step vs initial): {pair_repr_diff:.4f}")

    gpu_peak = get_gpu_mem_mb()
    avg_time = np.mean(epoch_times)
    print(f"\nGPU peak memory: {gpu_peak:.0f} MB")
    print(f"Avg epoch time: {avg_time:.1f}s")
    print(f"Total time: {sum(epoch_times):.0f}s")

    # Verdicts — verify no dead-init lock
    print("\n--- VERDICTS ---")
    ok = True
    if loss_history[-1] >= loss_history[0]:
        print("FAIL: loss did not decrease")
        ok = False
    if abs(q_final - q_init) < 1e-6:
        print("FAIL: Q norm unchanged — dead path!")
        ok = False
    if abs(v_final - v_init) < 1e-6:
        print("FAIL: V norm unchanged — DEAD-INIT LOCK!")
        ok = False
    if abs(op_final - out_proj_init) < 1e-6:
        print("FAIL: out_proj unchanged — DEAD-INIT LOCK!")
        ok = False
    if abs(bp_final - bias_proj_init) < 1e-6:
        print("FAIL: bias_proj unchanged — pair-bias path dead")
        ok = False
    if abs(dp_final - delta_proj_init) < 1e-6:
        print("FAIL: delta_proj unchanged — bidirectional path dead")
        ok = False
    if pair_repr_diff < 0.01:
        print("WARN: pair_repr barely changed after K-step")

    if ok:
        print("T7 SANITY PASS — no dead-init, all 6 modules learning")
    else:
        print("T7 SANITY FAIL — investigate before HPC submission")


if __name__ == '__main__':
    main()
