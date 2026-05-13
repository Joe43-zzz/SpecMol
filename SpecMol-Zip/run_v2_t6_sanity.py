"""Step 4: V2-T6 single-cell BACE training sanity check.

50 epochs, seed=9, batch_size=512 (match T5 baseline), GPU.
Reports: loss curve, bias delta, Q/K/out_proj norms, pair_repr diff, GPU stats.
"""

import argparse
import gc
import os
import sys
import time

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
    args = parser.parse_args()

    device = args.device
    seed = 9
    epochs = args.epochs
    batch_size = 512   # match T5 baseline
    lr = 0.0001
    hid_dim = 512
    K = 10
    path = 'down_task_v2'

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

    print("\nCreating T6 model...")
    model = LH_Direct_V2(
        in_dim=93, hid_dim=hid_dim, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn='relu', type='tri', pair_dim=64, t6=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    # Record inits
    bias_init = model.pair_to_edge_weight.mlp[2].bias.item()
    q_norm_init = model.encoder.prop1.node_to_pair.q_proj.weight.norm().item()
    k_norm_init = model.encoder.prop1.node_to_pair.k_proj.weight.norm().item()
    out_norm_init = model.encoder.prop1.node_to_pair.out_proj.weight.norm().item()

    print(f"  Bias init:     {bias_init:.4f}")
    print(f"  Q norm init:   {q_norm_init:.6f}")
    print(f"  K norm init:   {k_norm_init:.6f}")
    print(f"  out_proj init: {out_norm_init:.6f} (should be 0)")

    loss_history = []
    epoch_times = []

    print(f"\n{'Epoch':>5} | {'Loss':>10} | {'Bias':>8} | {'Q norm':>10} | {'K norm':>10} | {'out_proj':>10} | {'Time(s)':>7}")
    print("-" * 80)

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
        q_norm = model.encoder.prop1.node_to_pair.q_proj.weight.norm().item()
        k_norm = model.encoder.prop1.node_to_pair.k_proj.weight.norm().item()
        out_norm = model.encoder.prop1.node_to_pair.out_proj.weight.norm().item()

        if epoch < 5 or epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{epoch:>5} | {avg_loss:>10.6f} | {bias_now:>8.4f} | {q_norm:>10.6f} | {k_norm:>10.6f} | {out_norm:>10.6f} | {dt:>7.1f}")

    # --- pair_repr diff measurement ---
    model.eval()
    with torch.no_grad():
        # Forward the same 32 molecules and observe pair_repr evolution
        sample_batch2 = next(iter(DataLoader(data[:32], batch_size=32, shuffle=False)))
        feat = sample_batch2.x.to(device)
        edge_index = sample_batch2.edge_index.to(device)
        batch_vec = sample_batch2.batch.to(device)
        pr = sample_batch2.pair_repr_edge.to(device).clone()
        pei = sample_batch2.pair_edge_index.to(device)

        ew = model.pair_to_edge_weight(pr, pei, edge_index, batch_vec)
        # Run highpass prop with T6 to get final pair_repr
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
    print("STEP 4 SANITY CHECK SUMMARY")
    print("=" * 60)

    bias_final = model.pair_to_edge_weight.mlp[2].bias.item()
    q_final = model.encoder.prop1.node_to_pair.q_proj.weight.norm().item()
    k_final = model.encoder.prop1.node_to_pair.k_proj.weight.norm().item()
    out_final = model.encoder.prop1.node_to_pair.out_proj.weight.norm().item()

    print(f"\nLoss curve:")
    print(f"  Epoch  0: {loss_history[0]:.6f}")
    for i in [4, 9, 19, 29, 39, 49]:
        if i < len(loss_history):
            print(f"  Epoch {i:>2}: {loss_history[i]:.6f}")
    print(f"  Decreased: {loss_history[-1] < loss_history[0]}")
    print(f"  Any NaN: {any(np.isnan(l) for l in loss_history)}")

    decreasing = sum(1 for i in range(1, len(loss_history)) if loss_history[i] < loss_history[i-1])
    print(f"  Decreasing steps: {decreasing}/{epochs-1}")

    print(f"\nBias: {bias_init:.4f} -> {bias_final:.4f} (delta={bias_final - bias_init:+.4f})")
    print(f"Q norm: {q_norm_init:.6f} -> {q_final:.6f} (delta={q_final - q_norm_init:+.6f})")
    print(f"K norm: {k_norm_init:.6f} -> {k_final:.6f} (delta={k_final - k_norm_init:+.6f})")
    print(f"out_proj norm: {out_norm_init:.6f} -> {out_final:.6f} (delta={out_final - out_norm_init:+.6f})")

    print(f"\npair_repr diff (32 mols, post-training K-step vs initial): {pair_repr_diff:.4f}")

    gpu_peak = get_gpu_mem_mb()
    avg_time = np.mean(epoch_times)
    print(f"\nGPU peak memory: {gpu_peak:.0f} MB")
    print(f"Avg epoch time: {avg_time:.1f}s")
    print(f"Total time: {sum(epoch_times):.0f}s")

    # Verdicts
    print("\n--- VERDICTS ---")
    ok = True
    if loss_history[-1] >= loss_history[0]:
        print("FAIL: loss did not decrease")
        ok = False
    if abs(bias_final - bias_init) < 0.001:
        print("WARN: bias barely moved (delta < 0.001)")
    if abs(q_final - q_norm_init) < 1e-6 and abs(k_final - k_norm_init) < 1e-6:
        print("FAIL: Q/K norms unchanged — dead path!")
        ok = False
    if out_final < 1e-8:
        print("FAIL: out_proj still zero — T6 not learning")
        ok = False
    if pair_repr_diff < 0.01:
        print("WARN: pair_repr barely changed after K-step")

    if ok:
        print("STEP 4 PASS")
    else:
        print("STEP 4 FAIL — investigate before proceeding")


if __name__ == '__main__':
    main()
