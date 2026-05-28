"""Tox21 12-task BCE smoke test.

Run this BEFORE sbatching real Tox21 training. It builds a small synthetic
batch with Tox21's exact shape (12 binary labels, ~30% masked-out entries
encoded as 999) and verifies:

1. The downstream probe path (classification_probe_batch + the per-task
   weighted BCE loop in train_classification_probe_epoch) runs without
   shape / NaN errors at n_task=12.
2. Gradients flow into the LogReg head.
3. The loss is finite at initialization.

Runs in ~10 seconds on CPU. Skips Uni-Mol pair injection (uses 2D-only
LH_Direct, the V0 baseline path), since the BCE / multi-label path is
exercised independently of which encoder is used.

Usage:
    python test_tox21_bce_smoke.py
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader

from model_gnn_pre import LH_Direct, LogReg
from train_utils import classification_probe_batch, train_classification_probe_epoch


N_TASKS = 12       # Tox21
N_MOLS = 50
N_ATOMS_PER_MOL = 12
ATOM_FEAT_DIM = 93
EDGE_FEAT_DIM = 11
FP_DIM = 1489
HID_DIM = 64       # smaller than the 512 prod default so the smoke test stays fast
MISSING_LABEL_FRAC = 0.30  # ~30% of (mol, task) cells masked, per Tox21 sparsity


def build_synthetic_batch():
    """Build a list of N_MOLS PyG Data objects with 12-label classification shape."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    data_list = []
    for _ in range(N_MOLS):
        n = N_ATOMS_PER_MOL
        x = torch.randn(n, ATOM_FEAT_DIM)
        # Star graph (atom 0 connected to all others) + reverse edges for symmetry
        src = torch.cat([torch.zeros(n - 1, dtype=torch.long),
                         torch.arange(1, n)])
        dst = torch.cat([torch.arange(1, n),
                         torch.zeros(n - 1, dtype=torch.long)])
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.zeros(edge_index.size(1), EDGE_FEAT_DIM)
        # 12-label binary y with ~30% masked entries (encoded as 999.)
        y = torch.randint(0, 2, (N_TASKS,)).float()
        mask_drop = torch.rand(N_TASKS) < MISSING_LABEL_FRAC
        y[mask_drop] = 999.0
        w = (y != 999.0).float()  # weight = 1 if label valid else 0
        fps = torch.randn(FP_DIM)

        data_list.append(Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            y=y, w=w, fps=fps,
        ))
    return data_list


def run_smoke():
    device = "cpu"

    print(f"[smoke] building synthetic Tox21-shape batch: "
          f"{N_MOLS} mols, {N_TASKS} tasks, ~{int(100*MISSING_LABEL_FRAC)}% masked")
    data_list = build_synthetic_batch()
    loader = DataLoader(data_list, batch_size=16, shuffle=False)

    print(f"[smoke] instantiating LH_Direct (V0 path, hid_dim={HID_DIM}, K=4)")
    model = LH_Direct(
        in_dim=ATOM_FEAT_DIM, hid_dim=HID_DIM, K=4, dprate=0.0, dropout=0.0,
        is_bns=False, act_fn="relu", type="tri",
    ).to(device)
    model.eval()

    logreg = LogReg(hid_dim=HID_DIM, n_classes=N_TASKS).to(device)
    optimizer = torch.optim.Adam(logreg.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")  # ignored by train_classification_probe_epoch

    print("[smoke] step 1: classification_probe_batch shape check")
    for batch in loader:
        logits, y, mask = classification_probe_batch(
            batch, model, logreg, device, n_task=N_TASKS, return_shaped=True, fp_only=False,
        )
        assert logits.shape == (batch.num_graphs, N_TASKS), \
            f"logits shape {tuple(logits.shape)} != ({batch.num_graphs}, {N_TASKS})"
        assert y.shape == logits.shape, f"y shape mismatch: {tuple(y.shape)}"
        assert mask.shape == logits.shape, f"mask shape mismatch: {tuple(mask.shape)}"
        assert torch.isfinite(logits).all(), "non-finite logits at init"
        print(f"  ok: logits={tuple(logits.shape)}  valid={int(mask.sum())}/{mask.numel()}")
        break

    print("[smoke] step 2: train_classification_probe_epoch (1 epoch)")
    initial_w = logreg.fc.weight.clone().detach()
    train_classification_probe_epoch(
        loader, model, logreg, optimizer, loss_fn, device,
        n_task=N_TASKS, fp_only=False,
    )
    delta_w = (logreg.fc.weight.detach() - initial_w).abs().sum().item()
    assert delta_w > 0, f"LogReg weights did not change after 1 epoch (delta={delta_w})"
    print(f"  ok: LogReg weight delta after 1 epoch = {delta_w:.6f}")

    print("[smoke] step 3: 5 epochs, loss should be finite")
    for ep in range(5):
        train_classification_probe_epoch(
            loader, model, logreg, optimizer, loss_fn, device,
            n_task=N_TASKS, fp_only=False,
        )
    # one final batch to check loss finiteness
    logreg.eval()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    for batch in loader:
        with torch.no_grad():
            logits, y, mask = classification_probe_batch(
                batch, model, logreg, device, n_task=N_TASKS, return_shaped=True, fp_only=False,
            )
            per_elem = bce_none(logits, y.float()) * mask.float()
            denom = mask.sum().clamp(min=1.0)
            loss = per_elem.sum() / denom
            assert torch.isfinite(loss), f"non-finite loss after 5 epochs: {loss.item()}"
            print(f"  ok: final loss = {loss.item():.4f}")
        break

    print("\n[smoke] ALL OK -- safe to sbatch Tox21 training.")


if __name__ == "__main__":
    run_smoke()
