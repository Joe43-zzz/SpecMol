"""Sanity check: verify FreeSolv data loads + forward/backward works."""

import sys
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)


def load_split(root, split):
    from torch_geometric.data import InMemoryDataset
    import os
    path = os.path.join(root, "processed", f"freesolv_{split}.pt")
    if not os.path.exists(path):
        print(f"FAIL: {path} not found")
        return None

    class _W(InMemoryDataset):
        def __init__(self, p):
            # Use a temp root to satisfy PyG, then override data directly
            import tempfile
            tmp = tempfile.mkdtemp()
            super().__init__(root=tmp)
            self.data, self.slices = torch.load(p)
        @property
        def processed_file_names(self):
            return []
        def process(self):
            pass

    return _W(path)


def check_baseline():
    print("=" * 50)
    print("CHECK 1: Baseline data (2D-only)")
    print("=" * 50)
    for split in ["train", "valid", "test", "all"]:
        ds = load_split("down_task_freesolv_2d", split)
        if ds is None:
            return False
        print(f"  {split}: {len(ds)} graphs, x={ds[0].x.shape}, y={ds[0].y}")
    print("  OK")

    print("\nCHECK 2: Baseline forward + backward")
    from model_gnn_pre import LH_Direct
    ds = load_split("down_task_freesolv_2d", "all")
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model = LH_Direct(in_dim=93, hid_dim=64, K=5, dprate=0.5, dropout=0.0,
                       is_bns=False, act_fn="relu", type="tri")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    batch = next(iter(loader))
    low_x, high_x, spec_x, x_fp = model(batch, device)
    print(f"  Forward OK: spec_x={spec_x.shape}, x_fp={x_fp.shape}")

    cos = nn.CosineSimilarity(dim=1)
    loss = -cos(spec_x, x_fp).mean()
    loss.backward()
    print(f"  Backward OK: loss={loss.item():.4f}")
    return True


def check_v2():
    print("\n" + "=" * 50)
    print("CHECK 3: V2-T5 data (with pair_repr)")
    print("=" * 50)
    for split in ["train", "valid", "test", "all"]:
        ds = load_split("down_task_freesolv_v2", split)
        if ds is None:
            return False
        d = ds[0]
        has_pair = hasattr(d, 'pair_repr_edge') and d.pair_repr_edge is not None
        print(f"  {split}: {len(ds)} graphs, pair_repr={'YES' if has_pair else 'NO'}" +
              (f" shape={d.pair_repr_edge.shape}" if has_pair else ""))
    print("  OK")

    print("\nCHECK 4: V2-T5 forward + backward")
    from model_gnn_pre_v2 import LH_Direct_V2
    ds = load_split("down_task_freesolv_v2", "all")
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model = LH_Direct_V2(in_dim=93, hid_dim=64, K=5, dprate=0.5, dropout=0.0,
                          is_bns=False, act_fn="relu", type="tri")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    batch = next(iter(loader))
    low_x, high_x, spec_x, x_fp = model(batch, device)
    print(f"  Forward OK: spec_x={spec_x.shape}, x_fp={x_fp.shape}")

    cos = nn.CosineSimilarity(dim=1)
    loss = -cos(spec_x, x_fp).mean()
    loss.backward()
    print(f"  Backward OK: loss={loss.item():.4f}")

    # Check regression downstream head
    from model_gnn_pre import LogReg
    logreg = LogReg(hid_dim=64, n_classes=1).to(device)
    with torch.no_grad():
        _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
    embed = torch.cat([spec_x, x_fp], dim=1)
    preds = logreg(embed)
    mse = nn.MSELoss()(preds, y.reshape(-1, 1))
    mse.backward()
    print(f"  Regression head OK: preds={preds.shape}, mse={mse.item():.4f}")
    return True


if __name__ == "__main__":
    ok1 = check_baseline()
    ok2 = check_v2()
    if ok1 and ok2:
        print("\n" + "=" * 50)
        print("ALL SANITY CHECKS PASSED")
        print("=" * 50)
    else:
        print("\nSOME CHECKS FAILED")
        sys.exit(1)
