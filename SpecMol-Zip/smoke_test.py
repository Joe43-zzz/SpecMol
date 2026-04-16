import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from utils_fp_downstream import TestbedDataset
from model_gnn_pre_v2 import LH_Direct
from main_pretrain import ours_loss
import time

def smoke_test():
    device = 'cpu'  # 先在CPU上测

    # === 加载数据 ===
    print("Loading BACE dataset...")
    data = TestbedDataset(root='down_task', dataset='all', task='bace', type='tri')
    print(f"Total molecules: {len(data)}")

    # 检查第一个样本的属性
    sample = data[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"  x shape: {sample.x.shape}")
    print(f"  edge_index shape: {sample.edge_index.shape}")
    if hasattr(sample, 'pos'):
        print(f"  pos shape: {sample.pos.shape}")
    if hasattr(sample, 'atom_type'):
        print(f"  atom_type shape: {sample.atom_type.shape}")

    # === 构造小batch ===
    loader = DataLoader(data, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    print(f"\nBatch info:")
    print(f"  total_nodes: {batch.x.size(0)}")
    print(f"  num_graphs: {batch.num_graphs}")
    print(f"  pos shape: {batch.pos.shape if hasattr(batch, 'pos') else 'None'}")

    # === 实例화모델（带pair repr） ===
    print("\nInstantiating model with num_heads=8...")
    model = LH_Direct(
        in_dim=93,
        hid_dim=128,  # 小一点省内存
        K=5,          # 少一点层数加速
        dprate=0.5,
        dropout=0.0,
        is_bns=False,
        act_fn='relu',
        type='tri',
        num_heads=8,
        num_atom_types=64,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # === 跑3次forward+backward ===
    print("\nRunning 3 forward+backward iterations...")
    model.train()
    for step in range(3):
        start = time.time()
        low_x, high_x, spec_x, x_fp = model(batch, device)

        loss = ours_loss(low_x, high_x, spec_x, x_fp, alpha=1.0)

        optimizer.zero_grad()
        loss.backward()

        # 检查梯度流向pair repr相关参数
        pair_init_grad = model.pair_repr_init.gbf.means.weight.grad
        gate_grad = model.encoder.prop1.gate_proj[0].weight.grad
        pair_update_grad = model.encoder.prop1.pair_update.q_proj.weight.grad

        print(f"Step {step}: loss={loss.item():.4f}, time={time.time()-start:.2f}s")
        print(f"  gbf.means.grad norm: {pair_init_grad.norm().item():.10e}")
        print(f"  gate_proj.grad norm: {gate_grad.norm().item():.10e}")
        print(f"  pair_update.q_proj.grad norm: {pair_update_grad.norm().item():.10e}")
        print(f"  pair_update.q_proj.grad abs max: {pair_update_grad.abs().max().item():.10e}")
        print(f"  pair_update.q_proj.grad is all zero: {(pair_update_grad == 0).all().item()}")

        optimizer.step()

    # === 对比测试：无pair repr ===
    print("\n--- Control: model without pair repr (num_heads=None) ---")
    model_nopair = LH_Direct(
        in_dim=93,
        hid_dim=128,
        K=5,
        dprate=0.5,
        dropout=0.0,
        is_bns=False,
        act_fn='relu',
        type='tri',
        num_heads=None,
    ).to(device)

    optimizer2 = optim.Adam(model_nopair.parameters(), lr=1e-4)

    for step in range(3):
        start = time.time()
        low_x, high_x, spec_x, x_fp = model_nopair(batch, device)
        loss = ours_loss(low_x, high_x, spec_x, x_fp, alpha=1.0)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        print(f"Step {step} (no pair): loss={loss.item():.4f}, time={time.time()-start:.2f}s")

    print("\nSmoke test passed!")

if __name__ == '__main__':
    smoke_test()
