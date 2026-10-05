"""Micro sanity: 3 epochs, batch=64, CPU. Verify T6 training loop works."""

import torch
import torch.optim as optim
from torch_geometric.data import InMemoryDataset, DataLoader


class PreloadedDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(root=None)
        self.data, self.slices = torch.load(pt_path, weights_only=False)
    def _download(self): pass
    def _process(self): pass


from model_gnn_pre_v2 import LH_Direct_V2
from train_utils import ours_loss, set_seed


def main():
    set_seed(9)

    device = 'cpu'
    data = PreloadedDataset('down_task_v2/processed/bace_all.pt')
    print(f"Dataset: {len(data)} molecules")

    model = LH_Direct_V2(
        in_dim=93, hid_dim=512, K=10, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn='relu', type='tri', pair_dim=64, t6=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)

    bias_init = model.pair_to_edge_weight.mlp[2].bias.item()
    print(f"Bias init: {bias_init:.4f}")
    print(f"Q norm init: {model.encoder.prop1.node_to_pair.q_proj.weight.norm().item():.6f}")

    # Only train on first 64 molecules for speed
    loader = DataLoader(data[:64], batch_size=32, shuffle=True)

    losses = []
    for epoch in range(3):
        model.train()
        total_loss, total_num = 0.0, 0
        for batch in loader:
            low_x, high_x, spec_x, x_fp = model(batch, device)
            loss = ours_loss(low_x, high_x, spec_x, x_fp)
            assert not torch.isnan(loss), f"NaN at epoch {epoch}!"
            total_num += batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg = total_loss / total_num
        losses.append(avg)

        bias = model.pair_to_edge_weight.mlp[2].bias.item()
        q_norm = model.encoder.prop1.node_to_pair.q_proj.weight.norm().item()
        k_norm = model.encoder.prop1.node_to_pair.k_proj.weight.norm().item()
        print(f"Epoch {epoch}: loss={avg:.6f}  bias={bias:.4f}  Q={q_norm:.6f}  K={k_norm:.6f}")

    print(f"\nLoss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"Decreased: {losses[-1] < losses[0]}")
    print(f"No NaN: True")
    print(f"Q/K learned from zero: {q_norm > 1e-8}")
    print("MICRO SANITY PASS" if losses[-1] < losses[0] else "MICRO SANITY FAIL")


if __name__ == '__main__':
    main()
