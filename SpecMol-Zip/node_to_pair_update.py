import torch
import torch.nn as nn


class NodeToPairUpdate(nn.Module):
    """
    Residual pair-repr update using Chebyshev intermediate results T_k.

    Mirrors Uni-Mol's QK^T pair update but replaces attention outputs with
    explicit Q/K projections of the ChebNet-II propagation state T_k.

    Update rule (per head):
        update[i, j, h] = sum_d( Q[i,h,d] * K[j,h,d] ) * scaling
        pair_repr        = pair_repr + update   (intra-graph only)
    """

    def __init__(self, node_dim, proj_dim=32, num_heads=8):
        super().__init__()
        self.proj_dim  = proj_dim
        self.num_heads = num_heads
        self.scaling   = proj_dim ** -0.5

        self.q_proj = nn.Linear(node_dim, proj_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(node_dim, proj_dim * num_heads, bias=False)

    def forward(self, t_k, pair_repr, batch):
        """
        Args:
            t_k       : [total_nodes, D]          Cheby state at step k
            pair_repr : [total_nodes, total_nodes, H]  current pair repr
            batch     : [total_nodes]             PyG batch vector

        Returns:
            pair_repr : [total_nodes, total_nodes, H]  updated pair repr
        """
        N = t_k.size(0)

        # --- Q / K projections -------------------------------------------
        Q = self.q_proj(t_k).view(N, self.num_heads, self.proj_dim)  # [N, H, d]
        K = self.k_proj(t_k).view(N, self.num_heads, self.proj_dim)  # [N, H, d]

        # --- outer product update ----------------------------------------
        # update[i, j, h] = sum_d Q[i,h,d] * K[j,h,d]
        update = torch.einsum('ihd,jhd->ijh', Q, K) * self.scaling   # [N, N, H]

        # --- intra-graph mask --------------------------------------------
        # mask[i, j] = True  iff i and j belong to the same graph
        mask = batch.unsqueeze(0) == batch.unsqueeze(1)               # [N, N]
        update = update * mask.unsqueeze(-1)                          # [N, N, H]

        return pair_repr + update


if __name__ == '__main__':
    torch.manual_seed(0)

    # ------------------------------------------------------------------ #
    # Test 1: single graph — basic shape and non-trivial update
    # ------------------------------------------------------------------ #
    N, D, H, proj_dim = 15, 64, 8, 32

    t_k       = torch.randn(N, D)
    pair_repr = torch.randn(N, N, H)
    batch     = torch.zeros(N, dtype=torch.long)

    model = NodeToPairUpdate(node_dim=D, proj_dim=proj_dim, num_heads=H)
    out = model(t_k, pair_repr, batch)

    assert out.shape == (N, N, H), f'Shape mismatch: {out.shape}'
    assert not torch.allclose(out, pair_repr), 'pair_repr was not updated'

    print('Test 1 passed: shape correct and update is non-trivial')

    # ------------------------------------------------------------------ #
    # Test 2: PyG batching — cross-graph isolation
    # ------------------------------------------------------------------ #
    N0, N1 = 10, 5
    total   = N0 + N1
    D2, H2, proj_dim2 = 64, 8, 32

    t_k2       = torch.randn(total, D2)
    pair_repr2 = torch.zeros(total, total, H2)          # all-zero baseline
    batch2     = torch.tensor([0]*N0 + [1]*N1)

    model2 = NodeToPairUpdate(node_dim=D2, proj_dim=proj_dim2, num_heads=H2)
    out2   = model2(t_k2, pair_repr2, batch2)

    # Cross-graph position: node 0 (graph 0) vs node N0 (graph 1) → must stay 0
    cross = out2[0, N0, :]
    assert torch.allclose(cross, torch.zeros(H2)), \
        f'Cross-graph pair was updated: {cross}'

    # Intra-graph position: node 0 vs node 5 (both in graph 0) → must change
    intra = out2[0, 5, :]
    assert not torch.allclose(intra, torch.zeros(H2)), \
        'Intra-graph pair was NOT updated'

    print('Test 2 passed: cross-graph positions unchanged, intra-graph positions updated')

    # ------------------------------------------------------------------ #
    # Test 3: gradient flow
    # ------------------------------------------------------------------ #
    t_k3       = torch.randn(N, D, requires_grad=True)
    pair_repr3 = torch.randn(N, N, H)
    batch3     = torch.zeros(N, dtype=torch.long)

    out3 = model(t_k3, pair_repr3, batch3)
    out3.sum().backward()

    assert t_k3.grad is not None,              'No gradient on t_k'
    assert model.q_proj.weight.grad is not None, 'No gradient on q_proj.weight'

    print('Test 3 passed: gradients flow through t_k and q_proj.weight')

    print('All tests passed')
