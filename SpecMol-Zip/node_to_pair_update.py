"""T6: Sparse NodeToPairUpdate module.

Residual pair-repr update using Chebyshev intermediate results T_k.
Sparse version: operates on pair_edge_index positions only (O(E) memory),
mathematically equivalent to dense QK^T on all-pairs within each graph.

Update rule:
    qk[e] = sum_d( Q[src_e, d] * K[dst_e, d] ) * scaling   (scalar per edge)
    delta[e] = out_proj(qk[e])                               (project to pair_dim)
    pair_repr_edge = pair_repr_edge + delta
"""

import torch
import torch.nn as nn


class NodeToPairUpdate(nn.Module):

    def __init__(self, node_dim, pair_dim=64, proj_dim=32):
        super().__init__()
        self.proj_dim = proj_dim
        self.pair_dim = pair_dim
        self.scaling = proj_dim ** -0.5

        self.q_proj = nn.Linear(node_dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(node_dim, proj_dim, bias=False)
        self.out_proj = nn.Linear(1, pair_dim, bias=False)

        # Zero-init out_proj so delta=0 at init → T6 starts equivalent to T5.
        # Q/K use default (Kaiming) init so qk is nonzero and out_proj gets
        # gradient from epoch 1; Q/K get gradient once out_proj becomes nonzero.
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, t_k, pair_repr_edge, pair_edge_index):
        """
        Args:
            t_k             : [N_total, node_dim]   Cheby state at step k
            pair_repr_edge  : [E_allpairs, pair_dim] current pair repr (mutable)
            pair_edge_index : [2, E_allpairs]        all-pairs edge index

        Returns:
            pair_repr_edge  : [E_allpairs, pair_dim] updated pair repr
        """
        q = self.q_proj(t_k)                                    # [N, proj_dim]
        k = self.k_proj(t_k)                                    # [N, proj_dim]

        src, dst = pair_edge_index
        qk = (q[src] * k[dst]).sum(-1, keepdim=True) * self.scaling  # [E, 1]
        delta = self.out_proj(qk)                                # [E, pair_dim]

        return pair_repr_edge + delta


if __name__ == '__main__':
    torch.manual_seed(0)

    def make_all_pairs_edge_index(num_nodes, offset=0):
        idx = torch.arange(num_nodes, dtype=torch.long)
        src = idx.repeat_interleave(num_nodes - 1) + offset
        dst_list = []
        for i in range(num_nodes):
            others = torch.cat([idx[:i], idx[i+1:]])
            dst_list.append(others + offset)
        dst = torch.cat(dst_list)
        return torch.stack([src, dst], dim=0)

    # ------------------------------------------------------------------ #
    # Test 1: single graph — basic shape and non-trivial update
    # ------------------------------------------------------------------ #
    N, D, pair_dim, proj_dim = 15, 64, 64, 32

    t_k = torch.randn(N, D)
    pair_ei = make_all_pairs_edge_index(N)
    E = pair_ei.size(1)
    pair_repr = torch.randn(E, pair_dim)

    model = NodeToPairUpdate(node_dim=D, pair_dim=pair_dim, proj_dim=proj_dim)
    # Q/K have default (Kaiming) init, but out_proj is zero → delta=0 at init
    # Break out_proj zero-init to get nonzero update for shape test
    nn.init.normal_(model.out_proj.weight, std=0.02)

    out = model(t_k, pair_repr, pair_ei)
    assert out.shape == (E, pair_dim), f'Shape mismatch: {out.shape}'
    assert not torch.allclose(out, pair_repr), 'pair_repr was not updated'
    print('Test 1 passed: shape correct and update is non-trivial')

    # ------------------------------------------------------------------ #
    # Test 2: zero-init out_proj → no update (T6 ≡ T5 sanity)
    # ------------------------------------------------------------------ #
    model_zero = NodeToPairUpdate(node_dim=D, pair_dim=pair_dim, proj_dim=proj_dim)
    # out_proj is zero-init by default → delta = out_proj(qk) = 0
    out_zero = model_zero(t_k, pair_repr, pair_ei)
    assert torch.allclose(out_zero, pair_repr), \
        f'Zero-init out_proj should produce no update, diff={(out_zero - pair_repr).abs().max()}'
    print('Test 2 passed: zero-init out_proj produces no update')

    # ------------------------------------------------------------------ #
    # Test 3: batched (2 molecules) — no cross-graph leakage
    # ------------------------------------------------------------------ #
    N0, N1 = 10, 5
    pair_ei_a = make_all_pairs_edge_index(N0, offset=0)
    pair_ei_b = make_all_pairs_edge_index(N1, offset=N0)
    pair_ei_bat = torch.cat([pair_ei_a, pair_ei_b], dim=1)
    E_bat = pair_ei_bat.size(1)

    t_k_bat = torch.randn(N0 + N1, D)
    pair_repr_bat = torch.randn(E_bat, pair_dim)

    out_bat = model(t_k_bat, pair_repr_bat, pair_ei_bat)
    assert out_bat.shape == (E_bat, pair_dim), f'Shape: {out_bat.shape}'
    # Cross-graph isolation: pair_edge_index only has intra-graph edges,
    # so cross-graph pairs are never indexed → no leakage by construction.
    print('Test 3 passed: batched forward works, cross-graph isolation by construction')

    # ------------------------------------------------------------------ #
    # Test 4: gradient flow
    # ------------------------------------------------------------------ #
    model.zero_grad()
    t_k_g = torch.randn(N, D, requires_grad=True)
    pair_repr_g = torch.randn(E, pair_dim, requires_grad=True)
    out_g = model(t_k_g, pair_repr_g, pair_ei)
    out_g.sum().backward()

    assert t_k_g.grad is not None, 'No gradient on t_k'
    assert pair_repr_g.grad is not None, 'No gradient on pair_repr_edge'
    assert model.q_proj.weight.grad is not None, 'No gradient on q_proj'
    assert model.out_proj.weight.grad is not None, 'No gradient on out_proj'
    print('Test 4 passed: gradients flow through all paths')

    print('All tests passed')
