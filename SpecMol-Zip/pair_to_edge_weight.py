"""T4: PairToEdgeWeight module.

Maps all-pairs pair_repr to scalar edge weights on chemical bond edges,
for use as dynamic Laplacian weights in V2 ChebNet II.
"""

import torch
import torch.nn as nn


class PairToEdgeWeight(nn.Module):
    def __init__(self, pair_dim=64, hidden_dim=32, nullify=False, bias_init=5.0):
        super().__init__()
        self.nullify = nullify
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.mlp[2].bias, bias_init)

    def forward(self, pair_repr_edge, all_pairs_edge_index, chem_edge_index, batch):
        """
        Args:
            pair_repr_edge:        [E_allpairs, pair_dim]
            all_pairs_edge_index:  [2, E_allpairs]  (global PyG node indices)
            chem_edge_index:       [2, E_chembond]   (global PyG node indices)
            batch:                 [N_total]
        Returns:
            edge_weight: [E_chembond] in (0, 1)
        """
        N = batch.size(0)
        device = batch.device
        E_chem = chem_edge_index.size(1)

        # Separate self-loops (not in all-pairs i!=j index) from real edges
        not_selfloop = chem_edge_index[0] != chem_edge_index[1]

        # Pack (src, dst) into unique int key = src * N + dst
        # Use searchsorted instead of N*N table to avoid OOM on large batches
        all_enc = all_pairs_edge_index[0] * N + all_pairs_edge_index[1]
        all_enc_sorted, sort_perm = all_enc.sort()

        # Only look up non-self-loop edges
        chem_src = chem_edge_index[0][not_selfloop]
        chem_dst = chem_edge_index[1][not_selfloop]
        chem_ij = chem_src * N + chem_dst
        chem_ji = chem_dst * N + chem_src
        pos_ij = torch.searchsorted(all_enc_sorted, chem_ij)
        pos_ji = torch.searchsorted(all_enc_sorted, chem_ji)
        # Guard: searchsorted can return out-of-bounds or wrong index
        pos_ij = pos_ij.clamp(max=all_enc_sorted.size(0) - 1)
        pos_ji = pos_ji.clamp(max=all_enc_sorted.size(0) - 1)
        if not (all_enc_sorted[pos_ij] == chem_ij).all():
            raise ValueError("Some chem bond edges not found in all-pairs edge index")
        if not (all_enc_sorted[pos_ji] == chem_ji).all():
            raise ValueError("Some chem bond edges (reverse) not found in all-pairs edge index")
        idx_ij = sort_perm[pos_ij]
        idx_ji = sort_perm[pos_ji]

        # Raw MLP scores for both directions
        raw_ij = self.mlp(pair_repr_edge[idx_ij])  # [E_nonloop, 1]
        raw_ji = self.mlp(pair_repr_edge[idx_ji])  # [E_nonloop, 1]

        # Symmetrize before sigmoid (get_laplacian 'sym' does NOT auto-symmetrize)
        raw_avg = (raw_ij + raw_ji) / 2.0
        w_nonloop = torch.sigmoid(raw_avg).squeeze(-1)  # [E_nonloop]

        # Assemble full edge_weight: self-loops get weight 1.0
        edge_weight = torch.ones(E_chem, device=device)
        edge_weight[not_selfloop] = w_nonloop

        if self.nullify:
            return torch.ones_like(edge_weight)

        return edge_weight


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    def make_all_pairs_edge_index(num_nodes, offset=0):
        """All (i,j) pairs with i!=j, global indices starting at offset."""
        idx = torch.arange(num_nodes, dtype=torch.long)
        src = idx.repeat_interleave(num_nodes - 1) + offset
        # For each node i, dst = all nodes except i
        dst_list = []
        for i in range(num_nodes):
            others = torch.cat([idx[:i], idx[i+1:]])
            dst_list.append(others + offset)
        dst = torch.cat(dst_list)
        return torch.stack([src, dst], dim=0)

    def make_deterministic_chem_edges(num_nodes, num_bonds, offset=0):
        """Deterministic undirected chem edges: first num_bonds pairs from (0,1),(0,2),..."""
        src, dst = [], []
        count = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if count >= num_bonds:
                    break
                src.extend([i + offset, j + offset])
                dst.extend([j + offset, i + offset])
                count += 1
            if count >= num_bonds:
                break
        return torch.tensor([src, dst], dtype=torch.long)

    model = PairToEdgeWeight(pair_dim=64)
    pair_dim = 64

    # -------------------------------------------------------------------
    # T1: Single molecule forward
    # -------------------------------------------------------------------
    print("=" * 60)
    print("T1: Single molecule forward")
    N1 = 10
    E_all1 = N1 * (N1 - 1)  # 90

    all_pairs_ei = make_all_pairs_edge_index(N1)
    chem_ei = make_deterministic_chem_edges(N1, 15)  # 15 undirected = 30 directed
    E_chem = chem_ei.size(1)  # 30
    batch = torch.zeros(N1, dtype=torch.long)
    pair_repr = torch.randn(E_all1, pair_dim)

    w = model(pair_repr, all_pairs_ei, chem_ei, batch)

    assert w.shape == (E_chem,), f"Shape mismatch: {w.shape} vs ({E_chem},)"
    assert (w > 0).all() and (w < 1).all(), f"Range violation: min={w.min():.4f}, max={w.max():.4f}"
    assert w.mean() > 0.98, f"Mean too low: {w.mean():.4f} (expected > 0.98 with bias=+5)"
    print(f"  PASS | shape={tuple(w.shape)}, range=[{w.min():.4f}, {w.max():.4f}], mean={w.mean():.4f}")

    # -------------------------------------------------------------------
    # T2: Numerical correctness (manual check for one edge)
    # -------------------------------------------------------------------
    print("=" * 60)
    print("T2: Numerical correctness")
    # Pick chem edge k=0: (0, 1)
    i, j = chem_ei[0, 0].item(), chem_ei[1, 0].item()
    N = N1
    all_enc = all_pairs_ei[0] * N + all_pairs_ei[1]
    all_enc_sorted, sort_perm = all_enc.sort()
    row_ij = sort_perm[torch.searchsorted(all_enc_sorted, torch.tensor(i * N + j))].item()
    row_ji = sort_perm[torch.searchsorted(all_enc_sorted, torch.tensor(j * N + i))].item()
    with torch.no_grad():
        raw_ij = model.mlp(pair_repr[row_ij:row_ij+1])
        raw_ji = model.mlp(pair_repr[row_ji:row_ji+1])
        expected = torch.sigmoid((raw_ij + raw_ji) / 2.0).item()
    actual = w[0].item()
    assert abs(actual - expected) < 1e-5, f"Mismatch: actual={actual:.6f} vs expected={expected:.6f}"
    print(f"  PASS | edge ({i},{j}): actual={actual:.6f}, expected={expected:.6f}")

    # -------------------------------------------------------------------
    # T3: PyG batch consistency (no cross-molecule leakage)
    # -------------------------------------------------------------------
    print("=" * 60)
    print("T3: PyG batch consistency")
    N_a, N_b = 5, 8
    E_all_a = N_a * (N_a - 1)  # 20
    E_all_b = N_b * (N_b - 1)  # 56

    all_pairs_a = make_all_pairs_edge_index(N_a, offset=0)
    all_pairs_b = make_all_pairs_edge_index(N_b, offset=N_a)
    all_pairs_bat = torch.cat([all_pairs_a, all_pairs_b], dim=1)

    chem_a = make_deterministic_chem_edges(N_a, 5, offset=0)   # 10 directed
    chem_b = make_deterministic_chem_edges(N_b, 7, offset=N_a)  # 14 directed
    chem_bat = torch.cat([chem_a, chem_b], dim=1)
    expected_total = chem_a.size(1) + chem_b.size(1)

    batch_bat = torch.cat([torch.zeros(N_a, dtype=torch.long),
                           torch.ones(N_b, dtype=torch.long)])
    pair_repr_bat = torch.randn(E_all_a + E_all_b, pair_dim)

    w_bat = model(pair_repr_bat, all_pairs_bat, chem_bat, batch_bat)
    assert w_bat.shape == (expected_total,), f"Shape: {w_bat.shape} vs ({expected_total},)"

    # Verify mol A chem edges index into mol A pair_repr only
    N_total = N_a + N_b
    all_enc_check = all_pairs_bat[0] * N_total + all_pairs_bat[1]
    all_enc_sorted_check, sort_perm_check = all_enc_check.sort()

    for k in range(chem_a.size(1)):
        ci, cj = chem_a[0, k].item(), chem_a[1, k].item()
        assert ci < N_a and cj < N_a, f"Mol A edge ({ci},{cj}) out of range"
        pidx = sort_perm_check[torch.searchsorted(all_enc_sorted_check, torch.tensor(ci * N_total + cj))].item()
        assert pidx < E_all_a, f"Mol A edge ({ci},{cj}) -> pair idx {pidx} >= {E_all_a}"

    for k in range(chem_b.size(1)):
        ci, cj = chem_b[0, k].item(), chem_b[1, k].item()
        assert ci >= N_a and cj >= N_a, f"Mol B edge ({ci},{cj}) out of range"
        pidx = sort_perm_check[torch.searchsorted(all_enc_sorted_check, torch.tensor(ci * N_total + cj))].item()
        assert pidx >= E_all_a, f"Mol B edge ({ci},{cj}) -> pair idx {pidx} < {E_all_a}"

    print(f"  PASS | shape={tuple(w_bat.shape)}, no cross-molecule leakage")

    # -------------------------------------------------------------------
    # T4: Backward + grad
    # -------------------------------------------------------------------
    print("=" * 60)
    print("T4: Backward + grad")
    model.zero_grad()
    pair_repr_grad = torch.randn(E_all1, pair_dim, requires_grad=True)
    w_grad = model(pair_repr_grad, all_pairs_ei, chem_ei, batch)
    w_grad.sum().backward()

    all_have_grad = all(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.parameters())
    assert all_have_grad, "Some MLP parameters have no gradient"
    assert pair_repr_grad.grad is not None, "pair_repr_edge has no gradient"
    print("  PASS | all MLP params and input have gradients")

    # -------------------------------------------------------------------
    # T5: Bias=+5 mean check + symmetry verification
    # -------------------------------------------------------------------
    print("=" * 60)
    print("T5: Bias=+5 mean + symmetry")
    fresh_model = PairToEdgeWeight(pair_dim=64)
    pair_repr_fresh = torch.randn(E_all1, pair_dim)
    # Symmetric chem edges: (0,1),(1,0),(2,3),(3,2)
    sym_chem = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    w_fresh = fresh_model(pair_repr_fresh, all_pairs_ei, sym_chem, batch)

    mean_val = w_fresh.mean().item()
    assert mean_val > 0.98, f"Mean {mean_val:.4f} not > 0.98"
    # Symmetry: w(0,1) == w(1,0), w(2,3) == w(3,2)
    assert abs(w_fresh[0].item() - w_fresh[1].item()) < 1e-6, "w(0,1) != w(1,0)"
    assert abs(w_fresh[2].item() - w_fresh[3].item()) < 1e-6, "w(2,3) != w(3,2)"
    print(f"  PASS | mean={mean_val:.4f}, w(0,1)==w(1,0), w(2,3)==w(3,2)")

    print("=" * 60)
    print("All 5 tests passed.")
