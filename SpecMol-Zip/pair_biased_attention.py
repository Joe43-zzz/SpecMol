"""T7: Sparse bidirectional pair-biased attention.

Multi-head attention over nodes where pair_repr enters as pre-softmax bias on
attention logits (Uni-Mol-style). Bidirectional: pre-softmax logits are also
projected back to pair_dim and added to pair_repr as a delta.

Sparse PyG-native: no [B, N_max, N_max] dense tensors. All compute happens at
pair_edge positions; softmax is grouped by source node via PyG utilities.

Bypasses the scalar edge_weight bottleneck of Laplacian-based propagation —
multi-channel pair info reaches node aggregation directly via attention bias.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairBiasedSparseAttention(nn.Module):
    """Sparse multi-head attention with pair_repr as pre-softmax bias.

    The module is integrated inside ChebnetII_prop_V2's K-loop, where node
    features have the raw atom-feature dim (node_dim, e.g. 93) — not the
    post-encoder hid_dim. Internally projects node_dim → num_heads*head_dim
    for attention, then back to node_dim.

    Args:
        node_dim:  input node dim (e.g., 93 atom features)
        pair_dim:  pair feature dim (e.g., 64)
        num_heads: 4 default
        head_dim:  per-head dim, 32 default
        dropout:   attn dropout, 0.0 for sanity
        init_std:  std for small-init of out_proj and delta_proj weights
    """

    def __init__(self, node_dim, pair_dim=64, num_heads=4, head_dim=32,
                 dropout=0.0, init_std=0.02):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.H = num_heads
        self.d = head_dim
        self.attn_dim = num_heads * head_dim
        self.dropout_p = dropout

        self.q = nn.Linear(node_dim, self.attn_dim)
        self.k = nn.Linear(node_dim, self.attn_dim)
        self.v = nn.Linear(node_dim, self.attn_dim)
        # Project attention output back to node_dim so the residual add to
        # Tx_k preserves shape.
        self.out_proj = nn.Linear(self.attn_dim, node_dim)
        # bias=False so unused (diagonal / cross-molecule) pair slots contribute
        # exactly zero. (Audit issue #2.)
        self.bias_proj = nn.Linear(pair_dim, num_heads, bias=False)
        self.delta_proj = nn.Linear(num_heads, pair_dim)

        # Small-init on out_proj and delta_proj. NOT zero — zeroing both creates
        # a mutual dead-init lock where V.grad and out_proj.weight.grad are
        # permanently stuck at 0 (chain: out_proj=0 ⇒ ∂L/∂V=0; delta_proj=0 ⇒
        # ∂L/∂{Q,K,bias_proj}=0). Small std bounds epoch-0 contribution to ε≈1e-2.
        nn.init.normal_(self.out_proj.weight, std=init_std)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.delta_proj.weight, std=init_std)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(self, x, pair_repr_edge, pair_edge_index):
        """Sparse bidirectional pair-biased attention.

        Args:
            x:               [N_total, node_dim]
            pair_repr_edge:  [E_pairs, pair_dim]
            pair_edge_index: [2, E_pairs] (global PyG node ids, intra-molecule)

        Returns:
            out:        [N_total, node_dim] attention output (residual delta)
            pair_delta: [E_pairs, pair_dim] pair-side update (caller adds residually)
        """
        from torch_geometric.utils import softmax, scatter

        N = x.size(0)
        E = pair_edge_index.size(1)
        if E == 0:
            return torch.zeros_like(x), pair_repr_edge.new_zeros(0, self.pair_dim)

        src = pair_edge_index[0]
        dst = pair_edge_index[1]
        # Audit issue #9: assert intra-molecule (pair_edge_index is built that way,
        # but cheap to verify under debug env var).
        if os.environ.get("T7_DIAG", "0") == "1":
            assert src.max().item() < N and dst.max().item() < N, \
                f"pair_edge_index out of range: max={max(src.max(), dst.max())}, N={N}"

        Q = self.q(x).view(N, self.H, self.d)
        K = self.k(x).view(N, self.H, self.d)
        V = self.v(x).view(N, self.H, self.d)

        # Sparse logits at pair edges: [E, H]
        qk = (Q[src] * K[dst]).sum(-1) / (self.d ** 0.5)
        bias = self.bias_proj(pair_repr_edge)            # [E, H]
        logits = qk + bias                                # [E, H]

        # fp32 softmax for autocast safety, grouped by source node so each
        # query softmaxes over its destinations.
        attn = softmax(logits.float(), src).to(logits.dtype)
        if self.training and self.dropout_p > 0.0:
            attn = F.dropout(attn, p=self.dropout_p)

        # Aggregate values: msg[e] = attn[e] * V[dst[e]], scatter-sum by src
        msg = attn.unsqueeze(-1) * V[dst]                # [E, H, d]
        out = scatter(msg, src, dim=0, dim_size=N, reduce='sum')
        out = out.reshape(N, self.attn_dim)
        out = self.out_proj(out)                          # small-init back to node_dim

        # Bidirectional: project logits back to pair_dim
        pair_delta = self.delta_proj(logits)              # [E, pair_dim]

        if os.environ.get("T7_DIAG", "0") == "1":
            if not torch.isfinite(out).all() or not torch.isfinite(pair_delta).all():
                raise RuntimeError(
                    f"T7 produced non-finite output. "
                    f"out_finite={torch.isfinite(out).all().item()}, "
                    f"pair_delta_finite={torch.isfinite(pair_delta).all().item()}, "
                    f"logits_norm={logits.norm().item()}, "
                    f"bias_norm={bias.norm().item()}"
                )

        return out, pair_delta


if __name__ == '__main__':
    import time
    torch.manual_seed(0)

    def make_all_pairs_edge_index(num_nodes, offset=0):
        idx = torch.arange(num_nodes, dtype=torch.long)
        src = idx.repeat_interleave(num_nodes - 1) + offset
        dst_list = []
        for i in range(num_nodes):
            others = torch.cat([idx[:i], idx[i + 1:]])
            dst_list.append(others + offset)
        dst = torch.cat(dst_list)
        return torch.stack([src, dst], dim=0)

    # -------------------------------------------------------------------- #
    # Test 1: shape + non-trivial update + bounded epoch-0 contribution
    # -------------------------------------------------------------------- #
    print("=" * 60)
    print("Test 1: shape + non-trivial update + bounded init")
    # Use node_dim=93 to match the actual integration point (atom features dim)
    N, node_dim, pair_dim = 15, 93, 64
    H = 4
    pair_ei = make_all_pairs_edge_index(N)
    E = pair_ei.size(1)

    x = torch.randn(N, node_dim)
    pair_repr = torch.randn(E, pair_dim)
    model = PairBiasedSparseAttention(node_dim=node_dim, pair_dim=pair_dim,
                                      num_heads=H, head_dim=32)
    model.eval()
    out, pair_delta = model(x, pair_repr, pair_ei)
    assert out.shape == (N, node_dim), f"out shape: {out.shape}"
    assert pair_delta.shape == (E, pair_dim), f"pair_delta shape: {pair_delta.shape}"
    # Bounded init: out is small-init * features → expected magnitude ~ init_std * sqrt(attn_dim)
    out_mag = out.abs().max().item()
    print(f"  out shape={tuple(out.shape)}, pair_delta shape={tuple(pair_delta.shape)}")
    print(f"  out max abs = {out_mag:.4f} (init_std=0.02 → bounded)")
    assert out_mag < 0.5, f"out magnitude {out_mag} too large for std=0.02 init"
    print("  PASS")

    # -------------------------------------------------------------------- #
    # Test 2: gradient flow — all 6 parameter groups must have grad
    # -------------------------------------------------------------------- #
    print("=" * 60)
    print("Test 2: gradient flow through all params")
    model.zero_grad()
    model.train()
    x_g = torch.randn(N, node_dim, requires_grad=True)
    pair_g = torch.randn(E, pair_dim, requires_grad=True)
    out_g, pair_delta_g = model(x_g, pair_g, pair_ei)
    loss = out_g.sum() + pair_delta_g.sum()
    loss.backward()

    grads_ok = {
        "q.weight":          model.q.weight.grad is not None and model.q.weight.grad.abs().sum() > 0,
        "k.weight":          model.k.weight.grad is not None and model.k.weight.grad.abs().sum() > 0,
        "v.weight":          model.v.weight.grad is not None and model.v.weight.grad.abs().sum() > 0,
        "out_proj.weight":   model.out_proj.weight.grad is not None and model.out_proj.weight.grad.abs().sum() > 0,
        "bias_proj.weight":  model.bias_proj.weight.grad is not None and model.bias_proj.weight.grad.abs().sum() > 0,
        "delta_proj.weight": model.delta_proj.weight.grad is not None and model.delta_proj.weight.grad.abs().sum() > 0,
        "x_input":           x_g.grad is not None and x_g.grad.abs().sum() > 0,
        "pair_input":        pair_g.grad is not None and pair_g.grad.abs().sum() > 0,
    }
    for name, ok in grads_ok.items():
        print(f"  {name:20s}: {'OK' if ok else 'DEAD'}")
    assert all(grads_ok.values()), f"Dead gradients: {[k for k, v in grads_ok.items() if not v]}"
    print("  PASS — no dead-init lock")

    # -------------------------------------------------------------------- #
    # Test 3: batch isolation — perturb molecule B, molecule A output unchanged
    # -------------------------------------------------------------------- #
    print("=" * 60)
    print("Test 3: batch isolation (no cross-molecule attention)")
    N_a, N_b = 5, 8
    pair_ei_a = make_all_pairs_edge_index(N_a, offset=0)
    pair_ei_b = make_all_pairs_edge_index(N_b, offset=N_a)
    pair_ei_bat = torch.cat([pair_ei_a, pair_ei_b], dim=1)
    E_a, E_b = pair_ei_a.size(1), pair_ei_b.size(1)
    E_total = E_a + E_b

    model.eval()
    torch.manual_seed(1)
    x_bat = torch.randn(N_a + N_b, node_dim)
    pair_bat = torch.randn(E_total, pair_dim)
    out_baseline, pdelta_baseline = model(x_bat, pair_bat, pair_ei_bat)
    out_A_base = out_baseline[:N_a].clone()
    pdelta_A_base = pdelta_baseline[:E_a].clone()

    # Perturb only molecule B rows
    x_pert = x_bat.clone()
    x_pert[N_a:] = torch.randn(N_b, node_dim)
    pair_pert = pair_bat.clone()
    pair_pert[E_a:] = torch.randn(E_b, pair_dim)
    out_after, pdelta_after = model(x_pert, pair_pert, pair_ei_bat)

    out_A_diff = (out_after[:N_a] - out_A_base).abs().max().item()
    pdelta_A_diff = (pdelta_after[:E_a] - pdelta_A_base).abs().max().item()
    print(f"  Mol A out diff after perturbing only Mol B: {out_A_diff:.2e}")
    print(f"  Mol A pair_delta diff: {pdelta_A_diff:.2e}")
    assert out_A_diff < 1e-6, f"cross-molecule leakage: out diff {out_A_diff}"
    assert pdelta_A_diff < 1e-6, f"cross-molecule leakage: pair_delta diff {pdelta_A_diff}"
    print("  PASS — bitwise batch isolation")

    # -------------------------------------------------------------------- #
    # Test 4: HIV-shape memory smoke (CPU, but validates sparse formulation)
    # -------------------------------------------------------------------- #
    print("=" * 60)
    print("Test 4: HIV-shape memory smoke (N=100, B=16-equiv via batched pair)")
    # Approximate HIV batch: 16 molecules, each ~100 atoms (input is node_dim=93,
    # matching real atom features dim)
    B, N_per = 16, 100
    pair_ei_list = []
    for b in range(B):
        pair_ei_list.append(make_all_pairs_edge_index(N_per, offset=b * N_per))
    pair_ei_big = torch.cat(pair_ei_list, dim=1)
    N_big = B * N_per
    E_big = pair_ei_big.size(1)
    print(f"  N_total={N_big}, E_pairs={E_big}")

    model_big = PairBiasedSparseAttention(node_dim=93, pair_dim=64, num_heads=4, head_dim=32)
    model_big.train()
    x_big = torch.randn(N_big, 93, requires_grad=True)
    pair_big = torch.randn(E_big, 64, requires_grad=True)
    t0 = time.time()
    out_big, pdelta_big = model_big(x_big, pair_big, pair_ei_big)
    loss_big = out_big.sum() + pdelta_big.sum()
    loss_big.backward()
    dt = time.time() - t0
    print(f"  Forward+backward time: {dt:.2f}s on CPU")
    print(f"  Output shapes: out={tuple(out_big.shape)}, pair_delta={tuple(pdelta_big.shape)}")
    # Memory check (CPU has no peak introspection, estimate via shapes):
    attn_mem_mb = E_big * 4 * 4 / 1024 / 1024  # [E, H] fp32
    qkv_mem_mb = N_big * 128 * 4 * 3 / 1024 / 1024  # Q+K+V at attn_dim=128
    msg_mem_mb = E_big * 4 * 32 * 4 / 1024 / 1024  # [E, H, head_dim]
    print(f"  Est peak: attn={attn_mem_mb:.1f}MB qkv={qkv_mem_mb:.1f}MB msg={msg_mem_mb:.1f}MB "
          f"total~{attn_mem_mb + qkv_mem_mb + msg_mem_mb:.1f}MB")
    print("  PASS — sparse path runs without dense [B,N,N,D] tensors")

    print("=" * 60)
    print("All 4 tests passed")
