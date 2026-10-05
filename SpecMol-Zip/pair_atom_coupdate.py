"""T8: Faithful Uni-Mol-style atom<->pair co-update stack.

Motivation
----------
The injection ladder this file completes:

    T5  scalar gate          : 64-dim pair -> 1 scalar per chem bond -> Laplacian
    T6  per-step scalar      : pair updated from node Q.K (scalar) -> STILL 1 scalar to nodes
    T7  pair-bias attention  : multi-channel pair reaches nodes via attention bias,
                               BUT one block reused across the K Chebyshev steps and the
                               pair update is delta_proj(scalar-per-head logits), and the
                               per-head residual gate empirically FREEZES under the
                               contrastive objective (sigmoid ~0.12, never moves).
    T8  faithful co-update   : THIS FILE. A *stack* of L independent transformer-style
                               co-update layers, run ONCE before the Chebyshev K-loop
                               (decoupled, so no 2^K recurrence blow-up). Each layer:
                                 (1) pair -> atom : per-head pre-softmax pair bias on
                                     sparse node attention (T7's good part).
                                 (2) atom -> pair : update pair from the per-head
                                     query.key Hadamard/channel vector (Q[i,h] * K[j,h]
                                     in R^d, concatenated over heads -> Linear ->
                                     pair_dim), NOT a scalar projection of the logits.
                                     This is multi-channel (vs T6/T7's scalar) but uses
                                     the raw pre-softmax, pre-bias QK -- it is NOT the
                                     full Uni-Mol realized-attention atom->pair map.
                                 (3) iterate L times with independent weights.

Why this is the right control for the paper
-------------------------------------------
Three independent ablations already say the BACE/BBBP null is a *data* ceiling, not an
injection artifact (nullify >= V2-T5; random-pair ~ real pair; Uni-Mol-direct < RF). T8
is the top rung of the ladder: it removes the "your null is just a 64->1 bottleneck"
reviewer attack by showing the result is robust even when the pair->node path is a full
multi-channel, multi-layer, Uni-Mol-faithful co-update. On geometry-driven regression
(FreeSolv / ESOL) it is also the one place a *conditional positive* could appear.

Soft start (load-bearing for fair comparison)
---------------------------------------------
Each layer's node-side and pair-side contributions are scaled by a per-layer ReZero
scalar initialised to 0 (``node_gate``, ``pair_gate``). At init the stack is the
IDENTITY map: it returns x unchanged and pair unchanged, so an encoder using T8 is
*bitwise identical to V2-T5 at epoch 0* (same near-0.993 static gate, same Laplacian).
The optimiser opens the gates only if the geometry helps. Unlike T7's sigmoid(-2) gate
(which starts at 0.12 and then froze), a ReZero scalar starting at exactly 0 has a clean
gradient and the well-documented ReZero/SkipInit property of training stably from
identity. The multi-channel-vs-scalar question is isolated by ``pair_update='qk_hadamard'``
(multi-channel; ``'qk_outer'`` is a deprecated alias) vs ``'logits'`` (scalar
per-head logit projection, the same rule T6/T7 use).

Sparse / batch-isolated
-----------------------
All compute happens at ``pair_edge_index`` positions (intra-molecule all-pairs, built by
the data pipeline). Softmax is grouped by source node via PyG ``softmax``; value
aggregation via ``scatter`` with ``dim_size=N``. No dense [B,N,N,D] tensors; no
cross-molecule leakage (verified in __main__ Test 3).
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairAtomCoUpdateLayer(nn.Module):
    """One Uni-Mol-style co-update block (atom self-attention with pair bias +
    atom->pair update), sparse over ``pair_edge_index``.

    Args:
        node_dim:    node feature dim (the Chebyshev state dim, e.g. 93 raw atom feats).
        pair_dim:    pair feature dim (Uni-Mol pair repr, e.g. 64).
        num_heads:   attention heads (4 default).
        head_dim:    per-head dim (32 default).
        dropout:     attention dropout.
        ffn_mult:    node FFN hidden multiplier.
        pair_update: 'qk_hadamard' (multi-channel: per-head Q*K Hadamard -> pair;
                     'qk_outer' accepted as deprecated alias) or
                     'logits' (scalar-per-head logits -> pair, as T6/T7).
        init_std:    small-init std for projections that feed gated residuals.
    """

    def __init__(self, node_dim, pair_dim=64, num_heads=4, head_dim=32,
                 dropout=0.0, ffn_mult=2, pair_update='qk_hadamard', init_std=0.02):
        super().__init__()
        # 'qk_outer' kept as a deprecated alias for 'qk_hadamard' (the update is a
        # per-head Hadamard/channel product Q[i,h]*K[j,h] in R^d, NOT a [d,d] outer
        # product -- the original name was a misnomer; review 2026-06-01).
        if pair_update == 'qk_outer':
            pair_update = 'qk_hadamard'
        assert pair_update in ('qk_hadamard', 'logits'), pair_update
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.H = num_heads
        self.d = head_dim
        self.attn_dim = num_heads * head_dim
        self.dropout_p = dropout
        self.pair_update = pair_update

        # Pre-LayerNorm on x and pair (the E1 fix carried over from T7: without
        # it the pair bias dominated qk by ~400x and training NaN'd).
        self.x_norm = nn.LayerNorm(node_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)

        # Node attention projections.
        self.node_proj = nn.Sequential(nn.Linear(node_dim, self.attn_dim), nn.GELU())
        self.q = nn.Linear(self.attn_dim, self.attn_dim)
        self.k = nn.Linear(self.attn_dim, self.attn_dim)
        self.v = nn.Linear(self.attn_dim, self.attn_dim)
        # bias-free out_proj so the ReZero node_gate can drive the node-side
        # contribution to exactly 0 (VF-equivalence floor).
        self.out_proj = nn.Linear(self.attn_dim, node_dim, bias=False)

        # pair -> attention bias (per head).
        self.bias_proj = nn.Linear(pair_dim, num_heads, bias=True)

        # atom -> pair update head.
        if pair_update == 'qk_hadamard':
            # Per-head Hadamard (channel) product Q[i,h]*K[j,h] in R^d, concat over
            # heads -> [E, H*d] -> Linear -> pair_dim. Carries multi-channel
            # attention content into the pair. NOTE: this is the raw pre-softmax,
            # pre-bias QK channel (it does NOT include the pair bias, the softmax
            # weights, or the value path); it is multi-channel (vs T6/T7's scalar
            # logit projection) but is NOT the full Uni-Mol realized-attention
            # atom->pair map. See review 2026-06-01 (faithfulness finding).
            self.pair_update_head = nn.Linear(self.attn_dim, pair_dim, bias=False)
        else:
            # T7-equivalent ablation: scalar-per-head logits -> pair_dim.
            self.pair_update_head = nn.Linear(num_heads, pair_dim, bias=False)

        # Node FFN (standard transformer block).
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * ffn_mult), nn.GELU(),
            nn.Linear(node_dim * ffn_mult, node_dim),
        )
        self.ffn_norm = nn.LayerNorm(node_dim)

        # ReZero gates: init 0 -> layer is identity at start (soft start).
        self.node_gate = nn.Parameter(torch.zeros(1))
        self.pair_gate = nn.Parameter(torch.zeros(1))
        self.ffn_gate = nn.Parameter(torch.zeros(1))

        # Small init on the gated-residual projections so that once the gate
        # opens, the first updates are gentle.
        nn.init.normal_(self.bias_proj.weight, std=init_std)
        nn.init.zeros_(self.bias_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=init_std)
        nn.init.normal_(self.pair_update_head.weight, std=init_std)

    def forward(self, x, pair_repr_edge, pair_edge_index):
        """Args:
            x:               [N, node_dim]
            pair_repr_edge:  [E, pair_dim]
            pair_edge_index: [2, E]  (global PyG ids, intra-molecule)
        Returns:
            x_out:    [N, node_dim]   (x + gated attention + gated FFN)
            pair_out: [E, pair_dim]   (pair + gated atom->pair update)
        """
        from torch_geometric.utils import softmax, scatter

        N = x.size(0)
        E = pair_edge_index.size(1)
        if E == 0:
            return x, pair_repr_edge

        src = pair_edge_index[0]
        dst = pair_edge_index[1]

        x_n = self.x_norm(x)
        pair_n = self.pair_norm(pair_repr_edge)
        x_proj = self.node_proj(x_n)

        Q = self.q(x_proj).view(N, self.H, self.d)
        K = self.k(x_proj).view(N, self.H, self.d)
        V = self.v(x_proj).view(N, self.H, self.d)

        # per-head channel product at each edge: [E, H, d]
        qk_chan = Q[src] * K[dst]
        qk = qk_chan.sum(-1) / (self.d ** 0.5)          # [E, H] scaled logits
        bias = self.bias_proj(pair_n)                    # [E, H] pair -> attn bias
        logits = qk + bias                               # [E, H]

        attn = softmax(logits.float(), src).to(logits.dtype)   # grouped by src
        if self.training and self.dropout_p > 0.0:
            attn = F.dropout(attn, p=self.dropout_p)

        # pair -> atom: aggregate values
        msg = attn.unsqueeze(-1) * V[dst]                # [E, H, d]
        out = scatter(msg, src, dim=0, dim_size=N, reduce='sum')  # [N, H, d]
        out = out.reshape(N, self.attn_dim)
        out = self.out_proj(out)                          # [N, node_dim], bias-free
        x = x + torch.tanh(self.node_gate) * out          # ReZero gated residual

        # node FFN (gated residual)
        x = x + torch.tanh(self.ffn_gate) * self.ffn(self.ffn_norm(x))

        # atom -> pair: per-head Hadamard channel update vs scalar-logit ablation
        if self.pair_update == 'qk_hadamard':
            pair_src = qk_chan.reshape(E, self.attn_dim)  # [E, H*d]
        else:
            pair_src = logits                              # [E, H]
        pair_delta = self.pair_update_head(pair_src)       # [E, pair_dim]
        pair_repr_edge = pair_repr_edge + torch.tanh(self.pair_gate) * pair_delta

        return x, pair_repr_edge


class PairAtomCoUpdateStack(nn.Module):
    """Stack of L PairAtomCoUpdateLayer blocks (true iteration).

    Returns co-evolved node features and pair repr. At init (all ReZero gates 0)
    this is the identity map, so an encoder built on it is bitwise-equal to
    V2-T5 at epoch 0.
    """

    def __init__(self, node_dim, pair_dim=64, num_heads=4, head_dim=32,
                 num_layers=4, dropout=0.0, ffn_mult=2,
                 pair_update='qk_outer', init_std=0.02):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            PairAtomCoUpdateLayer(
                node_dim=node_dim, pair_dim=pair_dim, num_heads=num_heads,
                head_dim=head_dim, dropout=dropout, ffn_mult=ffn_mult,
                pair_update=pair_update, init_std=init_std,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, pair_repr_edge, pair_edge_index):
        if pair_edge_index.size(1) == 0:
            return x, pair_repr_edge
        for layer in self.layers:
            x, pair_repr_edge = layer(x, pair_repr_edge, pair_edge_index)
        if os.environ.get("T8_DIAG", "0") == "1":
            if not (torch.isfinite(x).all() and torch.isfinite(pair_repr_edge).all()):
                raise RuntimeError("T8 produced non-finite output")
        return x, pair_repr_edge

    def gate_values(self):
        """Diagnostics: per-layer (node_gate, pair_gate, ffn_gate) after tanh."""
        out = []
        for i, layer in enumerate(self.layers):
            out.append({
                "layer": i,
                "node_gate": float(torch.tanh(layer.node_gate).item()),
                "pair_gate": float(torch.tanh(layer.pair_gate).item()),
                "ffn_gate": float(torch.tanh(layer.ffn_gate).item()),
            })
        return out


class PairAtomCoUpdateStepStack(nn.Module):
    """T9: K INDEPENDENT co-update layers consumed ONE PER Chebyshev step.

    Difference vs PairAtomCoUpdateStack (T8): T8 runs all L layers ONCE before the
    spectral K-loop (a sidecar, then discards the pair). This stack is called
    step-by-step BY the K-loop via ``step(k, x, pair, pe)`` so atom & pair
    co-evolve THROUGH the spectral depth with a PERSISTENT pair state -- fixing
    T8's D2 (sidecar-not-interleaved) and D5 (pair discarded). The rich pair
    reaches nodes ONLY through the per-step attention message (multi-channel, NOT
    re-scalarized): the caller accumulates ``node_delta`` into its attn_acc and
    injects it once after the loop (the proven T7 site), so the Chebyshev
    recurrence stays pure (no 2^K blow-up). At init all ReZero gates are 0 -> each
    layer is the identity -> node_delta == 0 and pair unchanged -> T9 epoch-0 is
    bitwise-equal to V2-T5 (same static Laplacian, same near-0.993 gate).
    """

    def __init__(self, node_dim, pair_dim=64, num_heads=4, head_dim=32,
                 num_steps=10, dropout=0.0, ffn_mult=2,
                 pair_update='qk_hadamard', init_std=0.02):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.num_steps = num_steps
        self.layers = nn.ModuleList([
            PairAtomCoUpdateLayer(
                node_dim=node_dim, pair_dim=pair_dim, num_heads=num_heads,
                head_dim=head_dim, dropout=dropout, ffn_mult=ffn_mult,
                pair_update=pair_update, init_std=init_std,
            )
            for _ in range(num_steps)
        ])

    def step(self, k, x, pair_repr_edge, pair_edge_index):
        """Run co-update layer for Chebyshev step k.

        Returns (node_delta, new_pair_state):
          node_delta = x_out - x  [N, node_dim]  (the multi-channel attention+FFN
                       message; caller adds it to attn_acc, NOT back into Tx_k)
          new_pair_state            [E, pair_dim] (persists to the next step)
        Layer index is clamped to [0, num_steps-1] so callers with K+1 steps reuse
        the last layer rather than indexing out of range.
        """
        if pair_edge_index.size(1) == 0:
            return torch.zeros_like(x), pair_repr_edge
        idx = k if k < self.num_steps else self.num_steps - 1
        x_out, pair_out = self.layers[idx](x, pair_repr_edge, pair_edge_index)
        return x_out - x, pair_out

    def gate_values(self):
        out = []
        for i, layer in enumerate(self.layers):
            out.append({
                "layer": i,
                "node_gate": float(torch.tanh(layer.node_gate).item()),
                "pair_gate": float(torch.tanh(layer.pair_gate).item()),
                "ffn_gate": float(torch.tanh(layer.ffn_gate).item()),
            })
        return out


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == '__main__':
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

    print("=" * 64)
    print("Test 1: shapes + SOFT START (gates 0 -> identity == V2-T5 at epoch 0)")
    N, node_dim, pair_dim, H = 15, 93, 64, 4
    pair_ei = make_all_pairs_edge_index(N)
    E = pair_ei.size(1)
    x = torch.randn(N, node_dim)
    pair = torch.randn(E, pair_dim)
    stack = PairAtomCoUpdateStack(node_dim, pair_dim, num_heads=H, head_dim=32,
                                  num_layers=4, pair_update='qk_hadamard')
    stack.eval()
    x_out, pair_out = stack(x, pair, pair_ei)
    assert x_out.shape == (N, node_dim), x_out.shape
    assert pair_out.shape == (E, pair_dim), pair_out.shape
    # ReZero gates init 0 -> tanh(0)=0 -> identity map
    assert torch.allclose(x_out, x, atol=1e-6), \
        f"soft-start broken on x: max diff {(x_out - x).abs().max().item():.2e}"
    assert torch.allclose(pair_out, pair, atol=1e-6), \
        f"soft-start broken on pair: max diff {(pair_out - pair).abs().max().item():.2e}"
    print(f"  x_out={tuple(x_out.shape)} pair_out={tuple(pair_out.shape)}")
    print("  PASS — at init T8 is the identity map (epoch 0 == V2-T5)")

    print("=" * 64)
    print("Test 2: gradient flow through all param groups (no dead init)")
    stack.train()
    stack.zero_grad()
    x_g = torch.randn(N, node_dim, requires_grad=True)
    pair_g = torch.randn(E, pair_dim, requires_grad=True)
    # open the gates a touch so the residual paths carry gradient
    with torch.no_grad():
        for layer in stack.layers:
            layer.node_gate.fill_(0.1)
            layer.pair_gate.fill_(0.1)
            layer.ffn_gate.fill_(0.1)
    xo, po = stack(x_g, pair_g, pair_ei)
    (xo.sum() + po.sum()).backward()
    checks = {}
    L0 = stack.layers[0]
    for name, p in [
        ("q", L0.q.weight), ("k", L0.k.weight), ("v", L0.v.weight),
        ("out_proj", L0.out_proj.weight), ("bias_proj", L0.bias_proj.weight),
        ("pair_update_head", L0.pair_update_head.weight),
        ("ffn0", L0.ffn[0].weight), ("node_gate", L0.node_gate),
        ("pair_gate", L0.pair_gate), ("ffn_gate", L0.ffn_gate),
        ("x_input", x_g), ("pair_input", pair_g),
    ]:
        checks[name] = (p.grad is not None) and (p.grad.abs().sum().item() > 0)
    for name, ok in checks.items():
        print(f"  {name:18s}: {'OK' if ok else 'DEAD'}")
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    print("  PASS — every parameter group receives gradient")

    print("=" * 64)
    print("Test 3: batch isolation (perturb molecule B -> molecule A unchanged)")
    Na, Nb = 5, 8
    ei_a = make_all_pairs_edge_index(Na, offset=0)
    ei_b = make_all_pairs_edge_index(Nb, offset=Na)
    ei = torch.cat([ei_a, ei_b], dim=1)
    Ea = ei_a.size(1)
    stack.eval()
    # open gates so a leak (if any) would show
    with torch.no_grad():
        for layer in stack.layers:
            layer.node_gate.fill_(0.5); layer.pair_gate.fill_(0.5); layer.ffn_gate.fill_(0.5)
    torch.manual_seed(1)
    xb = torch.randn(Na + Nb, node_dim)
    pb = torch.randn(ei.size(1), pair_dim)
    xo0, po0 = stack(xb, pb, ei)
    xA0, pA0 = xo0[:Na].clone(), po0[:Ea].clone()
    xb2 = xb.clone(); xb2[Na:] = torch.randn(Nb, node_dim)
    pb2 = pb.clone(); pb2[Ea:] = torch.randn(ei.size(1) - Ea, pair_dim)
    xo1, po1 = stack(xb2, pb2, ei)
    dA = (xo1[:Na] - xA0).abs().max().item()
    dpA = (po1[:Ea] - pA0).abs().max().item()
    print(f"  Mol A node diff after perturbing only Mol B: {dA:.2e}")
    print(f"  Mol A pair diff: {dpA:.2e}")
    assert dA < 1e-6 and dpA < 1e-6, f"cross-molecule leakage: {dA}, {dpA}"
    print("  PASS — bitwise batch isolation")

    print("=" * 64)
    print("Test 4: 'logits' (scalar) vs 'qk_hadamard' (multi-channel) + alias")
    stack_lg = PairAtomCoUpdateStack(node_dim, pair_dim, num_heads=H, head_dim=32,
                                     num_layers=2, pair_update='logits')
    with torch.no_grad():
        for layer in stack_lg.layers:
            layer.pair_gate.fill_(0.5); layer.node_gate.fill_(0.5)
    xo_lg, po_lg = stack_lg(x, pair, pair_ei)
    assert xo_lg.shape == (N, node_dim) and po_lg.shape == (E, pair_dim)
    print(f"  logits-variant pair_update_head in_features="
          f"{stack_lg.layers[0].pair_update_head.in_features} (== num_heads={H})")
    print(f"  qk_hadamard-variant pair_update_head in_features="
          f"{stack.layers[0].pair_update_head.in_features} (== H*d={H*32})")
    assert stack_lg.layers[0].pair_update_head.in_features == H
    assert stack.layers[0].pair_update_head.in_features == H * 32
    # deprecated 'qk_outer' alias must map to qk_hadamard
    stack_alias = PairAtomCoUpdateStack(node_dim, pair_dim, num_heads=H, head_dim=32,
                                        num_layers=1, pair_update='qk_outer')
    assert stack_alias.layers[0].pair_update == 'qk_hadamard'
    assert stack_alias.layers[0].pair_update_head.in_features == H * 32
    print("  PASS — multi-channel (qk_hadamard) vs scalar (logits) wired distinctly; "
          "qk_outer alias maps to qk_hadamard")

    print("=" * 64)
    print("Test 5: memory smoke at HIV-ish shape (B=16 x N=100, CPU)")
    import time
    B, Np = 16, 100
    eis = [make_all_pairs_edge_index(Np, offset=b * Np) for b in range(B)]
    ei_big = torch.cat(eis, dim=1)
    Nbig = B * Np
    big = PairAtomCoUpdateStack(93, 64, num_heads=4, head_dim=32, num_layers=4)
    big.train()
    xbig = torch.randn(Nbig, 93, requires_grad=True)
    pbig = torch.randn(ei_big.size(1), 64, requires_grad=True)
    t0 = time.time()
    xo, po = big(xbig, pbig, ei_big)
    (xo.sum() + po.sum()).backward()
    print(f"  N={Nbig} E={ei_big.size(1)} fwd+bwd {time.time()-t0:.2f}s on CPU")
    print(f"  gate_values()[0]={big.gate_values()[0]}")
    print("  PASS — sparse path, no dense [B,N,N,D]")

    print("=" * 64)
    print("ALL T8 TESTS PASSED")
