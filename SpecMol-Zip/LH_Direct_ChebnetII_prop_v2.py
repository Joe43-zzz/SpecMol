"""V2 ChebNet II propagation layer.

T5 mode: forward() uses static edge_weight (from PairToEdgeWeight), single Laplacian.
T6 mode: forward() dynamically updates pair_repr at each K-step via NodeToPairUpdate
         (bilinear Q·K → pair_dim residual), recomputing edge_weight and Laplacian.
T7 mode: forward() applies sparse pair-biased multi-head attention each K-step.
         pair_repr enters as pre-softmax bias on attention logits (bypassing the
         scalar edge_weight bottleneck); attention output is residually added to
         Tx_k (hybrid spectral-attention); pre-softmax logits feed back into
         pair_repr. Mutually exclusive with T6.
"""

import math
import os
import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
import torch.nn.functional as F
from torch.nn import Parameter

from node_to_pair_update import NodeToPairUpdate
from pair_biased_attention import PairBiasedSparseAttention
from pair_atom_coupdate import PairAtomCoUpdateStack, PairAtomCoUpdateStepStack


def cheby(i, x):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1
        T1 = x
        for ii in range(2, i + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2


def gamma_h(slides, beta_a_h, beta_b_h):
    length = len(slides)
    gamma_h = torch.zeros(length, dtype=slides.dtype, device=slides.device)
    for j in range(length):
        term1 = 1 / 2 * F.relu(beta_b_h) * (1 + torch.cos(((1 + j + slides[j]) / length) * torch.pi))
        gamma_h[j] = F.relu(beta_a_h) + term1
    return gamma_h


def gamma_l(slides, beta_a_l, beta_b_l):
    length = len(slides)
    gamma_l = torch.zeros(length, dtype=slides.dtype, device=slides.device)
    for j in range(length):
        term1 = 1 / 2 * F.relu(beta_b_l) * (1 + torch.cos(((1 + j + slides[j]) / length) * torch.pi))
        gamma_l[j] = F.relu(beta_a_l) - term1
    return gamma_l


class ChebnetII_prop_V2(MessagePassing):
    def __init__(self, K, node_dim=None, pair_dim=None, proj_dim=32,
                 t7=False, t7_num_heads=4, t7_head_dim=32,
                 t7_dropout=0.0, t7_init_std=0.02,
                 t7_disable_pair_update=False, t7_force_inject=False,
                 t8=False, t8_num_layers=4, t8_num_heads=4, t8_head_dim=32,
                 t8_dropout=0.0, t8_init_std=0.02, t8_pair_update='qk_hadamard',
                 t9=False, t9_num_heads=4, t9_head_dim=32,
                 t9_dropout=0.0, t9_init_std=0.02, t9_pair_update='qk_hadamard',
                 **kwargs):
        super(ChebnetII_prop_V2, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.node_axis = -2

        self.initial_val_low  = Parameter(torch.tensor(2.0), requires_grad=False)
        self.temp_low         = Parameter(torch.Tensor(self.K), requires_grad=True)
        self.temp_high        = Parameter(torch.Tensor(self.K), requires_grad=True)
        self.initial_val_high = Parameter(torch.tensor(0.0), requires_grad=False)

        self.beta_a_h = Parameter(torch.tensor(0.0), requires_grad=True)
        self.beta_a_l = Parameter(torch.tensor(2.0), requires_grad=True)
        self.beta_b_h = Parameter(torch.tensor(2.0), requires_grad=True)
        self.beta_b_l = Parameter(torch.tensor(2.0), requires_grad=True)

        self.slides_l = Parameter(torch.zeros(self.K + 1), requires_grad=True)
        self.slides_h = Parameter(torch.zeros(self.K + 1), requires_grad=True)

        # T6 / T7 / T8 are mutually exclusive dynamic pair paths.
        #   T6 = bilinear pair update only (scalar reaches nodes).
        #   T7 = pair-biased multi-head attention + bidirectional pair update,
        #        one block reused across the K Chebyshev steps.
        #   T8 = faithful Uni-Mol co-update STACK (L independent layers, per-head
        #        atom->pair update), run ONCE before the K-loop (decoupled), then
        #        the co-evolved pair feeds the static Laplacian and the co-evolved
        #        node features are residually added to the spectral output.
        if t8 and node_dim is not None and pair_dim is not None:
            self.pair_coupdate = PairAtomCoUpdateStack(
                node_dim=node_dim, pair_dim=pair_dim,
                num_heads=t8_num_heads, head_dim=t8_head_dim,
                num_layers=t8_num_layers, dropout=t8_dropout,
                pair_update=t8_pair_update, init_std=t8_init_std,
            )
            # The co-evolved node features enter `out` as a DELTA residual
            # (x_co - x), NOT a separately-gated term. The stack's own per-layer
            # ReZero gates start at 0, so at init x_co == x exactly -> delta == 0
            # -> the whole T8 path is bitwise-equal to V2-T5 (epoch-0 sanity).
            # Using the delta (rather than an extra outer gate on raw x_co) is
            # deliberate: an outer gate initialised at 0 would block gradient to
            # the inner node-side gates (a chained-ReZero cold start, since the
            # node path reaches `out` only through that outer gate). The delta
            # form lets the inner node gates receive gradient from step 1 while
            # preserving the identity start, and keeps the soft start in ONE
            # place (the per-layer gates). The pair side already reaches `out`
            # through the rebuilt Laplacian, independent of this residual.
            self.t8_enabled = True
            self.t7_enabled = False
            self.t6_enabled = False
            self.t7_disable_pair_update = False
        elif t7 and node_dim is not None and pair_dim is not None:
            # The node-side attention gate is now a per-head [num_heads] vector
            # inside PairBiasedSparseAttention (applied before its bias-free
            # out_proj), not a scalar here. T7 v1 (BACE 0.721 vs T5 0.837)
            # showed the node injection can corrupt the spectral path for the
            # frozen probe; per-head gates let the optimizer keep useful heads
            # and shut harmful ones independently. The gates init to -5.0 ->
            # sigmoid ≈ 0, so epoch 0 still equals the pure-T5 spectral path.
            self.pair_attn = PairBiasedSparseAttention(
                node_dim=node_dim, pair_dim=pair_dim,
                num_heads=t7_num_heads, head_dim=t7_head_dim,
                dropout=t7_dropout, init_std=t7_init_std,
                K_steps=K, use_layernorm=True,
                force_inject=t7_force_inject,
            )
            # T7-static-pair flag: if set, pair_clone += pair_delta is SKIPPED in
            # forward, so pair_repr stays at Uni-Mol values through K-loop.
            # Combined with the gates ≈ 0, this ablates ALL T7 dynamic behavior:
            # the model degenerates to "T5 + dead attention modules". If this
            # variant matches T5 AUC, then pair_repr update is the toxic component.
            self.t7_disable_pair_update = t7_disable_pair_update
            self.t7_enabled = True
            self.t6_enabled = False
            self.t8_enabled = False
        elif node_dim is not None and pair_dim is not None and not t9:
            self.node_to_pair = NodeToPairUpdate(
                node_dim=node_dim, pair_dim=pair_dim, proj_dim=proj_dim,
            )
            self.t6_enabled = True
            self.t7_enabled = False
            self.t8_enabled = False
        else:
            self.t6_enabled = False
            self.t7_enabled = False
            self.t8_enabled = False
        # T9: faithful per-step interleaved co-update. K INDEPENDENT layers
        # consumed one-per-Chebyshev-step with a PERSISTENT pair state (fixes T8's
        # D2 sidecar/D5 discard). Mutually exclusive with t6/t7/t8 (the chain above
        # sets those False when t9 is on). The spectral path is UNCHANGED from
        # V2-T5 (static Laplacian from the incoming edge_weight, no per-step
        # rebuild); the rich pair reaches nodes ONLY through the per-step attention
        # message accumulated into attn_acc -- so epoch-0 == V2-T5 (ReZero gates 0).
        self.t9_enabled = bool(t9 and node_dim is not None and pair_dim is not None)
        if self.t9_enabled:
            self.pair_coupdate_step = PairAtomCoUpdateStepStack(
                node_dim=node_dim, pair_dim=pair_dim,
                num_heads=t9_num_heads, head_dim=t9_head_dim,
                num_steps=K, dropout=t9_dropout,
                pair_update=t9_pair_update, init_std=t9_init_std,
            )
        self.latest_t6_stats = None
        self.latest_t7_stats = None
        self.latest_t8_stats = None
        self.latest_t9_stats = None

    def reset_parameters(self):
        self.temp_low.data.fill_(2.0 / self.K)
        self.temp_high.data.fill_(2.0 / self.K)

    def _build_laplacian(self, edge_index, edge_weight, dtype, num_nodes):
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight=edge_weight, normalization='sym',
            dtype=dtype, num_nodes=num_nodes,
        )
        return add_self_loops(edge_index1, norm1, fill_value=-1.0, num_nodes=num_nodes)

    @staticmethod
    def _edge_weight_stats(edge_weight):
        edge_weight = edge_weight.detach()
        if edge_weight.numel() == 0:
            return {
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "finite": True, "shape": tuple(edge_weight.shape),
            }
        return {
            "mean": float(edge_weight.mean().item()),
            "std": float(edge_weight.std().item()) if edge_weight.numel() > 1 else 0.0,
            "min": float(edge_weight.min().item()),
            "max": float(edge_weight.max().item()),
            "finite": bool(torch.isfinite(edge_weight).all().item()),
            "shape": tuple(edge_weight.shape),
        }

    @staticmethod
    def _pair_cosine(before, after):
        if not os.environ.get("T7_DIAG"):
            return 0.0
        before = before.detach().reshape(1, -1).float()
        after = after.detach().reshape(1, -1).float()
        cosine = F.cosine_similarity(before, after, dim=1).item()
        return max(min(cosine, 1.0), -1.0)

    def _update_pair_with_stats(self, t_k, pair_repr_edge, pair_edge_index, step):
        before = pair_repr_edge
        after = self.node_to_pair(t_k, before, pair_edge_index)
        with torch.no_grad():
            delta = after.detach() - before.detach()
            pair_norm = before.detach().norm().clamp_min(1e-12)
            drift = self._pair_cosine(before, after)
            self.latest_t6_stats["updates"].append({
                "step": int(step),
                "delta_pair_ratio": float((delta.norm() / pair_norm).item()),
                "pair_cosine_to_original": float(drift),
            })
        return after

    def _update_with_t7(self, t_k, pair_repr_edge, pair_edge_index, step):
        """Apply T7 attention: returns (out_attn for residual, updated pair_repr).

        out_attn is node-side residual delta [N, node_dim] to add to Tx_k.
        pair_delta is added to pair_repr_edge here (forward-local).
        """
        before = pair_repr_edge
        out_attn, pair_delta = self.pair_attn(t_k, before, pair_edge_index)
        after = before + pair_delta
        with torch.no_grad():
            pair_norm = before.detach().norm().clamp_min(1e-12)
            pair_delta_norm = pair_delta.detach().norm()
            drift = self._pair_cosine(before, after)
            out_norm = out_attn.detach().norm()
            self.latest_t7_stats["updates"].append({
                "step": int(step),
                "delta_pair_ratio": float((pair_delta_norm / pair_norm).item()),
                "pair_cosine_to_original": float(drift),
                "out_attn_norm": float(out_norm.item()),
            })
        return out_attn, after

    def forward(self, x, edge_index, edge_weight, highpass=True,
                pair_repr_edge=None, pair_edge_index=None, batch=None,
                pair_to_ew=None):
        """
        Args:
            x:              [N, F] node features
            edge_index:     [2, E_chembond] chem bond edges
            edge_weight:    [E_chembond] initial edge weights (T5 static / T6 initial)
            highpass:       bool
            --- T6-only args (all None → T5 fallback) ---
            pair_repr_edge: [E_allpairs, pair_dim] mutable pair repr state
            pair_edge_index:[2, E_allpairs] all-pairs edge index
            batch:          [N] PyG batch vector
            pair_to_ew:     PairToEdgeWeight module for recomputing edge weights

        Returns:
            out:            [N, F] filtered node features
            pair_repr_edge: [E_allpairs, pair_dim] final pair repr (T6) or None (T5)
        """
        if highpass:
            slides_tmp = 0.5 * torch.tanh(self.slides_h)
            coe_tmp = gamma_h(slides_tmp, self.beta_a_h, self.beta_b_h)
        else:
            slides_tmp = 0.5 * torch.tanh(self.slides_l)
            coe_tmp = gamma_l(slides_tmp, self.beta_a_l, self.beta_b_l)

        coe = coe_tmp.clone()
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        num_nodes = x.size(self.node_axis)
        t6 = self.t6_enabled and pair_repr_edge is not None
        t7 = self.t7_enabled and pair_repr_edge is not None
        t8 = getattr(self, "t8_enabled", False) and pair_repr_edge is not None
        t9 = getattr(self, "t9_enabled", False) and pair_repr_edge is not None
        self.latest_t6_stats = {
            "highpass": bool(highpass),
            "updates": [],
            "edge_weights": [],
        } if t6 else None
        self.latest_t7_stats = {
            "highpass": bool(highpass),
            "updates": [],
            "edge_weights": [],
        } if t7 else None
        # Pick a single "dynamic" flag for branches that are common to T6 and T7
        # (per-step pair-driven Laplacian rebuild). T8 is intentionally NOT
        # "dynamic": it co-evolves atom/pair ONCE before the K-loop, then runs the
        # ordinary static Chebyshev recurrence on the co-evolved Laplacian. This
        # keeps the Chebyshev polynomial structure intact and avoids the 2^K
        # recurrence blow-up that forced T7's awkward attn accumulation.
        dynamic = t6 or t7
        dyn_stats = self.latest_t6_stats if t6 else self.latest_t7_stats

        # --- T8: faithful co-update run ONCE, before any spectral propagation ---
        t8_x_delta = None
        if t8:
            x_co, pair_co = self.pair_coupdate(x, pair_repr_edge, pair_edge_index)
            # Node-side residual as a DELTA: at init x_co == x -> delta == 0
            # (soft start) and the inner node gates still receive gradient.
            t8_x_delta = x_co - x
            pair_repr_edge = pair_co
            # Rebuild edge_weight from the co-evolved pair so the spectral path
            # also sees the geometry (single recompute, not per-step).
            if pair_to_ew is not None:
                edge_weight = pair_to_ew(
                    pair_repr_edge, pair_edge_index, edge_index, batch,
                )
            with torch.no_grad():
                gates = self.pair_coupdate.gate_values()
                self.latest_t8_stats = {
                    "highpass": bool(highpass),
                    "layer_gates": gates,
                    "node_delta_norm": float(t8_x_delta.norm().item()),
                    "edge_weight": self._edge_weight_stats(edge_weight),
                }

        # --- Step 0: initial Laplacian from edge_weight ---
        edge_index_tilde, norm_tilde = self._build_laplacian(
            edge_index, edge_weight, x.dtype, num_nodes,
        )
        if dynamic:
            dyn_stats["edge_weights"].append({
                "step": 1,
                **self._edge_weight_stats(edge_weight),
            })

        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)

        # T7 accumulates attention outputs SEPARATELY from the Chebyshev
        # recurrence. Injecting out_attn into Tx_k inside the loop would amplify
        # by 2^K through the `Tx_{k+1} = 2*L*Tx_k - Tx_{k-1}` recurrence (a
        # 2000× blow-up was observed in smoke testing). Keeping Tx_k purely
        # spectral preserves the Chebyshev polynomial structure; pair info
        # still reaches node embeddings via (a) the accumulated attn_acc added
        # to `out` at the end, and (b) the per-step Laplacian rebuilt from the
        # updated pair_repr (T6-style indirect path).
        attn_acc = torch.zeros_like(Tx_1) if (t7 or t9) else None

        # Dynamic pair update after Tx_1
        if t6:
            pair_repr_edge = self._update_pair_with_stats(
                Tx_1, pair_repr_edge, pair_edge_index, step=1,
            )
        elif t7:
            out_attn, pair_after = self._update_with_t7(
                Tx_1, pair_repr_edge, pair_edge_index, step=1,
            )
            if not self.t7_disable_pair_update:
                pair_repr_edge = pair_after
            # S3b (VeriMAP Rung 2, 2026-05-27): uniform accumulation, no coe[i]
            # weighting. coe[i] are Chebyshev spectral coefficients whose
            # magnitudes and signs are calibrated to filter constraints, not to
            # weight attention outputs. Multiplying attention by coe[i] can
            # invert the attention signal (negative coe) and amplifies whatever
            # spectral shape exists, decoupled from attention quality.
            # The per-head attn_gate (inside PairBiasedSparseAttention) is the
            # legitimate scaling channel; final averaging by K happens at the
            # injection site below.
            attn_acc = attn_acc + out_attn          # was: + coe[1] * out_attn
        elif t9:
            # T9: per-step co-update layer 0 (Tx_1). Returns the multi-channel node
            # message (x_out - Tx_1) accumulated into attn_acc (NOT fed back into
            # Tx_1, so the Chebyshev recurrence stays pure); pair_repr_edge persists.
            node_delta, pair_repr_edge = self.pair_coupdate_step.step(
                0, Tx_1, pair_repr_edge, pair_edge_index,
            )
            attn_acc = attn_acc + node_delta

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            # Recompute edge_weight + Laplacian from updated pair_repr (T6 or T7)
            if dynamic:
                edge_weight = pair_to_ew(
                    pair_repr_edge, pair_edge_index, edge_index, batch,
                )
                dyn_stats["edge_weights"].append({
                    "step": int(i),
                    **self._edge_weight_stats(edge_weight),
                })
                edge_index_tilde, norm_tilde = self._build_laplacian(
                    edge_index, edge_weight, x.dtype, num_nodes,
                )

            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2

            # Dynamic pair update after Tx_k
            if t6:
                pair_repr_edge = self._update_pair_with_stats(
                    Tx_2, pair_repr_edge, pair_edge_index, step=i,
                )
            elif t7:
                out_attn, pair_after = self._update_with_t7(
                    Tx_2, pair_repr_edge, pair_edge_index, step=i,
                )
                if not self.t7_disable_pair_update:
                    pair_repr_edge = pair_after
                attn_acc = attn_acc + out_attn      # S3b: was + coe[i] * out_attn
            elif t9:
                # T9: per-step co-update layer (i-1) for Tx_i; pair persists.
                node_delta, pair_repr_edge = self.pair_coupdate_step.step(
                    i - 1, Tx_2, pair_repr_edge, pair_edge_index,
                )
                attn_acc = attn_acc + node_delta

            Tx_0, Tx_1 = Tx_1, Tx_2

        if t7:
            # attn_acc is already per-head gated inside PairBiasedSparseAttention
            # (each out_attn was scaled by sigmoid(pair_attn.attn_gate) before
            # the bias-free out_proj). At init the gates are -5.0 -> ≈0, so
            # epoch 0 still equals the pure-T5 spectral path.
            # S3b: divide by K so the accumulated attention magnitude doesn't
            # scale with the polynomial order. attn_gate controls total weight.
            out = out + attn_acc / max(self.K, 1)

        if t9:
            # T9: inject the accumulated per-step co-update messages ONCE (T7's
            # proven site, /K so magnitude doesn't scale with polynomial order).
            # ReZero gates init 0 -> attn_acc == 0 at epoch 0 -> bitwise-equal to
            # V2-T5 (the spectral path used the same static Laplacian, untouched).
            out = out + attn_acc / max(self.K, 1)
            with torch.no_grad():
                self.latest_t9_stats = {
                    "highpass": bool(highpass),
                    "layer_gates": self.pair_coupdate_step.gate_values(),
                    "attn_acc_norm": float(attn_acc.norm().item()),
                }

        if t8:
            # Add the co-evolved node-feature DELTA. At init the stack is the
            # identity map (per-layer ReZero gates = 0) so t8_x_delta == 0 and
            # epoch 0 is bitwise-equal to the pure-T5 spectral path.
            out = out + t8_x_delta

        if dynamic or t8 or t9:
            return out, pair_repr_edge
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
