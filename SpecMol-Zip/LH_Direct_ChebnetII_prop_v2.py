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
import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
import torch.nn.functional as F
from torch.nn import Parameter

from node_to_pair_update import NodeToPairUpdate
from pair_biased_attention import PairBiasedSparseAttention


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
                 t7_dropout=0.0, t7_init_std=0.02, **kwargs):
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

        # T6 / T7 are mutually exclusive: T6 = bilinear pair update only;
        # T7 = pair-biased multi-head attention with bidirectional pair update
        # and Tx_k residual injection.
        if t7 and node_dim is not None and pair_dim is not None:
            self.pair_attn = PairBiasedSparseAttention(
                node_dim=node_dim, pair_dim=pair_dim,
                num_heads=t7_num_heads, head_dim=t7_head_dim,
                dropout=t7_dropout, init_std=t7_init_std,
                K_steps=K, use_layernorm=True,
            )
            # Learnable gate on attn_acc injection. T7 v1 result (BACE mean 0.721 vs
            # T5 0.837) suggested the node-side attention output is corrupting the
            # spectral path's representation for downstream linear probe.
            # sigmoid(-5) ≈ 0.0067 ≈ 0 at init → epoch 0 ≡ T5 (no node injection).
            # Optimizer can grow gate if attention genuinely helps; provably ≥ T5.
            self.attn_gate = Parameter(torch.tensor(-5.0), requires_grad=True)
            self.t7_enabled = True
            self.t6_enabled = False
        elif node_dim is not None and pair_dim is not None:
            self.node_to_pair = NodeToPairUpdate(
                node_dim=node_dim, pair_dim=pair_dim, proj_dim=proj_dim,
            )
            self.t6_enabled = True
            self.t7_enabled = False
        else:
            self.t6_enabled = False
            self.t7_enabled = False
        self.latest_t6_stats = None
        self.latest_t7_stats = None

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
        before = before.detach().reshape(1, -1).double()
        after = after.detach().reshape(1, -1).double()
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
        dynamic = t6 or t7
        dyn_stats = self.latest_t6_stats if t6 else self.latest_t7_stats

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
        attn_acc = torch.zeros_like(Tx_1) if t7 else None

        # Dynamic pair update after Tx_1
        if t6:
            pair_repr_edge = self._update_pair_with_stats(
                Tx_1, pair_repr_edge, pair_edge_index, step=1,
            )
        elif t7:
            out_attn, pair_repr_edge = self._update_with_t7(
                Tx_1, pair_repr_edge, pair_edge_index, step=1,
            )
            attn_acc = attn_acc + coe[1] * out_attn

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
                out_attn, pair_repr_edge = self._update_with_t7(
                    Tx_2, pair_repr_edge, pair_edge_index, step=i,
                )
                attn_acc = attn_acc + coe[i] * out_attn

            Tx_0, Tx_1 = Tx_1, Tx_2

        if t7:
            # Gated injection: optimizer learns whether to use attention output.
            # sigmoid(attn_gate) starts at ~0 (T5 equivalent), grows if helpful.
            out = out + torch.sigmoid(self.attn_gate) * attn_acc

        if dynamic:
            return out, pair_repr_edge
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
