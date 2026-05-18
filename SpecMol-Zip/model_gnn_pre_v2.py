"""V2 model: Weighted Laplacian ChebNet II with Uni-Mol pair representations.

T5 mode (default): edge_weight computed once from static pair_repr.
T6 mode (t6=True): pair_repr updated dynamically at each K-step via
                    NodeToPairUpdate, with edge_weight recomputed per step.
T7 mode (t7=True): sparse pair-biased multi-head attention; pair_repr enters
                    as pre-softmax attention bias (bypassing scalar bottleneck);
                    bidirectional pair update from pre-softmax logits.
                    Mutually exclusive with t6 and t6_safe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool as gmp

from LH_Direct_ChebnetII_prop_v2 import ChebnetII_prop_V2
from node_to_pair_update import NodeToPairUpdate
from pair_to_edge_weight import PairToEdgeWeight


class ChebNetII_V2(nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50,
                 is_bns=False, act_fn='relu', pair_dim=None, proj_dim=32,
                 t7=False, t7_num_heads=4, t7_head_dim=32,
                 t7_dropout=0.0, t7_init_std=0.02,
                 t7_disable_pair_update=False):
        super(ChebNetII_V2, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        # T6: pass node_dim+pair_dim (without t7) → NodeToPairUpdate in prop layer
        # T7: pass node_dim+pair_dim+t7=True → PairBiasedSparseAttention in prop layer
        self.prop1 = ChebnetII_prop_V2(
            K=K, node_dim=num_features if pair_dim else None,
            pair_dim=pair_dim, proj_dim=proj_dim,
            t7=t7,
            t7_num_heads=t7_num_heads, t7_head_dim=t7_head_dim,
            t7_dropout=t7_dropout, t7_init_std=t7_init_std,
            t7_disable_pair_update=t7_disable_pair_update,
        )
        assert act_fn in ['relu', 'prelu']
        self.act_fn = nn.PReLU() if act_fn == 'prelu' else nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features, momentum=0.01)
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, x, edge_index, edge_weight, highpass=True,
                pair_repr_edge=None, pair_edge_index=None, batch=None,
                pair_to_ew=None):
        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)

        result = self.prop1(
            x, edge_index, edge_weight=edge_weight, highpass=highpass,
            pair_repr_edge=pair_repr_edge, pair_edge_index=pair_edge_index,
            batch=batch, pair_to_ew=pair_to_ew,
        )

        # T6 returns (out, pair_repr_final); T5 returns out
        if isinstance(result, tuple):
            x, pair_repr_final = result
        else:
            x = result
            pair_repr_final = None

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.is_bns:
            x = self.bn(x)

        x = self.lin1(x)
        x = self.act_fn(x)

        if pair_repr_final is not None:
            return x, pair_repr_final
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class LH_Direct_V2(nn.Module):
    def __init__(self, in_dim, hid_dim, K, dprate, dropout, is_bns, act_fn,
                 type='tri', pair_dim=64, pair_hidden_dim=32, nullify_pair=False,
                 t6=False, t6_safe=False, t6_safe_frozen_zero=False,
                 proj_dim=32, t6_safe_delta_init_std=1e-3, bias_init=5.0,
                 t7=False, t7_num_heads=4, t7_head_dim=32,
                 t7_dropout=0.0, t7_init_std=0.02,
                 t7_disable_pair_update=False, randomize_pair=False):
        super(LH_Direct_V2, self).__init__()
        # Mutual exclusion: at most one of t6 / t6_safe / t7 may be enabled.
        # Reasoning: each modifies the dynamic pair_repr update path; combining
        # would either compound updates non-deterministically (T6+T7) or attach
        # two pair-update modules to the same forward (T6_safe+T7).
        if sum([bool(t6), bool(t6_safe), bool(t7)]) > 1:
            raise ValueError(
                f"At most one of t6/t6_safe/t7 may be enabled, "
                f"got t6={t6}, t6_safe={t6_safe}, t7={t7}"
            )
        self.t6 = t6
        self.t6_safe = t6_safe
        self.t6_safe_frozen_zero = t6_safe_frozen_zero
        self.t7 = t7
        # Ablation: replace the Uni-Mol pair_repr with a same-shape Gaussian
        # before it enters pair_to_edge_weight. Tests whether the gating
        # improvement is geometry-driven or just MLP capacity exploiting a
        # learnable per-bond scalar (orthogonal to the 3D pair content).
        self.randomize_pair = randomize_pair
        self.encoder = ChebNetII_V2(
            num_features=in_dim,
            hidden=hid_dim,
            K=int(K),
            dprate=dprate,
            dropout=dropout,
            is_bns=is_bns,
            act_fn=act_fn,
            pair_dim=pair_dim if (t6 or t7) else None,
            proj_dim=proj_dim,
            t7=t7,
            t7_num_heads=t7_num_heads,
            t7_head_dim=t7_head_dim,
            t7_dropout=t7_dropout,
            t7_init_std=t7_init_std,
            t7_disable_pair_update=t7_disable_pair_update,
        )
        self.pair_to_edge_weight = PairToEdgeWeight(
            pair_dim=pair_dim, hidden_dim=pair_hidden_dim, nullify=nullify_pair,
            bias_init=bias_init,
        )
        self.node_to_pair_safe = None
        self.pair_gate = None
        self.latest_t6_safe_stats = None
        self.latest_current_t6_stats = []
        self.latest_current_t7_stats = []
        self.latest_grad_norms = None
        self.act_fn = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        if type == 'tri':
            self.mlp_input = 1489
        elif type == 'pub':
            self.mlp_input = 881
        elif type == 'maccs':
            self.mlp_input = 167
        elif type == 'erg':
            self.mlp_input = 441
        self.mlp = MLP(input_size=self.mlp_input, hidden_size=1024,
                       output_size=hid_dim, dropout=0.2)
        # Initialize T6-safe-only modules after all T5-shared modules so that
        # frozen-zero T6-safe can be compared against T5 under the same seed.
        if self.t6_safe:
            self.node_to_pair_safe = NodeToPairUpdate(
                node_dim=in_dim, pair_dim=pair_dim, proj_dim=proj_dim,
            )
            if not t6_safe_frozen_zero and t6_safe_delta_init_std > 0:
                nn.init.normal_(
                    self.node_to_pair_safe.out_proj.weight,
                    mean=0.0,
                    std=t6_safe_delta_init_std,
                )
            self.pair_gate = nn.Parameter(
                torch.tensor(-5.0), requires_grad=not t6_safe_frozen_zero,
            )

    @staticmethod
    def _tensor_stats(tensor):
        tensor = tensor.detach()
        return {
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std().item()) if tensor.numel() > 1 else 0.0,
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "finite": bool(torch.isfinite(tensor).all().item()),
            "shape": tuple(tensor.shape),
        }

    @staticmethod
    def _pair_cosine(before, after):
        before = before.detach().reshape(1, -1).double()
        after = after.detach().reshape(1, -1).double()
        cosine = F.cosine_similarity(before, after, dim=1).item()
        return max(min(cosine, 1.0), -1.0)

    @staticmethod
    def _grad_norm(module):
        total = 0.0
        found = False
        for param in module.parameters():
            if param.grad is None:
                continue
            found = True
            total += float(param.grad.detach().norm().item()) ** 2
        return total ** 0.5 if found else 0.0

    def capture_gradient_diagnostics(self):
        grad_norms = {
            "pair_to_edge_weight": self._grad_norm(self.pair_to_edge_weight),
        }
        if self.t6_safe and self.node_to_pair_safe is not None:
            grad_norms.update({
                "q_proj": self._grad_norm(self.node_to_pair_safe.q_proj),
                "k_proj": self._grad_norm(self.node_to_pair_safe.k_proj),
                "out_proj": self._grad_norm(self.node_to_pair_safe.out_proj),
            })
            if self.pair_gate is not None and self.pair_gate.grad is not None:
                grad_norms["pair_gate"] = float(self.pair_gate.grad.detach().abs().item())
            else:
                grad_norms["pair_gate"] = 0.0
        self.latest_grad_norms = grad_norms
        if self.t6 and hasattr(self.encoder.prop1, "node_to_pair"):
            node_to_pair = self.encoder.prop1.node_to_pair
            grad_norms.update({
                "q_proj": self._grad_norm(node_to_pair.q_proj),
                "k_proj": self._grad_norm(node_to_pair.k_proj),
                "out_proj": self._grad_norm(node_to_pair.out_proj),
            })
            self.latest_grad_norms = grad_norms
        if self.t7 and hasattr(self.encoder.prop1, "pair_attn"):
            pair_attn = self.encoder.prop1.pair_attn
            grad_norms.update({
                "t7_q":          self._grad_norm(pair_attn.q),
                "t7_k":          self._grad_norm(pair_attn.k),
                "t7_v":          self._grad_norm(pair_attn.v),
                "t7_out_proj":   self._grad_norm(pair_attn.out_proj),
                "t7_bias_proj":  self._grad_norm(pair_attn.bias_proj),
                "t7_delta_proj": self._grad_norm(pair_attn.delta_proj),
            })
            self.latest_grad_norms = grad_norms
        return grad_norms

    def _edge_weight_from_pair(self, pair_repr_edge, pair_edge_index, edge_index, batch):
        return self.pair_to_edge_weight(
            pair_repr_edge, pair_edge_index, edge_index, batch,
        )

    def _compute_edge_weight(self, data, device):
        pair_repr_edge = data.pair_repr_edge.to(device)
        if self.randomize_pair:
            # Same shape, same dtype, standard normal (Uni-Mol pair_repr is
            # typically O(1) in scale, so randn is a fair geometry-free baseline).
            pair_repr_edge = torch.randn_like(pair_repr_edge)
        pair_edge_index = data.pair_edge_index.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        return self._edge_weight_from_pair(
            pair_repr_edge, pair_edge_index, edge_index, batch,
        )

    def _compute_t6_safe_edge_weight(self, feat, edge_index, data, device):
        pair_orig = data.pair_repr_edge.to(device)
        pair_edge_index = data.pair_edge_index.to(device)
        batch = data.batch.to(device)

        if self.t6_safe_frozen_zero:
            gate = pair_orig.new_tensor(0.0)
            delta = torch.zeros_like(pair_orig)
            pair_updated = pair_orig
        else:
            pair_raw = self.node_to_pair_safe(feat, pair_orig, pair_edge_index)
            delta = pair_raw - pair_orig
            gate = torch.sigmoid(self.pair_gate)
            pair_updated = pair_orig + gate * delta

        edge_weight = self._edge_weight_from_pair(
            pair_updated, pair_edge_index, edge_index, batch,
        )

        with torch.no_grad():
            pair_norm = pair_orig.detach().norm().clamp_min(1e-12)
            delta_norm = delta.detach().norm()
            drift = self._pair_cosine(pair_orig, pair_updated)
            self.latest_t6_safe_stats = {
                "mode": "t6_safe_frozen_zero" if self.t6_safe_frozen_zero else "t6_safe",
                "gate": float(gate.detach().item()),
                "delta_norm": float(delta_norm.item()),
                "pair_norm": float(pair_norm.item()),
                "delta_pair_ratio": float((delta_norm / pair_norm).item()),
                "pair_cosine_to_original": float(drift),
                "edge_weight": self._tensor_stats(edge_weight),
                "edge_count": int(edge_index.size(1)),
            }
        return edge_weight

    def _encode(self, feat, edge_index, edge_weight, highpass, data, device):
        """Run encoder in T5, T6, or T7 mode, return node features only."""
        if self.t6 or self.t7:
            # T6 / T7 share the same input plumbing — the prop layer picks the
            # right dynamic update based on its t6_enabled / t7_enabled flags.
            pair_repr_edge = data.pair_repr_edge.to(device).clone()
            pair_edge_index = data.pair_edge_index.to(device)
            batch = data.batch.to(device)
            result = self.encoder(
                x=feat, edge_index=edge_index, edge_weight=edge_weight,
                highpass=highpass,
                pair_repr_edge=pair_repr_edge,
                pair_edge_index=pair_edge_index,
                batch=batch,
                pair_to_ew=self.pair_to_edge_weight,
            )
            if self.t6:
                stats = getattr(self.encoder.prop1, "latest_t6_stats", None)
                if stats is not None:
                    self.latest_current_t6_stats.append(stats)
            elif self.t7:
                stats = getattr(self.encoder.prop1, "latest_t7_stats", None)
                if stats is not None:
                    self.latest_current_t7_stats.append(stats)
            # Discard pair_repr_final; only need node features for downstream
            return result[0] if isinstance(result, tuple) else result
        else:
            return self.encoder(x=feat, edge_index=edge_index,
                                edge_weight=edge_weight, highpass=highpass)

    def _forward_impl(self, data, device):
        feat = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        fp = data.fps.to(device)
        self.latest_current_t6_stats = []
        self.latest_current_t7_stats = []

        if self.t6_safe:
            edge_weight = self._compute_t6_safe_edge_weight(feat, edge_index, data, device)
        else:
            edge_weight = self._compute_edge_weight(data, device)

        # H5 design note (2026-05-13 audit): in T6 mode each of the highpass and lowpass
        # branches calls self._encode → self.encoder, which clones pair_repr_edge inside.
        # Therefore the two branches update pair_repr along INDEPENDENT trajectories.
        # This is mathematically valid (each branch sees its own filter dynamics) but is
        # a non-obvious design choice. TODO future work: ablate shared-vs-independent pair
        # update (one branch updates, the other reuses).
        h1 = self._encode(feat, edge_index, edge_weight, True, data, device)
        high_x_mean = gmp(h1, batch)

        h2 = self._encode(feat, edge_index, edge_weight, False, data, device)
        low_x_mean = gmp(h2, batch)

        h = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        spec_x_mean = gmp(h, batch)

        fp = fp.reshape(len(fp) // self.mlp_input, self.mlp_input)
        x_fp = self.mlp(fp)

        return low_x_mean, high_x_mean, spec_x_mean, x_fp

    def get_embedding(self, data, device):
        low_x_mean, high_x_mean, spec_x_mean, x_fp = self._forward_impl(data, device)
        y = data.y.to(device)
        return low_x_mean, high_x_mean, spec_x_mean, x_fp, y

    def forward(self, data, device):
        return self._forward_impl(data, device)
