"""V2 model: Weighted Laplacian ChebNet II with Uni-Mol pair representations.

T5 mode (default): edge_weight computed once from static pair_repr.
T6 mode (t6=True): pair_repr updated dynamically at each K-step via
                    NodeToPairUpdate, with edge_weight recomputed per step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool as gmp

from LH_Direct_ChebnetII_prop_v2 import ChebnetII_prop_V2
from pair_to_edge_weight import PairToEdgeWeight


class ChebNetII_V2(nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50,
                 is_bns=False, act_fn='relu', pair_dim=None, proj_dim=32):
        super(ChebNetII_V2, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        # T6: pass node_dim and pair_dim to enable NodeToPairUpdate in prop layer
        self.prop1 = ChebnetII_prop_V2(
            K=K, node_dim=num_features if pair_dim else None,
            pair_dim=pair_dim, proj_dim=proj_dim,
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
                 t6=False, proj_dim=32, bias_init=5.0):
        super(LH_Direct_V2, self).__init__()
        self.t6 = t6
        self.encoder = ChebNetII_V2(
            num_features=in_dim,
            hidden=hid_dim,
            K=int(K),
            dprate=dprate,
            dropout=dropout,
            is_bns=is_bns,
            act_fn=act_fn,
            pair_dim=pair_dim if t6 else None,
            proj_dim=proj_dim,
        )
        self.pair_to_edge_weight = PairToEdgeWeight(
            pair_dim=pair_dim, hidden_dim=pair_hidden_dim, nullify=nullify_pair,
            bias_init=bias_init,
        )
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

    def _compute_edge_weight(self, data, device):
        pair_repr_edge = data.pair_repr_edge.to(device)
        pair_edge_index = data.pair_edge_index.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        return self.pair_to_edge_weight(
            pair_repr_edge, pair_edge_index, edge_index, batch,
        )

    def _encode(self, feat, edge_index, edge_weight, highpass, data, device):
        """Run encoder in T5 or T6 mode, return node features only."""
        if self.t6:
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

        edge_weight = self._compute_edge_weight(data, device)

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
