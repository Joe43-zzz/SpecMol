import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gmp

from LH_Direct_ChebnetII_prop import ChebnetII_prop
from pair_repr_init import PairReprInit


# ---------------------------------------------------------------------------
# Unchanged: LogReg, MLP, GNNCon
# ---------------------------------------------------------------------------

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes, dropout=0.2):
        super(LogReg, self).__init__()
        self.fc  = nn.Linear(hid_dim * 2, hid_dim)
        self.fc1 = nn.Linear(hid_dim, n_classes)
        self.dp  = nn.Dropout(dropout)

    def forward(self, x):
        ret = self.fc(x)
        ret = self.dp(ret)
        ret = self.fc1(ret)
        return ret


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(MLP, self).__init__()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.fc2     = nn.Linear(hidden_size, hidden_size)
        self.fc3     = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class GNNCon(torch.nn.Module):
    def __init__(self, eps=0., train_eps=True, n_output=512, num_features_xt=78,
                 output_dim=512, dropout=0.2, encoder1=None, encoder2=None):
        super(GNNCon, self).__init__()
        self.num_features_xt = num_features_xt
        self.n_output  = n_output
        self.dropout   = dropout
        self.output_dim = output_dim
        self.gat = encoder1
        self.gin = encoder2
        self.pre_head = nn.Sequential(
            nn.Linear(output_dim, 2048), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2048, 1024),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, self.n_output),
        )

    def forward(self, data1):
        x, edge_index, batch, x_size, edge_size, edge_attr = (
            data1.x, data1.edge_index, data1.batch,
            data1.x_size, data1.edge_size, data1.edge_attr,
        )
        x          = x.to('cuda')
        edge_index = edge_index.to('cuda')
        batch      = batch.to('cuda')
        edge_attr  = edge_attr.to('cuda')

        edge_index1, norm1 = get_laplacian(edge_index, edge_attr, normalization='sym', dtype=x.dtype)

        x1, weight = self.gat(x, edge_index, edge_attr, batch)
        out1 = self.pre_head(x1)
        x2   = self.gin(x, edge_index, edge_attr, batch)
        out2 = self.pre_head(x2)

        len_w = edge_index.shape[1]
        w1    = weight[1]
        ew1   = w1[:len_w]
        xw1   = w1[len_w:]
        return x1, out1, x2, out2, ew1, xw1


# ---------------------------------------------------------------------------
# ChebNetII  (adapted: pair_repr_dim → num_heads, edge_index_for_pair removed)
# ---------------------------------------------------------------------------

class ChebNetII(torch.nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50,
                 is_bns=False, act_fn='relu', num_heads=None):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(num_features, hidden)

        self.prop1 = ChebnetII_prop(
            K=K,
            num_heads=num_heads,
            node_dim=num_features,      # node feature dim before lin1
        )

        assert act_fn in ['relu', 'prelu']
        self.act_fn  = nn.PReLU() if act_fn == 'prelu' else nn.ReLU()
        self.bn      = torch.nn.BatchNorm1d(num_features, momentum=0.01)
        self.is_bns  = is_bns
        self.dprate  = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, highpass=True, pair_repr=None, batch=None):
        if self.dprate == 0.0:
            x, pair_repr = self.prop1(
                x, edge_index,
                edge_weight=edge_weight, highpass=highpass,
                pair_repr=pair_repr, batch=batch,
            )
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x, pair_repr = self.prop1(
                x, edge_index,
                edge_weight=edge_weight, highpass=highpass,
                pair_repr=pair_repr, batch=batch,
            )

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.is_bns:
            x = self.bn(x)

        x = self.lin1(x)
        x = self.act_fn(x)

        return x, pair_repr


# ---------------------------------------------------------------------------
# LH_Direct  (adapted: pair_repr_dim → num_heads, new pair repr pipeline)
# ---------------------------------------------------------------------------

class LH_Direct(nn.Module):
    def __init__(self, in_dim, hid_dim, K, dprate, dropout, is_bns, act_fn,
                 type='tri', num_heads=None, num_atom_types=64):
        super(LH_Direct, self).__init__()

        self.num_heads      = num_heads
        self.num_atom_types = num_atom_types
        edge_types          = num_atom_types * num_atom_types

        self.encoder = ChebNetII(
            num_features=in_dim,
            hidden=hid_dim,
            K=K,
            dprate=dprate,
            dropout=dropout,
            is_bns=is_bns,
            act_fn=act_fn,
            num_heads=num_heads,
        )

        self.act_fn = nn.ReLU()
        self.alpha  = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta   = nn.Parameter(torch.tensor(0.5), requires_grad=True)

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

        self.pair_repr_init = (
            PairReprInit(K=128, num_heads=num_heads, edge_types=edge_types, activation_fn='gelu')
            if num_heads is not None else None
        )

    def _build_full_pair_repr(self, data, batch, device):
        """
        Build dense pair repr [total_nodes, total_nodes, num_heads] on-the-fly
        from data.pos (3D coords) and data.x (atom features).

        Falls back to None when pos is absent (old datasets without 3D coords).
        """
        if self.pair_repr_init is None:
            return None

        pos = getattr(data, 'pos', None)
        if pos is None or not torch.is_tensor(pos) or pos.size(0) == 0:
            return None
        # After PyG batching pos.size(0) must equal total_nodes
        if pos.size(0) != batch.size(0):
            return None
        pos = pos.to(device=device, dtype=torch.float32)   # [N, 3]

        # --- pairwise Euclidean distance matrix ---
        pair_dist = torch.cdist(pos, pos, p=2)             # [N, N]

        # --- atom type index ---
        # Priority: data.atom_type (integer tensor, if stored in dataset)
        # Fallback:  argmax over data.x (works when first columns are one-hot)
        atom_type_attr = getattr(data, 'atom_type', None)
        if atom_type_attr is not None and torch.is_tensor(atom_type_attr):
            atom_type = atom_type_attr.to(device=device).long()
        else:
            x_feat    = data.x.to(device=device)
            atom_type = x_feat.argmax(dim=1) if x_feat.dim() == 2 else x_feat.long()

        atom_type = atom_type.clamp(0, self.num_atom_types - 1)

        # edge_type[i,j] = atom_type_i * num_atom_types + atom_type_j
        pair_edge_type = (
            atom_type.unsqueeze(1) * self.num_atom_types + atom_type.unsqueeze(0)
        )                                                   # [N, N]

        # --- PairReprInit (expects batch dim) ---
        pair_repr = self.pair_repr_init(
            pair_dist.unsqueeze(0),        # [1, N, N]
            pair_edge_type.unsqueeze(0),   # [1, N, N]
        ).squeeze(0)                        # [N, N, num_heads]

        # --- batch mask: zero out cross-graph positions ---
        batch_t = batch.to(device=device)
        mask    = (batch_t.unsqueeze(0) == batch_t.unsqueeze(1))  # [N, N]
        pair_repr = pair_repr * mask.unsqueeze(-1).float()

        return pair_repr

    def get_embedding(self, data, device):
        feat, edge_index, batch, y, edge_attr, fp, w = (
            data.x, data.edge_index, data.batch, data.y,
            data.edge_attr, data.fps, data.w,
        )
        feat       = feat.to(device)
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.to(device) if edge_attr is not None else None
        batch      = batch.to(device)
        fp         = fp.to(device)
        y          = y.to(device)

        pair_repr      = self._build_full_pair_repr(data, batch, device)
        high_pair_repr = pair_repr.clone() if pair_repr is not None else None
        low_pair_repr  = pair_repr.clone() if pair_repr is not None else None

        h1, high_pair_repr = self.encoder(
            x=feat, edge_index=edge_index,
            highpass=True, pair_repr=high_pair_repr, batch=batch,
        )
        high_x_mean = gmp(h1, batch)

        h2, low_pair_repr = self.encoder(
            x=feat, edge_index=edge_index,
            highpass=False, pair_repr=low_pair_repr, batch=batch,
        )
        low_x_mean = gmp(h2, batch)

        h          = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        spec_x_mean = gmp(h, batch)

        fp   = fp.reshape(len(fp) // self.mlp_input, self.mlp_input)
        x_fp = self.mlp(fp)

        return low_x_mean, high_x_mean, spec_x_mean, x_fp, y

    def forward(self, data, device):
        feat, edge_index, batch, y, edge_attr, fp, w = (
            data.x, data.edge_index, data.batch, data.y,
            data.edge_attr, data.fps, data.w,
        )
        feat       = feat.to(device)
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.to(device) if edge_attr is not None else None
        batch      = batch.to(device)
        fp         = fp.to(device)

        pair_repr      = self._build_full_pair_repr(data, batch, device)
        high_pair_repr = pair_repr.clone() if pair_repr is not None else None
        low_pair_repr  = pair_repr.clone() if pair_repr is not None else None

        h1, high_pair_repr = self.encoder(
            x=feat, edge_index=edge_index, edge_weight=edge_attr,
            highpass=True, pair_repr=high_pair_repr, batch=batch,
        )
        high_x_mean = gmp(h1, batch)

        h2, low_pair_repr = self.encoder(
            x=feat, edge_index=edge_index, edge_weight=edge_attr,
            highpass=False, pair_repr=low_pair_repr, batch=batch,
        )
        low_x_mean = gmp(h2, batch)

        h           = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        spec_x_mean = gmp(h, batch)

        fp   = fp.reshape(len(fp) // self.mlp_input, self.mlp_input)
        x_fp = self.mlp(fp)

        return low_x_mean, high_x_mean, spec_x_mean, x_fp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch
    from torch_geometric.data import Data, Batch

    torch.manual_seed(0)
    device = 'cpu'

    D, FP_DIM = 93, 1489

    def make_graph(N, E, seed, add_pos=True):
        torch.manual_seed(seed)
        x          = torch.randn(N, D)
        edge_index = torch.stack([
            torch.randint(0, N, (E,)),
            torch.randint(0, N, (E,)),
        ])
        y   = torch.randint(0, 2, (N,)).float()
        fps = torch.randn(N * FP_DIM)
        w   = torch.ones(N)
        kwargs = dict(x=x, edge_index=edge_index, y=y, fps=fps, w=w, edge_attr=None)
        if add_pos:
            kwargs['pos'] = torch.randn(N, 3)
        return Data(**kwargs)

    # ------------------------------------------------------------------ #
    # Test 1: pos exists → pair repr shape, diagonal dist == 0
    # ------------------------------------------------------------------ #
    N1 = 20
    data1 = make_graph(N1, 50, seed=1, add_pos=True)
    data1.batch = torch.zeros(N1, dtype=torch.long)

    model = LH_Direct(
        in_dim=D, hid_dim=512, K=3, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn='relu', num_heads=8, num_atom_types=64,
    )

    pr1 = model._build_full_pair_repr(data1, data1.batch, device)

    assert pr1 is not None,               'Expected pair repr, got None'
    assert pr1.shape == (N1, N1, 8),      f'Shape mismatch: {pr1.shape}'

    # pair_dist diagonal should be 0 → PairReprInit output at (i,i) reflects dist=0
    pos   = data1.pos
    diag_dist = torch.cdist(pos, pos, p=2).diagonal()
    assert torch.allclose(diag_dist, torch.zeros(N1)), 'Diagonal distances not zero'

    # single graph → batch all-zeros → no cross-graph masking → all positions can be nonzero
    assert pr1.abs().sum() > 0, 'pair repr is all-zero (unexpected for single graph)'

    print('Test 1 passed: pair repr shape correct, diagonal distances are zero')

    # ------------------------------------------------------------------ #
    # Test 2: no pos → safe fallback to None
    # ------------------------------------------------------------------ #
    data2 = make_graph(10, 20, seed=2, add_pos=False)
    data2.batch = torch.zeros(10, dtype=torch.long)

    pr2 = model._build_full_pair_repr(data2, data2.batch, device)
    assert pr2 is None, f'Expected None when pos absent, got {pr2}'

    print('Test 2 passed: no pos, _build_full_pair_repr returns None safely')

    # ------------------------------------------------------------------ #
    # Test 3: PyG batching — cross-graph isolation
    # ------------------------------------------------------------------ #
    g0 = make_graph(8,  16, seed=3, add_pos=True)
    g1 = make_graph(12, 24, seed=4, add_pos=True)
    batched = Batch.from_data_list([g0, g1])
    total   = batched.x.size(0)  # 20

    pr3 = model._build_full_pair_repr(batched, batched.batch, device)

    assert pr3 is not None,                  'Expected pair repr for batched input'
    assert pr3.shape == (total, total, 8),   f'Shape mismatch: {pr3.shape}'

    # Cross-graph: node 0 (graph 0) vs node 8 (first node of graph 1)
    cross = pr3[0, 8, :]
    assert torch.allclose(cross, torch.zeros(8)), \
        f'Cross-graph pair not zeroed: {cross}'

    # Intra-graph: node 0 vs node 5 (both in graph 0)
    intra = pr3[0, 5, :]
    assert intra.abs().sum() > 0, 'Intra-graph pair is all-zero (unexpected)'

    print('Test 3 passed: cross-graph zeroed, intra-graph non-zero')

    # ------------------------------------------------------------------ #
    # Test 4: full forward + backward with batched data
    # ------------------------------------------------------------------ #
    # Rebuild batched data with all required fields for LH_Direct.forward
    def make_full_graph(N, E, seed):
        torch.manual_seed(seed)
        return Data(
            x          = torch.randn(N, D),
            edge_index = torch.stack([torch.randint(0, N, (E,)),
                                      torch.randint(0, N, (E,))]),
            y          = torch.randint(0, 2, (N,)).float(),
            fps        = torch.randn(N * FP_DIM),
            w          = torch.ones(N),
            edge_attr  = None,
            pos        = torch.randn(N, 3),
        )

    g0f = make_full_graph(8,  16, seed=5)
    g1f = make_full_graph(12, 24, seed=6)
    batch4 = Batch.from_data_list([g0f, g1f])

    out4 = model(batch4, device)

    assert len(out4) == 4, f'Expected 4 outputs, got {len(out4)}'
    for t in out4:
        assert t.ndim == 2 and t.shape[1] == 512, f'Unexpected shape {t.shape}'

    sum(t.sum() for t in out4).backward()

    assert model.pair_repr_init.gbf.means.weight.grad is not None, \
        'No grad on pair_repr_init.gbf.means'
    assert model.encoder.prop1.gate_proj[0].weight.grad is not None, \
        'No grad on gate_proj'
    assert model.encoder.prop1.pair_update.q_proj.weight.grad is not None, \
        'No grad on pair_update.q_proj'

    print('Test 4 passed: full forward+backward, gradients flow end-to-end')

    print('All tests passed')
