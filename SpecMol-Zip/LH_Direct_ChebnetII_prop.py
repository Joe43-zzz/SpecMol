import math
import torch
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
import torch.nn.functional as F
from torch.nn import Parameter

from node_to_pair_update import NodeToPairUpdate


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


def presum_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx - 1] + h[idx - 1]
    return temp


def preminus_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx - 1] - h[idx - 1]
    return temp


def gamma_h(slides, beta_a_h, beta_b_h):
    length = len(slides)
    gamma_h = torch.zeros(length)
    for j in range(length):
        term1 = 1 / 2 * F.relu(beta_b_h) * (1 + torch.cos(((1 + j + slides[j]) / length) * torch.pi))
        gamma_h[j] = F.relu(beta_a_h) + term1
    return gamma_h


def gamma_l(slides, beta_a_l, beta_b_l):
    length = len(slides)
    gamma_l = torch.zeros(length)
    for j in range(length):
        term1 = 1 / 2 * F.relu(beta_b_l) * (1 + torch.cos(((1 + j + slides[j]) / length) * torch.pi))
        gamma_l[j] = F.relu(beta_a_l) - term1
    return gamma_l


def reverse_tensor(h):
    temp = torch.zeros_like(h)
    length = len(temp)
    for idx in range(0, length):
        temp[idx] = h[length - 1 - idx]
    return temp


class ChebnetII_prop(MessagePassing):
    def __init__(self, K, num_heads=None, node_dim=None, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.node_axis = -2
        self.num_heads = num_heads

        # Chebyshev filter parameters (unchanged from original)
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

        # Gate projection + pair update (only when pair repr is used)
        if num_heads is not None and node_dim is not None:
            self.gate_proj = nn.Sequential(
                nn.Linear(num_heads, num_heads),
                nn.GELU(),
                nn.Linear(num_heads, 1),
            )
            self.pair_update = NodeToPairUpdate(node_dim, proj_dim=32, num_heads=num_heads)
        else:
            self.gate_proj  = None
            self.pair_update = None

    def reset_parameters(self):
        self.temp_low.data.fill_(2.0 / self.K)
        self.temp_high.data.fill_(2.0 / self.K)

    def forward(self, x, edge_index, edge_weight=None, highpass=True, pair_repr=None, batch=None):
        # === Filter coefficient computation (unchanged) ===
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

        # === Laplacian construction (unaffected by pair repr) ===
        num_nodes = x.size(self.node_axis)
        # Only scalar (1-D) edge weights are valid for the Laplacian.
        # 2-D bond-feature tensors (e.g. edge_attr [E, 11]) must be ignored.
        lap_edge_weight = (
            edge_weight
            if edge_weight is not None and edge_weight.dim() == 1
            else None
        )
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight=lap_edge_weight, normalization='sym',
            dtype=x.dtype, num_nodes=num_nodes,
        )
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0, num_nodes=num_nodes,
        )

        # === Pre-compute gate from pair repr ===
        def compute_gate(pair_repr, edge_index_tilde):
            if pair_repr is None or self.gate_proj is None:
                return None
            src, dst = edge_index_tilde[0], edge_index_tilde[1]
            pair_feat = pair_repr[src, dst]                      # [E_tilde, H]
            return torch.sigmoid(self.gate_proj(pair_feat))      # [E_tilde, 1]

        # === Chebyshev recurrence ===
        Tx_0 = x
        gate = compute_gate(pair_repr, edge_index_tilde)
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, gate=gate, size=None)

        if pair_repr is not None and self.pair_update is not None:
            pair_repr = self.pair_update(Tx_1, pair_repr, batch)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            gate = compute_gate(pair_repr, edge_index_tilde)
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, gate=gate, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2

            if pair_repr is not None and self.pair_update is not None:
                before_norm = pair_repr.norm().item()
                pair_repr = self.pair_update(Tx_2, pair_repr, batch)
                after_norm = pair_repr.norm().item()
                diff = after_norm - before_norm
                print(f"    [K step {i}] pair_repr norm: {before_norm:.4f} -> {after_norm:.4f} (diff={diff:.4f})")

            Tx_0, Tx_1 = Tx_1, Tx_2

        return out, pair_repr

    def message(self, x_j, norm, gate=None):
        msg = norm.view(-1, 1) * x_j          # [E_tilde, D]
        if gate is not None:
            msg = msg * gate                   # gate [E_tilde, 1] broadcasts over D
        return msg

    def __repr__(self):
        return '{}(K={}, temp_low={}, temp_high={})'.format(
            self.__class__.__name__, self.K, self.temp_low, self.temp_high,
        )


if __name__ == '__main__':
    import torch
    from torch_geometric.data import Data, Batch

    torch.manual_seed(0)

    K, D, H = 3, 64, 8

    # ------------------------------------------------------------------
    # Test 1: no pair repr — output shape correct, second return is None
    # ------------------------------------------------------------------
    N, E = 10, 20
    x1         = torch.randn(N, D)
    src        = torch.randint(0, N, (E,))
    dst        = torch.randint(0, N, (E,))
    edge_index1 = torch.stack([src, dst])

    model_no_pair = ChebnetII_prop(K=K)
    out1, pr1 = model_no_pair(x1, edge_index1, pair_repr=None, batch=None)

    assert out1.shape == (N, D), f'Shape mismatch: {out1.shape}'
    assert pr1 is None, 'Expected None when no pair repr given'

    print('Test 1 passed: no pair repr, output shape correct, second return is None')

    # ------------------------------------------------------------------
    # Test 2: with pair repr — forward + backward
    # ------------------------------------------------------------------
    model_pair = ChebnetII_prop(K=K, num_heads=H, node_dim=D)

    x2          = torch.randn(N, D)
    pair_repr2  = torch.randn(N, N, H, requires_grad=True)
    batch2      = torch.zeros(N, dtype=torch.long)
    edge_index2 = torch.stack([
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
    ])

    out2, pr2 = model_pair(x2, edge_index2, pair_repr=pair_repr2, batch=batch2)

    assert out2.shape == (N, D),    f'Node output shape mismatch: {out2.shape}'
    assert pr2.shape  == (N, N, H), f'Pair repr shape mismatch: {pr2.shape}'

    (out2.sum() + pr2.sum()).backward()
    assert pair_repr2.grad is not None,                    'No grad on pair_repr2'
    assert model_pair.gate_proj[0].weight.grad is not None, 'No grad on gate_proj'
    assert model_pair.pair_update.q_proj.weight.grad is not None, 'No grad on pair_update.q_proj'

    print('Test 2 passed: pair repr forward+backward, shapes and gradients correct')

    # ------------------------------------------------------------------
    # Test 3: PyG batching — two graphs, 5 and 7 nodes
    # ------------------------------------------------------------------
    def make_graph(n, d, seed):
        torch.manual_seed(seed)
        ei = torch.stack([torch.randint(0, n, (n * 2,)), torch.randint(0, n, (n * 2,))])
        return Data(x=torch.randn(n, d), edge_index=ei)

    g0 = make_graph(5, D, seed=1)
    g1 = make_graph(7, D, seed=2)
    batch_data = Batch.from_data_list([g0, g1])

    total_nodes = batch_data.x.size(0)   # 12
    pair_repr3  = torch.randn(total_nodes, total_nodes, H, requires_grad=True)
    batch_vec   = batch_data.batch        # [12]

    model_batch = ChebnetII_prop(K=K, num_heads=H, node_dim=D)
    out3, pr3 = model_batch(
        batch_data.x,
        batch_data.edge_index,
        pair_repr=pair_repr3,
        batch=batch_vec,
    )

    assert out3.shape == (total_nodes, D),           f'Batched node output shape: {out3.shape}'
    assert pr3.shape  == (total_nodes, total_nodes, H), f'Batched pair repr shape: {pr3.shape}'

    out3.sum().backward()
    assert pair_repr3.grad is not None, 'No grad on pair_repr3 in batched run'

    print('Test 3 passed: PyG batching, shapes correct and gradients flow')

    print('All tests passed')
