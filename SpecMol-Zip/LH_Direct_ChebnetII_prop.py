import math
import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
import torch.nn.functional as F
from torch.nn import Parameter

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


class ChebnetII_prop(MessagePassing):
    def __init__(self, K, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)

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

    def reset_parameters(self):
        self.temp_low.data.fill_(2.0 / self.K)
        self.temp_high.data.fill_(2.0 / self.K)

    def forward(self, x, edge_index, edge_weight=None, highpass=True):
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

        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)
        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp_low={}, temp_high={})'.format(
            self.__class__.__name__, self.K, self.temp_low, self.temp_high,
        )

