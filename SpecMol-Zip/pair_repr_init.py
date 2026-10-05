import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, dist, edge_type):
        # dist: [B, N, N], edge_type: [B, N, N]
        mul = self.mul(edge_type).type_as(dist)   # [B, N, N, 1]
        bias = self.bias(edge_type).type_as(dist)  # [B, N, N, 1]
        x = mul * dist.unsqueeze(-1) + bias        # [B, N, N, 1]
        x = x.expand(-1, -1, -1, self.K)           # [B, N, N, K]
        mean = self.means.weight.float().view(-1)  # [K]
        std = self.stds.weight.float().view(-1).abs() + 1e-5  # [K]
        return gaussian(x.float(), mean, std).type_as(self.means.weight)  # [B, N, N, K]


def _get_activation(name):
    if name == 'gelu':
        return nn.GELU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f'Unknown activation: {name}')


class PairReprInit(nn.Module):
    def __init__(self, K=128, num_heads=8, edge_types=512, activation_fn='gelu'):
        super().__init__()
        self.gbf = GaussianLayer(K=K, edge_types=edge_types)
        self.gbf_proj = nn.Sequential(
            nn.Linear(K, K),
            _get_activation(activation_fn),
            nn.Linear(K, num_heads),
        )

    def forward(self, dist, edge_type):
        # dist: [B, N, N], edge_type: [B, N, N]
        x = self.gbf(dist, edge_type)   # [B, N, N, K]
        x = self.gbf_proj(x)            # [B, N, N, num_heads]
        return x


if __name__ == '__main__':
    torch.manual_seed(42)

    B, N, num_atom_types = 2, 10, 16
    num_heads = 8
    edge_types = num_atom_types * num_atom_types  # 256, well within default 512

    # Random symmetric distance matrix, diagonal = 0
    raw = torch.rand(B, N, N)
    dist = (raw + raw.transpose(-1, -2)) / 2
    dist.diagonal(dim1=-2, dim2=-1).zero_()

    # Random edge types: atom_type_i * num_atom_types + atom_type_j
    atom_i = torch.randint(0, num_atom_types, (B, N, N))
    atom_j = torch.randint(0, num_atom_types, (B, N, N))
    edge_type = atom_i * num_atom_types + atom_j

    model = PairReprInit(K=128, num_heads=num_heads, edge_types=512, activation_fn='gelu')

    out = model(dist, edge_type)
    assert out.shape == (B, N, N, num_heads), f'Shape mismatch: {out.shape}'

    out.sum().backward()
    assert model.gbf.means.weight.grad is not None, 'No gradient on gbf.means.weight'

    print('All tests passed')
