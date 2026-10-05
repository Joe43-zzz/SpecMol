import torch
import torch.nn as nn


def _get_activation(name):
    if name == 'gelu':
        return nn.GELU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f'Unknown activation: {name}')


class PairToNodeGate(nn.Module):
    """
    Gate messages in ChebNet-II propagation using pair representations.

    For each edge (i→j) in the sparse edge_index, looks up the pair repr
    at [i, j], projects it to a scalar gate in (0, 1), and multiplies the
    message elementwise.  This lets 3-D spatial information modulate how
    much information flows along each bond.
    """

    def __init__(self, num_heads, node_dim, activation_fn='gelu'):
        super().__init__()
        # Two-layer MLP: num_heads → num_heads → 1
        # Sigmoid is applied in forward (not here) so raw logits stay inspectable.
        self.proj = nn.Sequential(
            nn.Linear(num_heads, num_heads),
            _get_activation(activation_fn),
            nn.Linear(num_heads, 1),
        )

    def forward(self, message, pair_repr, edge_index):
        """
        Args:
            message   : [E, D]          sparse edge messages
            pair_repr : [N, N, H]       dense pair representations
            edge_index: [2, E]          sparse edge indices

        Returns:
            gated_message: [E, D]  message * gate
        """
        src, dst = edge_index[0], edge_index[1]          # [E]
        pair_feat = pair_repr[src, dst]                  # [E, H]
        gate = torch.sigmoid(self.proj(pair_feat))       # [E, 1]
        return message * gate                            # [E, D]


if __name__ == '__main__':
    torch.manual_seed(0)

    # ------------------------------------------------------------------ #
    # Test 1: basic shape and value-range check
    # ------------------------------------------------------------------ #
    total_nodes, E, D, H = 20, 50, 64, 8

    message   = torch.randn(E, D)
    pair_repr = torch.randn(total_nodes, total_nodes, H)
    edge_index = torch.randint(0, total_nodes, (2, E))

    model = PairToNodeGate(num_heads=H, node_dim=D, activation_fn='gelu')
    out = model(message, pair_repr, edge_index)

    assert out.shape == (E, D), f'Shape mismatch: {out.shape}'

    # gate is in (0, 1), so |gated_msg[i]| <= |message[i]|
    assert torch.all(out.abs() <= message.abs() + 1e-6), \
        'Gate is outside (0,1): output magnitude exceeds input'

    print('Test 1 passed: shape and value-range OK')

    # ------------------------------------------------------------------ #
    # Test 2: gradient flow
    # ------------------------------------------------------------------ #
    message2   = torch.randn(E, D, requires_grad=True)
    pair_repr2 = torch.randn(total_nodes, total_nodes, H, requires_grad=True)

    out2 = model(message2, pair_repr2, edge_index)
    out2.sum().backward()

    assert pair_repr2.grad is not None, 'No gradient on pair_repr'
    assert message2.grad is not None,   'No gradient on message'

    print('Test 2 passed: gradients flow through pair_repr and message')

    # ------------------------------------------------------------------ #
    # Test 3: gate values are strictly in (0, 1)
    # ------------------------------------------------------------------ #
    src, dst = edge_index[0], edge_index[1]
    pair_feat3 = pair_repr[src, dst]                        # [E, H]
    gate3 = torch.sigmoid(model.proj(pair_feat3))           # [E, 1]

    assert gate3.min().item() > 0.0, f'Gate min={gate3.min().item()} not > 0'
    assert gate3.max().item() < 1.0, f'Gate max={gate3.max().item()} not < 1'

    print(f'Test 3 passed: gate range ({gate3.min().item():.4f}, {gate3.max().item():.4f}) in (0, 1)')

    print('All tests passed')
