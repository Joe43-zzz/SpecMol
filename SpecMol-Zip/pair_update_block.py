import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


def _batch_graph_metadata(batch):
    if not hasattr(batch, "batch"):
        raise ValueError("PyG batch is missing batch vector")

    batch_index = batch.batch
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
    if hasattr(batch, "ptr"):
        counts = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        node_offsets = batch.ptr[:-1].tolist()
    else:
        counts = torch.bincount(batch_index, minlength=num_graphs).tolist()
        node_offsets = []
        running = 0
        for count in counts:
            node_offsets.append(running)
            running += count

    edge_index = batch.edge_index
    if edge_index.numel() == 0:
        edge_counts = [0 for _ in counts]
    else:
        edge_batch = batch_index.index_select(0, edge_index[0])
        edge_counts_tensor = torch.bincount(edge_batch, minlength=num_graphs)
        edge_counts = edge_counts_tensor.tolist()

    edge_offsets = []
    running_edges = 0
    for edge_count in edge_counts:
        edge_offsets.append(running_edges)
        running_edges += edge_count

    return counts, node_offsets, edge_counts, edge_offsets


def pyg_batch_to_dense_node_repr(node_repr, batch_index):
    dense_node_repr, valid_mask = to_dense_batch(node_repr, batch_index)
    padding_mask = ~valid_mask
    return dense_node_repr, padding_mask


def dense_pair_bias_from_pyg_batch(batch, pair_attr_name="edge_weight_3d", fill_value=0.0):
    if not hasattr(batch, pair_attr_name):
        raise ValueError(f"PyG batch is missing {pair_attr_name}")

    counts, node_offsets, edge_counts, edge_offsets = _batch_graph_metadata(batch)
    max_nodes = max(counts) if counts else 0
    edge_weight = getattr(batch, pair_attr_name)
    pair_bias = edge_weight.new_full((len(counts), max_nodes, max_nodes), float(fill_value))
    padding_mask = torch.ones((len(counts), max_nodes), dtype=torch.bool, device=edge_weight.device)

    edge_index = batch.edge_index
    for graph_idx, (num_nodes, node_offset, edge_count, edge_offset) in enumerate(
        zip(counts, node_offsets, edge_counts, edge_offsets)
    ):
        padding_mask[graph_idx, :num_nodes] = False
        graph_edge_index = (edge_index[:, edge_offset : edge_offset + edge_count] - node_offset).to(edge_weight.device)
        graph_edge_weight = edge_weight[edge_offset : edge_offset + edge_count]
        if graph_edge_index.numel() > 0:
            pair_bias[graph_idx, graph_edge_index[0], graph_edge_index[1]] = graph_edge_weight

    return pair_bias, padding_mask


def dense_pair_bias_from_sparse_edges(batch, edge_values, fill_value=0.0):
    counts, node_offsets, edge_counts, edge_offsets = _batch_graph_metadata(batch)
    max_nodes = max(counts) if counts else 0

    if edge_values.dim() == 1:
        pair_bias = edge_values.new_full((len(counts), max_nodes, max_nodes), float(fill_value))
    elif edge_values.dim() == 2:
        num_heads = int(edge_values.size(-1))
        pair_bias = edge_values.new_full((len(counts), num_heads, max_nodes, max_nodes), float(fill_value))
    else:
        raise ValueError(f"edge_values must be [E] or [E,H], got {tuple(edge_values.shape)}")

    padding_mask = torch.ones((len(counts), max_nodes), dtype=torch.bool, device=edge_values.device)
    edge_index = batch.edge_index

    for graph_idx, (num_nodes, node_offset, edge_count, edge_offset) in enumerate(
        zip(counts, node_offsets, edge_counts, edge_offsets)
    ):
        padding_mask[graph_idx, :num_nodes] = False
        graph_edge_index = (edge_index[:, edge_offset : edge_offset + edge_count] - node_offset).to(edge_values.device)
        graph_edge_values = edge_values[edge_offset : edge_offset + edge_count]
        if graph_edge_index.numel() == 0:
            continue
        if graph_edge_values.dim() == 1:
            pair_bias[graph_idx, graph_edge_index[0], graph_edge_index[1]] = graph_edge_values
        else:
            pair_bias[graph_idx, :, graph_edge_index[0], graph_edge_index[1]] = graph_edge_values.transpose(0, 1)

    return pair_bias, padding_mask


def dense_pair_repr_from_flat_pyg_batch(batch, pair_attr_name="pair_rep_flat"):
    if not hasattr(batch, pair_attr_name):
        raise ValueError(f"PyG batch is missing {pair_attr_name}")

    counts, _, _, _ = _batch_graph_metadata(batch)
    pair_flat = getattr(batch, pair_attr_name)
    num_heads = int(pair_flat.size(-1))
    max_nodes = max(counts) if counts else 0
    pair_bias = pair_flat.new_zeros((len(counts), num_heads, max_nodes, max_nodes))
    padding_mask = torch.ones((len(counts), max_nodes), dtype=torch.bool, device=pair_flat.device)

    ptr = 0
    for graph_idx, num_nodes in enumerate(counts):
        padding_mask[graph_idx, :num_nodes] = False
        graph_pair_flat = pair_flat[ptr : ptr + (num_nodes * num_nodes)]
        graph_pair = graph_pair_flat.view(num_nodes, num_nodes, num_heads).permute(2, 0, 1).contiguous()
        pair_bias[graph_idx, :, :num_nodes, :num_nodes] = graph_pair
        ptr += num_nodes * num_nodes

    return pair_bias, padding_mask


def sparse_edge_weight_from_dense_pair_bias(batch, dense_pair_bias):
    if dense_pair_bias.dim() != 3:
        raise ValueError(f"dense_pair_bias must be [B,N,N], got {tuple(dense_pair_bias.shape)}")

    counts, node_offsets, edge_counts, edge_offsets = _batch_graph_metadata(batch)
    edge_weights = []
    edge_index = batch.edge_index
    for graph_idx, (node_offset, edge_count, edge_offset) in enumerate(zip(node_offsets, edge_counts, edge_offsets)):
        graph_edge_index = edge_index[:, edge_offset : edge_offset + edge_count]
        if graph_edge_index.numel() > 0:
            local_edge_index = (graph_edge_index - node_offset).to(dense_pair_bias.device)
            edge_weights.append(dense_pair_bias[graph_idx, local_edge_index[0], local_edge_index[1]])
    if edge_weights:
        return torch.cat(edge_weights, dim=0)
    return dense_pair_bias.new_empty((0,))


class PairUpdateLayer(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0, residual_alpha=0.1, clamp_min=-5.0, clamp_max=5.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.pair_update_proj = nn.Linear(num_heads, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.node_norm = nn.LayerNorm(dim)
        self.pair_norm = nn.LayerNorm(num_heads)
        self.residual_alpha = residual_alpha
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _reshape_heads(self, tensor):
        batch_size, num_nodes, _ = tensor.shape
        return tensor.view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _expand_pair_bias(self, pair_bias):
        if pair_bias.dim() == 3:
            return pair_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1), False
        if pair_bias.dim() == 4:
            if pair_bias.size(1) != self.num_heads:
                batch_size, in_heads, num_nodes, _ = pair_bias.shape
                pair_bias = pair_bias.permute(0, 2, 3, 1).contiguous().view(-1, 1, in_heads)
                pair_bias = F.adaptive_avg_pool1d(pair_bias, self.num_heads)
                pair_bias = pair_bias.view(batch_size, num_nodes, num_nodes, self.num_heads).permute(0, 3, 1, 2).contiguous()
            return pair_bias, True
        raise ValueError(f"pair_bias must be [B,N,N] or [B,H,N,N], got {tuple(pair_bias.shape)}")

    def forward(self, node_repr, pair_bias, padding_mask):
        residual_node = node_repr
        pair_bias_heads, original_has_heads = self._expand_pair_bias(pair_bias)
        residual_pair = pair_bias_heads

        q = self._reshape_heads(self.q_proj(node_repr))
        k = self._reshape_heads(self.k_proj(node_repr))
        v = self._reshape_heads(self.v_proj(node_repr))

        qk_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        qk_logits = torch.clamp(qk_logits, min=self.clamp_min, max=self.clamp_max)
        pair_bias_heads = torch.clamp(pair_bias_heads, min=self.clamp_min, max=self.clamp_max)
        attn_logits = qk_logits + pair_bias_heads
        attn_logits = torch.clamp(attn_logits, min=self.clamp_min, max=self.clamp_max)

        if padding_mask is not None:
            attn_logits = attn_logits.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(node_repr.size(0), node_repr.size(1), self.dim)
        node_repr_updated = self.node_norm(residual_node + self.out_proj(context))

        pair_signal = qk_logits.permute(0, 2, 3, 1).contiguous()
        pair_delta = self.pair_update_proj(pair_signal)
        pair_delta = torch.clamp(pair_delta, min=self.clamp_min, max=self.clamp_max)
        pair_bias_updated = residual_pair + self.residual_alpha * pair_delta.permute(0, 3, 1, 2).contiguous()
        pair_bias_updated = self.pair_norm(pair_bias_updated.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        pair_bias_updated = torch.clamp(pair_bias_updated, min=self.clamp_min, max=self.clamp_max)

        if padding_mask is not None:
            valid_pair = (~padding_mask).float()
            valid_pair = valid_pair[:, None, :, None] * valid_pair[:, None, None, :]
            pair_bias_updated = pair_bias_updated * valid_pair

        delta_pair = pair_bias_updated - residual_pair
        if not original_has_heads:
            pair_bias_updated = pair_bias_updated.mean(dim=1)
            delta_pair = delta_pair.mean(dim=1)

        return node_repr_updated, pair_bias_updated, delta_pair


class PairUpdateBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0, layers=2, residual_alpha=0.1, clamp_min=-5.0, clamp_max=5.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PairUpdateLayer(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    residual_alpha=residual_alpha,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, node_repr, pair_bias, padding_mask):
        delta_pair_total = None
        for layer in self.layers:
            node_repr, pair_bias, delta_pair = layer(node_repr, pair_bias, padding_mask)
            delta_pair_total = delta_pair if delta_pair_total is None else (delta_pair_total + delta_pair)
        return node_repr, pair_bias, delta_pair_total
