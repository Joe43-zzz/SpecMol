"""T6 integration tests: 5 tests per V2_T6_MINI_SPEC Section 3."""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from LH_Direct_ChebnetII_prop_v2 import ChebnetII_prop_V2
from model_gnn_pre_v2 import ChebNetII_V2, LH_Direct_V2
from pair_to_edge_weight import PairToEdgeWeight
from node_to_pair_update import NodeToPairUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_all_pairs_edge_index(num_nodes, offset=0):
    """All (i,j) pairs with i!=j, global indices starting at offset."""
    idx = torch.arange(num_nodes, dtype=torch.long)
    src = idx.repeat_interleave(num_nodes - 1) + offset
    dst_list = []
    for i in range(num_nodes):
        others = torch.cat([idx[:i], idx[i + 1:]])
        dst_list.append(others + offset)
    dst = torch.cat(dst_list)
    return torch.stack([src, dst], dim=0)


def make_chem_edges(num_nodes, num_bonds, offset=0):
    """Deterministic undirected chem edges."""
    src, dst = [], []
    count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if count >= num_bonds:
                break
            src.extend([i + offset, j + offset])
            dst.extend([j + offset, i + offset])
            count += 1
        if count >= num_bonds:
            break
    return torch.tensor([src, dst], dtype=torch.long)


def make_fake_mol(num_nodes=10, num_bonds=12, feat_dim=93, pair_dim=64):
    """Create a fake molecular Data object with all V2 fields."""
    x = torch.randn(num_nodes, feat_dim)
    edge_index = make_chem_edges(num_nodes, num_bonds)
    pair_edge_index = make_all_pairs_edge_index(num_nodes)
    E_allpairs = pair_edge_index.size(1)
    pair_repr_edge = torch.randn(E_allpairs, pair_dim)
    fps = torch.randn(1489)  # tri fingerprint
    y = torch.tensor([1.0])
    return Data(
        x=x, edge_index=edge_index,
        pair_edge_index=pair_edge_index,
        pair_repr_edge=pair_repr_edge,
        fps=fps, y=y,
    )


# ---------------------------------------------------------------------------
# Test 1: test_t6_forward_runs
# ---------------------------------------------------------------------------

def test_t6_forward_runs():
    """Single molecule, batch=1, K=10, pair_dim=64 — forward doesn't crash."""
    torch.manual_seed(42)
    data = make_fake_mol(num_nodes=10, num_bonds=12, feat_dim=93, pair_dim=64)
    batch = Batch.from_data_list([data])

    model = LH_Direct_V2(
        in_dim=93, hid_dim=512, K=10, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn='relu', type='tri', pair_dim=64, t6=True,
    )
    model.eval()

    with torch.no_grad():
        low, high, spec, fp = model(batch, 'cpu')

    assert low.shape == (1, 512), f"low shape: {low.shape}"
    assert high.shape == (1, 512), f"high shape: {high.shape}"
    assert spec.shape == (1, 512), f"spec shape: {spec.shape}"
    assert fp.shape == (1, 512), f"fp shape: {fp.shape}"
    print("PASS test_t6_forward_runs")


# ---------------------------------------------------------------------------
# Test 2: test_t6_pair_repr_updates
# ---------------------------------------------------------------------------

def test_t6_pair_repr_updates():
    """Verify pair_repr actually changes after K-step propagation."""
    torch.manual_seed(42)
    data = make_fake_mol(num_nodes=10, num_bonds=12, feat_dim=93, pair_dim=64)
    batch = Batch.from_data_list([data])

    prop = ChebnetII_prop_V2(K=10, node_dim=93, pair_dim=64, proj_dim=32)
    pair_to_ew = PairToEdgeWeight(pair_dim=64)

    # Q/K have default Kaiming init; break out_proj zero-init so updates are nonzero
    nn.init.normal_(prop.node_to_pair.out_proj.weight, std=0.02)
    prop.reset_parameters()

    x = batch.x
    edge_index = batch.edge_index
    pair_repr_edge = batch.pair_repr_edge.clone()
    pair_edge_index = batch.pair_edge_index
    batch_vec = batch.batch

    # Initial edge_weight
    edge_weight = pair_to_ew(pair_repr_edge, pair_edge_index, edge_index, batch_vec)

    with torch.no_grad():
        out, pair_repr_final = prop(
            x, edge_index, edge_weight, highpass=True,
            pair_repr_edge=pair_repr_edge.clone(),
            pair_edge_index=pair_edge_index,
            batch=batch_vec,
            pair_to_ew=pair_to_ew,
        )

    diff = (pair_repr_final - pair_repr_edge).norm()
    assert diff > 1e-6, f"pair_repr was NOT updated: diff={diff:.2e}"
    print(f"PASS test_t6_pair_repr_updates (diff norm={diff:.4f})")


# ---------------------------------------------------------------------------
# Test 3: test_t6_zero_init_q_k  (CRITICAL retrofit sanity)
# ---------------------------------------------------------------------------

def test_t6_zero_init_q_k():
    """out_proj=0 → T6 output must be identical to T5 output (retrofit sanity)."""
    torch.manual_seed(42)
    data = make_fake_mol(num_nodes=10, num_bonds=12, feat_dim=93, pair_dim=64)
    batch = Batch.from_data_list([data])

    # T5 model (t6=False)
    model_t5 = LH_Direct_V2(
        in_dim=93, hid_dim=512, K=10, dprate=0.0, dropout=0.0,
        is_bns=False, act_fn='relu', type='tri', pair_dim=64, t6=False,
    )

    # T6 model with same weights
    model_t6 = LH_Direct_V2(
        in_dim=93, hid_dim=512, K=10, dprate=0.0, dropout=0.0,
        is_bns=False, act_fn='relu', type='tri', pair_dim=64, t6=True,
    )

    # Copy all shared weights from T5 to T6
    t5_state = model_t5.state_dict()
    t6_state = model_t6.state_dict()
    for key in t5_state:
        if key in t6_state:
            t6_state[key] = t5_state[key].clone()
    model_t6.load_state_dict(t6_state)

    # Verify out_proj is zero (default init) — this ensures delta=0
    out_norm = model_t6.encoder.prop1.node_to_pair.out_proj.weight.norm().item()
    assert out_norm == 0.0, f"out_proj not zero: {out_norm}"

    model_t5.eval()
    model_t6.eval()

    with torch.no_grad():
        out_t5 = model_t5(batch, 'cpu')
        out_t6 = model_t6(batch, 'cpu')

    for i, (name) in enumerate(['low', 'high', 'spec', 'fp']):
        diff = (out_t6[i] - out_t5[i]).abs().max().item()
        assert diff < 1e-5, f"T6≠T5 for {name}: max diff={diff:.2e}"

    print("PASS test_t6_zero_init_q_k (T6 with out_proj=0 ≡ T5)")


# ---------------------------------------------------------------------------
# Test 4: test_t6_gradient_flow
# ---------------------------------------------------------------------------

def test_t6_gradient_flow():
    """Gradients flow to NodeToPairUpdate Q/K and PairToEdgeWeight MLP."""
    torch.manual_seed(42)
    data = make_fake_mol(num_nodes=10, num_bonds=12, feat_dim=93, pair_dim=64)
    batch = Batch.from_data_list([data])

    model = LH_Direct_V2(
        in_dim=93, hid_dim=512, K=10, dprate=0.0, dropout=0.0,
        is_bns=False, act_fn='relu', type='tri', pair_dim=64, t6=True,
    )
    # Break out_proj zero-init so gradients flow through the full T6 path
    nn.init.normal_(model.encoder.prop1.node_to_pair.out_proj.weight, std=0.02)

    model.train()
    low, high, spec, fp = model(batch, 'cpu')
    loss = low.sum() + high.sum() + spec.sum() + fp.sum()
    loss.backward()

    # Check NodeToPairUpdate gradients
    q_grad = model.encoder.prop1.node_to_pair.q_proj.weight.grad
    k_grad = model.encoder.prop1.node_to_pair.k_proj.weight.grad
    out_grad = model.encoder.prop1.node_to_pair.out_proj.weight.grad
    assert q_grad is not None and q_grad.abs().sum() > 0, "No gradient on q_proj"
    assert k_grad is not None and k_grad.abs().sum() > 0, "No gradient on k_proj"
    assert out_grad is not None and out_grad.abs().sum() > 0, "No gradient on out_proj"

    # Check PairToEdgeWeight MLP gradients
    ew_grad = model.pair_to_edge_weight.mlp[0].weight.grad
    assert ew_grad is not None and ew_grad.abs().sum() > 0, "No gradient on PairToEdgeWeight"

    print("PASS test_t6_gradient_flow")


# ---------------------------------------------------------------------------
# Test 5: test_t6_batched
# ---------------------------------------------------------------------------

def test_t6_batched():
    """PyG Batch with 2 different-size molecules: forward + backward OK."""
    torch.manual_seed(42)
    mol_a = make_fake_mol(num_nodes=8, num_bonds=10, feat_dim=93, pair_dim=64)
    mol_b = make_fake_mol(num_nodes=12, num_bonds=15, feat_dim=93, pair_dim=64)
    batch = Batch.from_data_list([mol_a, mol_b])

    model = LH_Direct_V2(
        in_dim=93, hid_dim=512, K=10, dprate=0.0, dropout=0.0,
        is_bns=False, act_fn='relu', type='tri', pair_dim=64, t6=True,
    )
    # Break out_proj zero-init for gradient flow
    nn.init.normal_(model.encoder.prop1.node_to_pair.out_proj.weight, std=0.02)

    model.train()
    low, high, spec, fp = model(batch, 'cpu')

    assert low.shape == (2, 512), f"low shape: {low.shape}"
    assert high.shape == (2, 512), f"high shape: {high.shape}"

    loss = low.sum() + high.sum() + spec.sum() + fp.sum()
    loss.backward()

    # Verify T6-critical gradients exist
    q_grad = model.encoder.prop1.node_to_pair.q_proj.weight.grad
    k_grad = model.encoder.prop1.node_to_pair.k_proj.weight.grad
    out_grad = model.encoder.prop1.node_to_pair.out_proj.weight.grad
    ew_grad = model.pair_to_edge_weight.mlp[0].weight.grad
    assert q_grad is not None and q_grad.abs().sum() > 0, "No grad on q_proj (batched)"
    assert k_grad is not None and k_grad.abs().sum() > 0, "No grad on k_proj (batched)"
    assert out_grad is not None and out_grad.abs().sum() > 0, "No grad on out_proj (batched)"
    assert ew_grad is not None and ew_grad.abs().sum() > 0, "No grad on PairToEdgeWeight (batched)"

    print("PASS test_t6_batched (2 molecules, forward + backward OK)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_t6_forward_runs()
    test_t6_pair_repr_updates()
    test_t6_zero_init_q_k()
    test_t6_gradient_flow()
    test_t6_batched()
    print("\n" + "=" * 50)
    print("All 5 T6 integration tests PASSED")
    print("=" * 50)
