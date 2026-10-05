import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data


REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "SpecMol-Zip"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from LH_Direct_ChebnetII_prop import ChebnetII_prop
from model_gnn_pre import ChebNetII, LH_Direct


torch.manual_seed(42)


def assert_finite(tensor, name):
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"


def build_all_pairs_edge_index(num_nodes):
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    src = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, dtype=torch.long).repeat(num_nodes)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def make_synthetic_data(
    num_nodes,
    pair_repr_dim=64,
    node_dim=93,
    fp_dim=1489,
    edge_attr_dim=11,
    include_pair_repr_full=True,
    include_batch=True,
):
    edge_index = build_all_pairs_edge_index(num_nodes)
    num_edges = edge_index.size(1)
    pair_repr_full = torch.randn(num_nodes, num_nodes, pair_repr_dim, dtype=torch.float32)
    pair_repr_edge = pair_repr_full[edge_index[0], edge_index[1]].contiguous()

    data_kwargs = dict(
        x=torch.randn(num_nodes, node_dim, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.randn(num_edges, edge_attr_dim, dtype=torch.float32),
        pair_repr_edge=pair_repr_edge,
        pair_repr_dim=pair_repr_dim,
        edge_weight_3d=torch.rand(num_edges, dtype=torch.float32) + 0.1,
        y=torch.randn(1, dtype=torch.float32),
        fps=torch.randn(fp_dim, dtype=torch.float32),
        w=torch.ones(1, dtype=torch.float32),
    )
    if include_pair_repr_full:
        data_kwargs["pair_repr_full"] = pair_repr_full
    if include_batch:
        data_kwargs["batch"] = torch.zeros(num_nodes, dtype=torch.long)
    data = Data(**data_kwargs)
    return data


def test_1_data_object_construction():
    data = make_synthetic_data(num_nodes=5)
    expected_edges = 5 * 4

    assert data.x.shape == (5, 93), f"x shape mismatch: {tuple(data.x.shape)}"
    assert data.edge_index.shape == (2, expected_edges), f"edge_index shape mismatch: {tuple(data.edge_index.shape)}"
    assert data.edge_attr.shape == (expected_edges, 11), f"edge_attr shape mismatch: {tuple(data.edge_attr.shape)}"
    assert data.pair_repr_full.shape == (5, 5, 64), f"pair_repr_full shape mismatch: {tuple(data.pair_repr_full.shape)}"
    assert data.pair_repr_edge.shape == (expected_edges, 64), f"pair_repr_edge shape mismatch: {tuple(data.pair_repr_edge.shape)}"
    assert data.edge_weight_3d.shape == (expected_edges,), f"edge_weight_3d shape mismatch: {tuple(data.edge_weight_3d.shape)}"
    assert data.y.shape == (1,), f"y shape mismatch: {tuple(data.y.shape)}"
    assert data.fps.shape == (1489,), f"fps shape mismatch: {tuple(data.fps.shape)}"
    assert data.w.shape == (1,), f"w shape mismatch: {tuple(data.w.shape)}"
    assert data.batch.shape == (5,), f"batch shape mismatch: {tuple(data.batch.shape)}"
    assert data.pair_repr_dim == 64, f"pair_repr_dim mismatch: {data.pair_repr_dim}"

    expected_pair_repr_edge = data.pair_repr_full[data.edge_index[0], data.edge_index[1]]
    assert torch.allclose(data.pair_repr_edge, expected_pair_repr_edge), "pair_repr_edge does not match flattened pair_repr_full"
    assert not torch.any(data.edge_index[0] == data.edge_index[1]), "edge_index contains self-loops"
    assert data.edge_index.size(1) == 5 * (5 - 1), "edge count mismatch"

    print("TEST 1 PASSED: Data object construction")
    return data


def test_2_chebnetii_prop_forward_with_pair_repr(data):
    prop = ChebnetII_prop(K=3, pair_repr_dim=64, node_dim=93)
    prop.train()
    input_pair_repr = data.pair_repr_edge.clone()
    out, updated_pair_repr = prop(
        data.x,
        data.edge_index,
        pair_repr=input_pair_repr,
        edge_index_for_pair=data.edge_index,
        highpass=True,
    )

    assert out.shape == (5, 93), f"out shape mismatch: {tuple(out.shape)}"
    assert updated_pair_repr.shape == (20, 64), f"updated_pair_repr shape mismatch: {tuple(updated_pair_repr.shape)}"
    assert not torch.allclose(updated_pair_repr, data.pair_repr_edge), "updated_pair_repr is identical to input pair_repr_edge"
    assert_finite(out, "ChebnetII_prop out")
    assert_finite(updated_pair_repr, "ChebnetII_prop updated_pair_repr")

    print("TEST 2 PASSED: ChebnetII_prop forward with pair repr")


def test_3_chebnetii_prop_backward_compatibility_no_pair_repr(data):
    prop = ChebnetII_prop(K=3, pair_repr_dim=64, node_dim=93)
    prop.train()
    out, none_pair_repr = prop(
        data.x,
        data.edge_index,
        pair_repr=None,
        highpass=False,
    )

    assert out.shape == (5, 93), f"out shape mismatch: {tuple(out.shape)}"
    assert none_pair_repr is None, "Expected updated pair repr to be None"
    assert_finite(out, "ChebnetII_prop out without pair_repr")

    print("TEST 3 PASSED: ChebnetII_prop backward compatibility (no pair repr)")


def test_4_chebnetii_forward(data):
    model = ChebNetII(num_features=93, hidden=512, K=3, pair_repr_dim=64, node_dim=93)
    model.train()
    out, updated_pair_repr = model(
        data.x,
        data.edge_index,
        pair_repr=data.pair_repr_edge.clone(),
        edge_index_for_pair=data.edge_index,
        highpass=True,
    )

    assert out.shape == (5, 512), f"out shape mismatch: {tuple(out.shape)}"
    assert updated_pair_repr.shape == (20, 64), f"updated_pair_repr shape mismatch: {tuple(updated_pair_repr.shape)}"
    assert_finite(out, "ChebNetII out")
    assert_finite(updated_pair_repr, "ChebNetII updated_pair_repr")

    print("TEST 4 PASSED: ChebNetII forward")


def test_5_lh_direct_forward(data):
    model = LH_Direct(
        in_dim=93,
        hid_dim=512,
        K=3,
        dprate=0.5,
        dropout=0.0,
        is_bns=False,
        act_fn="relu",
        type="tri",
        pair_repr_dim=64,
    )
    model.train()
    low_x_mean, high_x_mean, spec_x_mean, x_fp = model(data, device="cpu")

    for tensor_name, tensor in (
        ("low_x_mean", low_x_mean),
        ("high_x_mean", high_x_mean),
        ("spec_x_mean", spec_x_mean),
        ("x_fp", x_fp),
    ):
        assert tensor.shape == (1, 512), f"{tensor_name} shape mismatch: {tuple(tensor.shape)}"
        assert_finite(tensor, tensor_name)

    print("TEST 5 PASSED: LH_Direct forward")
    return model, (low_x_mean, high_x_mean, spec_x_mean, x_fp)


def assert_nonzero_grad(param, name):
    assert param.grad is not None, f"{name}.grad is None"
    assert torch.count_nonzero(param.grad).item() > 0, f"{name}.grad is all zeros"


def test_6_backward_pass_gradients_flow(model, outputs):
    low_x_mean, high_x_mean, spec_x_mean, x_fp = outputs
    model.zero_grad(set_to_none=True)
    loss = low_x_mean.sum() + high_x_mean.sum() + spec_x_mean.sum() + x_fp.sum()
    loss.backward()

    prop = model.encoder.prop1
    assert_nonzero_grad(prop.pair_to_weight.weight, "pair_to_weight.weight")
    assert_nonzero_grad(prop.pair_to_message_weight.weight, "pair_to_message_weight.weight")
    assert_nonzero_grad(prop.node_to_pair_q.weight, "node_to_pair_q.weight")
    assert_nonzero_grad(prop.node_to_pair_k.weight, "node_to_pair_k.weight")

    print("TEST 6 PASSED: Backward pass - gradients flow to all pair repr parameters")


def test_7_pyg_batching():
    graph_sizes = [3, 5, 7]
    data_list = [
        make_synthetic_data(num_nodes=size, include_pair_repr_full=False, include_batch=False)
        for size in graph_sizes
    ]
    batch = Batch.from_data_list(data_list)

    expected_edges = sum(size * (size - 1) for size in graph_sizes)
    assert batch.pair_repr_edge.shape == (expected_edges, 64), (
        f"batched pair_repr_edge shape mismatch: {tuple(batch.pair_repr_edge.shape)}"
    )
    assert batch.edge_index.shape == (2, expected_edges), f"batched edge_index shape mismatch: {tuple(batch.edge_index.shape)}"

    model = LH_Direct(
        in_dim=93,
        hid_dim=512,
        K=3,
        dprate=0.5,
        dropout=0.0,
        is_bns=False,
        act_fn="relu",
        type="tri",
        pair_repr_dim=64,
    )
    model.train()
    low_x_mean, high_x_mean, spec_x_mean, x_fp = model(batch, device="cpu")

    for tensor_name, tensor in (
        ("low_x_mean", low_x_mean),
        ("high_x_mean", high_x_mean),
        ("spec_x_mean", spec_x_mean),
        ("x_fp", x_fp),
    ):
        assert tensor.shape == (3, 512), f"{tensor_name} shape mismatch: {tuple(tensor.shape)}"
        assert_finite(tensor, tensor_name)

    print("TEST 7 PASSED: PyG batching with variable-size graphs")


def main():
    failures = []
    shared_data = None
    shared_model = None
    shared_outputs = None

    tests = [
        ("TEST 1", lambda: test_1_data_object_construction()),
        ("TEST 2", lambda: test_2_chebnetii_prop_forward_with_pair_repr(shared_data)),
        ("TEST 3", lambda: test_3_chebnetii_prop_backward_compatibility_no_pair_repr(shared_data)),
        ("TEST 4", lambda: test_4_chebnetii_forward(shared_data)),
        ("TEST 5", lambda: test_5_lh_direct_forward(shared_data)),
        ("TEST 6", lambda: test_6_backward_pass_gradients_flow(shared_model, shared_outputs)),
        ("TEST 7", test_7_pyg_batching),
    ]

    for test_name, test_fn in tests:
        try:
            result = test_fn()
            if test_name == "TEST 1":
                shared_data = result
            elif test_name == "TEST 5":
                shared_model, shared_outputs = result
        except AssertionError as exc:
            print(f"{test_name} FAILED: {exc}")
            failures.append(f"{test_name}: {exc}")
        except Exception as exc:
            print(f"{test_name} FAILED: {exc}")
            failures.append(f"{test_name}: {exc}")

    if failures:
        print("FAILED TESTS:")
        for failure in failures:
            print(failure)
        raise AssertionError("Smoke tests failed")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
