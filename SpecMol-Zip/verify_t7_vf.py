"""Code-level verification functions (VFs) for the T7 pair-attention.

VeriMAP code VFs — fast, local, CPU-only. The contract: run this BEFORE any
T7 modification (to confirm the baseline is green) and AFTER every T7
modification (to confirm the modification preserved every invariant). A
modification is only allowed to proceed to an HPC experiment VF once all
four code VFs here are green.

  VF-equiv : with the gate shut and pair updates frozen, T7 is bitwise-identical
             to the pure-T5 spectral path. Verifies the safety-floor PLUMBING
             (a modification cannot make T7 worse than T5 when the gate is
             closed); it does NOT exercise the attention kernel — VF-nan and
             VF-grad cover that.
  VF-nan   : one fwd+bwd on a real V2 batch produces no NaN/Inf (loss + grads).
  VF-grad  : gradients reach every T7-specific parameter (q/k/v/out_proj/
             bias_proj/delta_proj + attn_gate).
  VF-batch : per-molecule isolation — perturbing one molecule's node AND pair
             inputs leaves every other molecule's pooled output bit-identical.

Usage:
    python verify_t7_vf.py [--data-root down_task_v2] [--task bace]

Exit code 0 = all VFs pass; 1 = at least one failed.
"""

import argparse
import sys
import traceback

import torch

try:  # PyG >= 2.5 moved DataLoader to torch_geometric.loader
    from torch_geometric.loader import DataLoader
except ImportError:  # pragma: no cover - older PyG
    from torch_geometric.data import DataLoader

from model_gnn_pre_v2 import LH_Direct_V2
from train_utils import load_pyg_inmemory_split, ours_loss, set_seed

DEVICE = "cpu"
HID_DIM = 512  # tracks main_pretrain.py --hid_dim default
K = 10         # tracks main_pretrain.py --K default
# Labels for the LH_Direct_V2._forward_impl return tuple.
OUTPUT_NAMES = ("low_x", "high_x", "spec_x", "x_fp")


def build_t7(in_dim, pair_dim, disable_pair_update=False, seed=0):
    """Build a T7 LH_Direct_V2 with a fixed seed so the test is reproducible."""
    set_seed(seed)
    return LH_Direct_V2(
        in_dim=in_dim, hid_dim=HID_DIM, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type="tri", pair_dim=pair_dim,
        t7=True, t7_disable_pair_update=disable_pair_update,
    ).to(DEVICE)


def load_batch(ds, n_graphs):
    """First `n_graphs` molecules of an already-loaded dataset as one batch."""
    loader = DataLoader(ds, batch_size=n_graphs, shuffle=False)
    return next(iter(loader))


def _fwd_bwd(model, batch):
    """One forward + backward; returns (output_tuple, loss)."""
    out = model(batch, DEVICE)
    loss = ours_loss(*out, alpha=1)
    model.zero_grad()
    loss.backward()
    return out, loss


# --------------------------------------------------------------------------- #
# VF-equiv
# --------------------------------------------------------------------------- #
def vf_equiv(ds, in_dim, pair_dim, tol=1e-4):
    """T7 with the gate shut + pair updates frozen must equal the pure-T5 path.

    Run A: T7 path active, attn_gate forced to -1e9 so sigmoid(gate)=0 kills the
           node-side attn_acc injection; t7_disable_pair_update keeps pair_repr
           static so the per-step Laplacian rebuild reproduces the T5 Laplacian.
    Run B: prop.t7_enabled flipped off -> the prop forward takes the
           dynamic=False pure-T5 branch on the SAME shared parameters.
    """
    model = build_t7(in_dim, pair_dim, disable_pair_update=True)
    model.eval()
    batch = load_batch(ds, n_graphs=6)
    prop = model.encoder.prop1
    # If t7_enabled were renamed/removed, Run B would silently stay on the T7
    # path and the test would pass meaninglessly — fail loudly instead.
    assert hasattr(prop, "t7_enabled"), \
        "prop.t7_enabled missing — VF-equiv Run B would silently no-op"

    saved_gate = prop.pair_attn.attn_gate.data.clone()
    try:
        with torch.no_grad():
            prop.pair_attn.attn_gate.data.fill_(-1e9)
            out_a = model(batch, DEVICE)
        prop.pair_attn.attn_gate.data.copy_(saved_gate)
        prop.t7_enabled = False
        with torch.no_grad():
            out_b = model(batch, DEVICE)
    finally:
        prop.pair_attn.attn_gate.data.copy_(saved_gate)
        prop.t7_enabled = True

    diffs = {n: (a - b).abs().max().item()
             for n, a, b in zip(OUTPUT_NAMES, out_a, out_b)}
    max_diff = max(diffs.values())
    ok = max_diff < tol
    print(f"  VF-equiv : max|A-B| = {max_diff:.2e}  (tol {tol})  {diffs}")
    print(f"             {'PASS' if ok else 'FAIL'} — T7 reduces to T5 safety floor")
    return ok


# --------------------------------------------------------------------------- #
# VF-nan
# --------------------------------------------------------------------------- #
def vf_nan(ds, in_dim, pair_dim):
    """One fwd+bwd on a real V2 batch must produce no NaN/Inf anywhere."""
    model = build_t7(in_dim, pair_dim)
    model.train()
    batch = load_batch(ds, n_graphs=8)
    out, loss = _fwd_bwd(model, batch)

    loss_ok = torch.isfinite(loss).all().item()
    out_ok = all(torch.isfinite(t).all().item() for t in out)
    grad_ok = all(
        torch.isfinite(p.grad).all().item()
        for p in model.parameters() if p.grad is not None
    )
    ok = loss_ok and out_ok and grad_ok
    print(f"  VF-nan   : loss={loss.item():.4f} finite={loss_ok} "
          f"outputs_finite={out_ok} grads_finite={grad_ok}")
    print(f"             {'PASS' if ok else 'FAIL'} — fwd+bwd numerically clean")
    return ok


# --------------------------------------------------------------------------- #
# VF-grad
# --------------------------------------------------------------------------- #
def vf_grad(ds, in_dim, pair_dim):
    """Gradients must reach every T7-specific parameter group.

    Runs the real T7 config (disable_pair_update=False). This is load-bearing,
    not a convenience: under disable_pair_update=True the updated pair_repr is
    discarded, so delta_proj's output reaches no loss term and is structurally
    dead — the grad test must use the config where every T7 param is on the
    graph.
    """
    model = build_t7(in_dim, pair_dim)  # disable_pair_update=False — see docstring
    model.train()
    batch = load_batch(ds, n_graphs=8)
    _fwd_bwd(model, batch)

    pa = model.encoder.prop1.pair_attn
    params = {
        "t7.q":          pa.q.weight,
        "t7.k":          pa.k.weight,
        "t7.v":          pa.v.weight,
        "t7.out_proj":   pa.out_proj.weight,
        "t7.bias_proj":  pa.bias_proj.weight,
        "t7.delta_proj": pa.delta_proj.weight,
        "t7.attn_gate":  pa.attn_gate,
    }
    status = {
        name: (p.grad is not None and torch.isfinite(p.grad).all().item()
               and p.grad.abs().sum().item() > 0.0)
        for name, p in params.items()
    }
    ok = all(status.values())
    dead = [k for k, v in status.items() if not v]
    print(f"  VF-grad  : {status}")
    print(f"             {'PASS' if ok else 'FAIL — dead: ' + str(dead)} "
          f"— all T7 params receive gradient")
    return ok


# --------------------------------------------------------------------------- #
# VF-batch
# --------------------------------------------------------------------------- #
def vf_batch(ds, in_dim, pair_dim, tol=1e-6):
    """Perturbing one molecule must leave every other molecule's output exact.

    Perturbs BOTH input channels of the last molecule — node features `x` and
    pair representations `pair_repr_edge` — because cross-molecule attention
    leakage could enter either through the Q/K/V path (from x) or the
    bias/delta path (from pair_repr_edge).
    """
    model = build_t7(in_dim, pair_dim)
    model.eval()
    batch = load_batch(ds, n_graphs=4)

    with torch.no_grad():
        out_base = model(batch, DEVICE)

        batch2 = batch.clone()
        last = int(batch2.batch.max().item())
        node_mask = batch2.batch == last
        batch2.x[node_mask] += torch.randn_like(batch2.x[node_mask])
        # pair_edge_index is intra-molecule, so the source node's graph id
        # identifies which pair rows belong to the last molecule.
        pair_mask = batch2.batch[batch2.pair_edge_index[0]] == last
        batch2.pair_repr_edge[pair_mask] += torch.randn_like(
            batch2.pair_repr_edge[pair_mask]
        )
        out_pert = model(batch2, DEVICE)

    # Rows 0..last-1 are the untouched molecules; each must be bit-identical.
    diffs = {n: (a[:last] - b[:last]).abs().max().item()
             for n, a, b in zip(OUTPUT_NAMES, out_base, out_pert)}
    max_diff = max(diffs.values())
    ok = max_diff < tol
    print(f"  VF-batch : max drift of mols 0..{last - 1} after perturbing "
          f"mol-{last} = {max_diff:.2e}  (tol {tol})  {diffs}")
    print(f"             {'PASS' if ok else 'FAIL'} — no cross-molecule leakage")
    return ok


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="T7 code-level verification functions")
    parser.add_argument(
        "--data-root", default="down_task_v2",
        help="V2-format dataset root with processed/<task>_all.pt (default "
             "down_task_v2 = local pre-B4 BACE V2 archive; fine for code VFs, "
             "which test structure not accuracy)",
    )
    parser.add_argument("--task", default="bace")
    args = parser.parse_args()

    print("=" * 68)
    print(f"T7 code VFs — data_root={args.data_root} task={args.task}")
    print("=" * 68)

    # Load the dataset once; the V2 'all' split is large (hundreds of MB) and
    # each VF only needs a handful of molecules from it.
    ds = load_pyg_inmemory_split(args.data_root, args.task, "all")
    sample = ds[0]
    in_dim = sample.x.size(1)
    pair_dim = sample.pair_repr_edge.size(1)
    print(f"dataset: {len(ds)} graphs  in_dim={in_dim}  pair_dim={pair_dim}\n")

    results = {}
    for name, fn in [
        ("VF-equiv", vf_equiv),
        ("VF-nan",   vf_nan),
        ("VF-grad",  vf_grad),
        ("VF-batch", vf_batch),
    ]:
        try:
            results[name] = fn(ds, in_dim, pair_dim)
        except Exception as exc:  # a crash IS a verification failure
            traceback.print_exc()
            print(f"  {name} : FAIL — raised {type(exc).__name__}: {exc}")
            results[name] = False
        print("-" * 68)

    n_pass = sum(results.values())
    summary = {k: ("PASS" if v else "FAIL") for k, v in results.items()}
    print(f"SUMMARY: {n_pass}/{len(results)} code VFs passed  -> {summary}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
