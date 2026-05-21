"""Code-level verification functions (VFs) for the T7 pair-attention.

VeriMAP code VFs — fast, local, CPU-only. The contract: run this BEFORE any
T7 modification (to confirm the baseline is green) and AFTER every T7
modification (to confirm the modification preserved every invariant). A
modification is only allowed to proceed to an HPC experiment VF once all
four code VFs here are green.

  VF-equiv : T7 with attn_gate -> -inf and t7_disable_pair_update=True reduces
             EXACTLY to the pure-T5 spectral path. This is the "provably >= T5"
             safety floor — every modification must keep it.
  VF-nan   : one fwd+bwd on a real V2 batch produces no NaN/Inf (loss + grads).
  VF-grad  : gradients reach every T7-specific parameter (q/k/v/out_proj/
             bias_proj/delta_proj + attn_gate).
  VF-batch : per-molecule isolation — perturbing one molecule leaves the other
             molecules' pooled outputs bit-identical (no cross-molecule
             attention leakage).

Usage:
    python verify_t7_vf.py [--data-root down_task_v2] [--task bace]

Exit code 0 = all VFs pass; 1 = at least one failed.
"""

import argparse
import sys

import torch

try:  # PyG >= 2.5 moved DataLoader to torch_geometric.loader
    from torch_geometric.loader import DataLoader
except ImportError:  # pragma: no cover - older PyG
    from torch_geometric.data import DataLoader

from model_gnn_pre_v2 import LH_Direct_V2
from train_utils import load_pyg_inmemory_split, ours_loss, set_seed

DEVICE = "cpu"
IN_DIM = 93
PAIR_DIM = 64
HID_DIM = 512
K = 10


def build_t7(disable_pair_update=False, seed=0):
    """Build a T7 LH_Direct_V2 with a fixed seed so the test is reproducible."""
    set_seed(seed)
    return LH_Direct_V2(
        in_dim=IN_DIM, hid_dim=HID_DIM, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type="tri", pair_dim=PAIR_DIM,
        t7=True, t7_disable_pair_update=disable_pair_update,
    ).to(DEVICE)


def load_batch(data_root, task, n_graphs):
    """Load the first `n_graphs` molecules from the V2 'all' split as one batch."""
    ds = load_pyg_inmemory_split(data_root, task, "all")
    loader = DataLoader(ds, batch_size=n_graphs, shuffle=False)
    return next(iter(loader))


# --------------------------------------------------------------------------- #
# VF-equiv
# --------------------------------------------------------------------------- #
def vf_equiv(data_root, task, tol=1e-4):
    """T7(gate->-inf, disable_pair_update) must equal the pure-T5 spectral path.

    Run A: T7 path active, attn_gate forced to -1e9 so sigmoid(gate)=0 kills the
           node-side attn_acc injection; t7_disable_pair_update keeps pair_repr
           static so the per-step Laplacian rebuild reproduces the T5 Laplacian.
    Run B: prop.t7_enabled flipped off -> the prop forward takes the dynamic=False
           pure-T5 branch on the SAME shared parameters.
    A and B must be numerically identical (up to float accumulation noise).
    """
    model = build_t7(disable_pair_update=True)
    model.eval()
    batch = load_batch(data_root, task, n_graphs=6)
    prop = model.encoder.prop1

    with torch.no_grad():
        saved_gate = prop.attn_gate.data.clone()
        prop.attn_gate.data.fill_(-1e9)
        out_a = model(batch, DEVICE)
        prop.attn_gate.data.copy_(saved_gate)

        prop.t7_enabled = False
        out_b = model(batch, DEVICE)
        prop.t7_enabled = True

    names = ["low_x", "high_x", "spec_x", "x_fp"]
    diffs = {n: (a - b).abs().max().item() for n, a, b in zip(names, out_a, out_b)}
    max_diff = max(diffs.values())
    ok = max_diff < tol
    print(f"  VF-equiv : max|A-B| = {max_diff:.2e}  (tol {tol})  per-head={diffs}")
    print(f"             {'PASS' if ok else 'FAIL'} — T7 reduces to T5 safety floor")
    return ok


# --------------------------------------------------------------------------- #
# VF-nan
# --------------------------------------------------------------------------- #
def vf_nan(data_root, task):
    """One fwd+bwd on a real V2 batch must produce no NaN/Inf anywhere."""
    model = build_t7()
    model.train()
    batch = load_batch(data_root, task, n_graphs=8)

    out = model(batch, DEVICE)
    loss = ours_loss(*out, alpha=1)
    model.zero_grad()
    loss.backward()

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
def vf_grad(data_root, task):
    """Gradients must reach every T7-specific parameter group."""
    model = build_t7()  # disable_pair_update=False so delta_proj is on the graph
    model.train()
    batch = load_batch(data_root, task, n_graphs=8)

    out = model(batch, DEVICE)
    loss = ours_loss(*out, alpha=1)
    model.zero_grad()
    loss.backward()

    pa = model.encoder.prop1.pair_attn
    params = {
        "t7.q":          pa.q.weight,
        "t7.k":          pa.k.weight,
        "t7.v":          pa.v.weight,
        "t7.out_proj":   pa.out_proj.weight,
        "t7.bias_proj":  pa.bias_proj.weight,
        "t7.delta_proj": pa.delta_proj.weight,
        "t7.attn_gate":  model.encoder.prop1.attn_gate,
    }
    status = {}
    for name, p in params.items():
        g = p.grad
        status[name] = g is not None and torch.isfinite(g).all().item() \
            and g.abs().sum().item() > 0.0
    ok = all(status.values())
    dead = [k for k, v in status.items() if not v]
    print(f"  VF-grad  : {status}")
    print(f"             {'PASS' if ok else 'FAIL — dead: ' + str(dead)} "
          f"— all T7 params receive gradient")
    return ok


# --------------------------------------------------------------------------- #
# VF-batch
# --------------------------------------------------------------------------- #
def vf_batch(data_root, task, tol=1e-6):
    """Perturbing the last molecule must leave molecule 0's pooled output exact.

    Cross-molecule attention leakage would make molecule 0 depend on molecule
    N-1's node features; this asserts it does not.
    """
    model = build_t7()
    model.eval()
    batch = load_batch(data_root, task, n_graphs=4)

    with torch.no_grad():
        out_base = model(batch, DEVICE)

        batch2 = batch.clone()
        last = int(batch2.batch.max().item())
        mask = batch2.batch == last
        batch2.x[mask] = batch2.x[mask] + torch.randn_like(batch2.x[mask])
        out_pert = model(batch2, DEVICE)

    names = ["low_x", "high_x", "spec_x", "x_fp"]
    # graph 0 is row 0 of each pooled [n_graphs, hid] output
    diffs = {n: (a[0] - b[0]).abs().max().item()
             for n, a, b in zip(names, out_base, out_pert)}
    max_diff = max(diffs.values())
    ok = max_diff < tol
    print(f"  VF-batch : mol-0 max drift after perturbing mol-{last} = "
          f"{max_diff:.2e}  (tol {tol})  {diffs}")
    print(f"             {'PASS' if ok else 'FAIL'} — no cross-molecule leakage")
    return ok


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="T7 code-level verification functions")
    parser.add_argument("--data-root", default="down_task_v2",
                        help="V2-format dataset root (must have processed/<task>_all.pt)")
    parser.add_argument("--task", default="bace")
    args = parser.parse_args()

    print("=" * 68)
    print(f"T7 code VFs — data_root={args.data_root} task={args.task}")
    print("=" * 68)

    results = {}
    for name, fn in [
        ("VF-equiv", vf_equiv),
        ("VF-nan",   vf_nan),
        ("VF-grad",  vf_grad),
        ("VF-batch", vf_batch),
    ]:
        try:
            results[name] = fn(args.data_root, args.task)
        except Exception as exc:  # a crash IS a verification failure
            import traceback
            traceback.print_exc()
            print(f"  {name} : FAIL — raised {type(exc).__name__}: {exc}")
            results[name] = False
        print("-" * 68)

    n_pass = sum(results.values())
    print(f"SUMMARY: {n_pass}/{len(results)} code VFs passed  -> "
          f"{dict((k, 'PASS' if v else 'FAIL') for k, v in results.items())}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
