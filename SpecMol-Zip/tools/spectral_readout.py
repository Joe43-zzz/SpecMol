"""tools/spectral_readout.py -- free-CPU spectral readout for ChebInject.

Two analyses that back the paper's mechanism claims without any new training:

  Part A (EXACT, checkpoint-free).  Under symmetric Laplacian normalization the
    *mean* of the V2-T5 edge gate is spectrally invisible: L_sym is identical for
    every uniform edge weight c>0 (because the c cancels in A_ij / sqrt(d_i d_j)),
    while a HETEROGENEOUS weight vector does move the spectrum.  So the scalar
    gate can only act through per-edge heterogeneity -- never through its average
    strength.  We verify this numerically on a drug-like skeleton.

  Part B (checkpoint-light).  Reconstruct the LEARNED ChebNetII low/high-pass
    filter responses h(lambda), lambda in [0,2], from a trained checkpoint's
    {slides_*, beta_*} parameters, and overlay a real-pair (V2-T5) vs random-pair
    (V2-T5-null) model.  If the two learned filters coincide, the frozen pair
    injection did not reshape the spectral filter -- consistent with the
    real-pair ~= random-pair finding.

Faithfulness: the reparam (gamma_l/gamma_h), the node->coefficient interpolation,
and the response convention out = coe[0]/2 * T0 + sum_{i>=1} coe[i] * T_i are
copied verbatim from LH_Direct_ChebnetII_prop_v2.py.  The scaled spectral
variable is lambda~ = lambda_sym - 1 (the model adds self-loops with
fill_value=-1.0, i.e. propagates on L_sym - I, whose spectrum is [-1, 1]).

Usage:
  python tools/spectral_readout.py \
      --real   pkl/best_spec_model_bace_v2_19_1777393627.pkl \
      --random pkl/best_spec_model_bace_v2null_19_1778075283.pkl \
      --out    paper/audits/spectral_readout

Part A always runs.  Part B runs only if --real (and optionally --random) load.
Writes <out>.json always; <out>.png too if matplotlib is importable.
K is inferred from the checkpoint (len(slides_l) - 1); no --K needed.
"""
import argparse
import json
import os

import numpy as np

# --------------------------------------------------------------------------- #
# ChebNetII reparam, copied faithfully from LH_Direct_ChebnetII_prop_v2.py
# --------------------------------------------------------------------------- #

def _relu(x):
    return np.maximum(x, 0.0)


def gamma_lh(slides, beta_a, beta_b, highpass):
    """Filter VALUES at the K+1 Chebyshev nodes (ChebNetII property).

    Mirrors gamma_l (highpass=False) / gamma_h (highpass=True):
      term1 = 0.5 * relu(beta_b) * (1 + cos(((1+j+slides[j]) / L) * pi))
      gamma[j] = relu(beta_a) -/+ term1            (low: minus, high: plus)
    where L = len(slides) = K+1.
    """
    L = len(slides)
    j = np.arange(L)
    term1 = 0.5 * _relu(beta_b) * (1.0 + np.cos(((1 + j + slides) / L) * np.pi))
    return _relu(beta_a) + (term1 if highpass else -term1)


def cheby_vec(i, x):
    """T_i(x) evaluated on a vector x (numpy), matching cheby() recurrence."""
    if i == 0:
        return np.ones_like(x)
    if i == 1:
        return x
    T0 = np.ones_like(x)
    T1 = x
    for _ in range(2, i + 1):
        T2 = 2 * x * T1 - T0
        T0, T1 = T1, T2
    return T1


def node_values_to_coeffs(coe_tmp, K):
    """Convert filter values at Chebyshev nodes -> Chebyshev expansion coeffs.

    Replicates the loop in ChebnetII_prop_V2.forward:
      x_0 = cos((K+0.5) pi / (K+1)),  x_j = cos((K-j+0.5) pi / (K+1)) for j>=1
      coe[i] = (2/(K+1)) * sum_j coe_tmp[j] * T_i(x_j)
    """
    coe = np.zeros(K + 1)
    x0 = np.cos((K + 0.5) * np.pi / (K + 1))
    for i in range(K + 1):
        acc = coe_tmp[0] * _cheby_scalar(i, x0)
        for jj in range(1, K + 1):
            xj = np.cos((K - jj + 0.5) * np.pi / (K + 1))
            acc += coe_tmp[jj] * _cheby_scalar(i, xj)
        coe[i] = 2.0 * acc / (K + 1)
    return coe


def _cheby_scalar(i, x):
    if i == 0:
        return 1.0
    if i == 1:
        return x
    T0, T1 = 1.0, x
    for _ in range(2, i + 1):
        T0, T1 = T1, 2 * x * T1 - T0
    return T1


def filter_response(slides_param, beta_a, beta_b, K, highpass, grid_xtilde):
    """h(lambda~) on grid_xtilde in [-1,1].  lambda_sym = lambda~ + 1 in [0,2]."""
    slides = 0.5 * np.tanh(slides_param)              # == prop forward
    coe_tmp = gamma_lh(slides, beta_a, beta_b, highpass)
    coe = node_values_to_coeffs(coe_tmp, K)
    # out = coe[0]/2 * T0 + sum_{i>=1} coe[i] * T_i
    h = 0.5 * coe[0] * cheby_vec(0, grid_xtilde)
    for i in range(1, K + 1):
        h = h + coe[i] * cheby_vec(i, grid_xtilde)
    return h, coe_tmp, coe


# --------------------------------------------------------------------------- #
# Part A: sym-Laplacian spectrum -- mean invisible, heterogeneity matters
# --------------------------------------------------------------------------- #

def _drug_like_edges():
    """A small fused-bicyclic skeleton (11 atoms, 12 bonds), connected."""
    return [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),   # 6-ring
        (4, 6), (6, 7), (7, 8), (8, 5),                    # fused 5-ring on 4-5
        (2, 9), (0, 10),                                   # two substituents
    ]


def sym_laplacian_eigs(n, edges, weights):
    """Eigenvalues of L_sym = I - D^{-1/2} A D^{-1/2} for a weighted graph."""
    A = np.zeros((n, n))
    for (u, v), w in zip(edges, weights):
        A[u, v] += w
        A[v, u] += w
    d = A.sum(1)
    dinv = np.zeros_like(d)
    nz = d > 0
    dinv[nz] = 1.0 / np.sqrt(d[nz])
    L = np.eye(n) - (dinv[:, None] * A * dinv[None, :])
    L = 0.5 * (L + L.T)                                   # symmetrize numerically
    return np.sort(np.linalg.eigvalsh(L))


def part_a():
    edges = _drug_like_edges()
    n = 1 + max(max(u, v) for u, v in edges)
    m = len(edges)
    rng = np.random.default_rng(0)
    hetero = rng.uniform(0.2, 1.0, m)

    eig_c1 = sym_laplacian_eigs(n, edges, np.ones(m) * 1.0)
    eig_c7 = sym_laplacian_eigs(n, edges, np.ones(m) * 7.0)   # different MEAN
    eig_het = sym_laplacian_eigs(n, edges, hetero)            # heterogeneous

    mean_invisible_maxdiff = float(np.max(np.abs(eig_c1 - eig_c7)))
    hetero_l2 = float(np.linalg.norm(eig_c1 - eig_het))

    return {
        "graph": {"n_atoms": n, "n_bonds": m, "edges": edges},
        "eig_uniform_c1": eig_c1.tolist(),
        "eig_uniform_c7": eig_c7.tolist(),
        "eig_heterogeneous": eig_het.tolist(),
        "mean_invisible_maxabsdiff_c1_vs_c7": mean_invisible_maxdiff,
        "heterogeneity_l2_shift_c1_vs_hetero": hetero_l2,
        "verdict": (
            "uniform gate mean is spectrally invisible "
            f"(max|d eig|={mean_invisible_maxdiff:.2e} between c=1 and c=7); "
            f"heterogeneity moves the spectrum (L2 shift={hetero_l2:.3f})."
        ),
    }


# --------------------------------------------------------------------------- #
# Part B: learned filter response from checkpoint(s)
# --------------------------------------------------------------------------- #

def _find_spectral_params(state):
    """Locate {slides_l, slides_h, beta_a_l, beta_a_h, beta_b_l, beta_b_h}.

    Scans the state_dict keys for the param suffixes (handles any module prefix
    and, if several prop instances exist, takes the first consistent group).
    """
    suffixes = ["slides_l", "slides_h",
                "beta_a_l", "beta_a_h", "beta_b_l", "beta_b_h"]
    found = {}
    for s in suffixes:
        cand = [k for k in state if k.split(".")[-1] == s]
        if not cand:
            raise KeyError(f"param '{s}' not found in checkpoint")
        found[s] = np.asarray(state[cand[0]].detach().cpu().numpy()
                              if hasattr(state[cand[0]], "detach")
                              else state[cand[0]], dtype=np.float64)
    return found


def _load_state(path):
    import torch  # local import: only Part B needs torch
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # could be a raw state_dict already, or a wrapper holding one
        if all(hasattr(v, "shape") for v in obj.values()):
            return obj
        for key in ("state_dict", "model", "spec_model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise TypeError(f"cannot extract a state_dict from {path}")


def curves_for_checkpoint(path, grid_xtilde):
    p = _find_spectral_params(_load_state(path))
    K = len(p["slides_l"]) - 1
    out = {"path": os.path.basename(path), "K": K}
    for tag, hp in [("low", False), ("high", True)]:
        sl = p["slides_l"] if not hp else p["slides_h"]
        ba = p["beta_a_l"] if not hp else p["beta_a_h"]
        bb = p["beta_b_l"] if not hp else p["beta_b_h"]
        h, coe_tmp, coe = filter_response(sl, float(ba), float(bb), K, hp,
                                          grid_xtilde)
        out[tag] = {
            "response": h.tolist(),
            "node_values": coe_tmp.tolist(),   # h at the K+1 Chebyshev nodes
            "cheb_coeffs": coe.tolist(),
        }
    return out


# --------------------------------------------------------------------------- #

def maybe_plot(result, grid_lambda, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                # pragma: no cover
        print(f"[plot] matplotlib unavailable ({e}); JSON written, skipping PNG")
        return False

    b = result.get("part_b")
    if not b or "real" not in b:
        print("[plot] no Part-B curves; skipping PNG")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, band in zip(axes, ["low", "high"]):
        ax.plot(grid_lambda, b["real"][band]["response"], label="real pair (V2-T5)",
                lw=2.2)
        if "random" in b:
            ax.plot(grid_lambda, b["random"][band]["response"],
                    label="random pair (null)", lw=2.0, ls="--")
        ax.set_title(f"{band}-pass learned filter")
        ax.set_xlabel(r"$\lambda$ (sym-normalized Laplacian)")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"filter response $h(\lambda)$")
    fig.suptitle("ChebInject learned spectral filter: real vs random pair "
                 f"(K={b['real']['K']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default=None, help="real-pair V2-T5 checkpoint .pkl")
    ap.add_argument("--random", default=None, help="random-pair (null) checkpoint")
    ap.add_argument("--grid", type=int, default=201, help="lambda samples in [0,2]")
    ap.add_argument("--out", default="paper/audits/spectral_readout",
                    help="output path stem (writes .json and, if possible, .png)")
    args = ap.parse_args()

    grid_xtilde = np.linspace(-1.0, 1.0, args.grid)      # lambda~ in [-1,1]
    grid_lambda = grid_xtilde + 1.0                      # lambda_sym in [0,2]

    result = {"lambda_grid": grid_lambda.tolist()}

    # Part A -- always
    result["part_a"] = part_a()
    print("[part A]", result["part_a"]["verdict"])

    # Part B -- if checkpoints given/loadable
    pb = {}
    if args.real:
        try:
            pb["real"] = curves_for_checkpoint(args.real, grid_xtilde)
            print(f"[part B] real:   {pb['real']['path']} (K={pb['real']['K']})")
        except Exception as e:
            print(f"[part B] real load failed: {e}")
    if args.random:
        try:
            pb["random"] = curves_for_checkpoint(args.random, grid_xtilde)
            print(f"[part B] random: {pb['random']['path']} (K={pb['random']['K']})")
        except Exception as e:
            print(f"[part B] random load failed: {e}")
    if "real" in pb and "random" in pb:
        for band in ("low", "high"):
            r = np.asarray(pb["real"][band]["response"])
            n = np.asarray(pb["random"][band]["response"])
            pb.setdefault("real_vs_random_l2", {})[band] = float(np.linalg.norm(r - n))
        print("[part B] real-vs-random filter L2:", pb["real_vs_random_l2"])
    if pb:
        result["part_b"] = pb

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[out] wrote {args.out}.json")

    maybe_plot(result, grid_lambda, args.out + ".png")


if __name__ == "__main__":
    main()
