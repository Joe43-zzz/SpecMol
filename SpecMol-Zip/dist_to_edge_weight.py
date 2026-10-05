"""T8: 3D distance-weighted Laplacian via through-space contact edges.

The ChebNetII spectral encoder computes Chebyshev polynomials of the symmetric
normalized Laplacian of the *chemical-bond* graph. That operator is, by
construction, blind to through-space proximity: two atoms that are far along the
bond path but close in 3D (e.g. across a ring, a folded chain, an H-bond) are
spectrally unrelated. T8 augments the bond graph with **contact edges** between
atom pairs whose interatomic distance d_ij < r_cut, weighted by a learnable
radial-basis function w(d). Building the Laplacian on this augmented graph
changes its eigenbasis (Fiedler vector, low/high-pass cuts) in a way the
bond-graph spectrum *cannot represent* — the only proposal that alters the
operator the spectrum is computed on, rather than adding a parallel branch.

Two properties make this a clean, defensible contribution:
  - It passes the random-pair control by construction: real distances vs a
    shuffled/random tensor give different operators; capacity is identical.
  - It is principled for a spectral GNN (adaptive spectral filtering on a
    geometry-augmented molecular Laplacian; cf. SPECTRA 2025).

Inputs are the per-edge interatomic distances `pair_dist_edge` aligned to the
all-pairs `pair_edge_index` already stored in the .pt by
make_bace_from_unimol.attach_unimol_pair_to_data (T8 block).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceToEdgeWeight(nn.Module):
    """Map interatomic distances to a learnable, positive contact-edge weight.

    w(d) = softplus( sum_k a_k * exp(-(d - mu_k)^2 / (2 sigma^2)) ), a_k learnable.

    Args:
        num_rbf: number of Gaussian basis centers spread over [0, r_cut].
        r_cut: contact cutoff in Angstrom; pairs with d >= r_cut are dropped.
        exclude_bonded: if a bond edge_index is provided to forward(), skip
            contacts that are already chemical bonds (avoid double-counting;
            the bond edges are added separately with their own weights).
    """

    def __init__(self, num_rbf=16, r_cut=5.0, init_scale=0.1):
        super().__init__()
        self.r_cut = float(r_cut)
        centers = torch.linspace(0.0, self.r_cut, num_rbf)
        self.register_buffer("mu", centers)
        # sigma = spacing between centers (standard Gaussian-basis choice).
        self.sigma = float(self.r_cut / max(num_rbf - 1, 1))
        # Learnable RBF coefficients; small positive init so contacts start as a
        # gentle perturbation of the bond Laplacian (stability inside the K-loop).
        self.coeff = nn.Parameter(torch.full((num_rbf,), float(init_scale)))

    def rbf_weight(self, dist):
        # dist: [E] -> w: [E] positive
        basis = torch.exp(-((dist.unsqueeze(-1) - self.mu) ** 2) / (2.0 * self.sigma ** 2))
        return F.softplus((basis * self.coeff).sum(-1))

    def forward(self, pair_dist_edge, pair_edge_index, bond_edge_index=None):
        """Return (contact_edge_index [2,Ec], contact_weight [Ec]) for d < r_cut.

        pair_edge_index holds ordered pairs i!=j (both directions), so the
        returned contact graph is symmetric -- required by get_laplacian('sym').
        """
        device = pair_dist_edge.device
        keep = pair_dist_edge < self.r_cut
        if bond_edge_index is not None and bond_edge_index.size(1) > 0:
            # Drop contacts that are already chemical bonds (compare via a hashed
            # node-pair key; num_nodes-agnostic, just needs a large multiplier).
            mult = int(pair_edge_index.max().item()) + 1 if pair_edge_index.numel() else 1
            bond_key = bond_edge_index[0] * mult + bond_edge_index[1]
            pair_key = pair_edge_index[0] * mult + pair_edge_index[1]
            is_bond = torch.isin(pair_key, bond_key)
            keep = keep & (~is_bond)
        ei = pair_edge_index[:, keep]
        w = self.rbf_weight(pair_dist_edge[keep])
        return ei, w


class RBFPairEmbedding(nn.Module):
    """Option A2: embed raw interatomic distance d_ij -> a [E, pair_dim] pair
    representation -- a RAW, end-to-end-TRAINABLE geometry pair that REPLACES the
    frozen single-conformer Uni-Mol pair_repr as the input to the T9 co-update.

    Dismantles all three limitations of the frozen Uni-Mol injection:
      - raw (not Uni-Mol-mediated): reads d_ij directly,
      - trainable/refineable: gradient flows to the projection (the model learns
        which distance bands matter per pair channel),
      - H-includable: if pair_dist_edge keeps hydrogens, so does this.

    Unlike DistanceToEdgeWeight (scalar contact-edge weight, prunes d>=r_cut),
    this keeps the FULL all-pairs set and returns a pair_dim vector per edge:
      basis_k(d) = exp(-(d-mu_k)^2 / (2 sigma^2))   (num_rbf fixed centers)
      pair       = Linear(num_rbf -> pair_dim)(basis)   (the trainable part)
    proj init small so it starts as a gentle, near-flat pair (and combined with
    T9's ReZero gates at 0, epoch-0 stays == V2-T5 regardless of this output).
    """

    def __init__(self, num_rbf=32, r_cut=8.0, pair_dim=64, init_std=0.02):
        super().__init__()
        self.r_cut = float(r_cut)
        self.register_buffer("mu", torch.linspace(0.0, self.r_cut, num_rbf))
        self.sigma = float(self.r_cut / max(num_rbf - 1, 1))
        self.proj = nn.Linear(num_rbf, pair_dim)
        nn.init.normal_(self.proj.weight, std=init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, pair_dist_edge):
        # pair_dist_edge: [E] -> [E, pair_dim]
        basis = torch.exp(-((pair_dist_edge.unsqueeze(-1) - self.mu) ** 2)
                          / (2.0 * self.sigma ** 2))
        return self.proj(basis)


def augment_laplacian_edges(bond_edge_index, bond_edge_weight, contact_edge_index,
                            contact_weight, num_nodes):
    """Concatenate chemical-bond edges (weight 1.0 or the V2 gate weight) with the
    T8 contact edges into one weighted edge set for get_laplacian. Bonds keep
    their existing weights; contacts use the learnable RBF weights.
    """
    if bond_edge_weight is None:
        bond_edge_weight = bond_edge_index.new_ones(bond_edge_index.size(1), dtype=contact_weight.dtype)
    ei = torch.cat([bond_edge_index, contact_edge_index], dim=1)
    ew = torch.cat([bond_edge_weight.to(contact_weight.dtype), contact_weight], dim=0)
    return ei, ew
