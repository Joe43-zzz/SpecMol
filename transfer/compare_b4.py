"""Compare pre-B4 vs post-B4 BACE data files. Run on HPC compute node."""
import torch
import os, sys
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse.linalg as sla
import numpy as np

PRE = os.path.expanduser('~/zhoutianyang/SpecMol/SpecMol-Zip/down_task_v2/processed/bace_all.pt')
POST = os.path.expanduser('~/zhoutianyang/SpecMol/SpecMol-Zip/down_task_bace_v2/processed/bace_all.pt')

def load(p):
    obj = torch.load(p, map_location='cpu')
    # InMemoryDataset .pt is typically (data, slices) tuple or (data, slices, ...) — handle both
    if isinstance(obj, tuple):
        data, slices = obj[0], obj[1]
    else:
        data, slices = obj['data'], obj['slices']
    return data, slices

def get_mol(data, slices, idx):
    """Reconstruct molecule idx from collated InMemoryDataset."""
    d = {}
    for k in slices:
        s = slices[k]
        a, b = int(s[idx]), int(s[idx+1])
        v = getattr(data, k)
        if v is None: continue
        # determine cat dim
        if 'index' in k:
            d[k] = v[:, a:b]
        elif v.dim() == 0:
            d[k] = v
        elif v.size(0) == int(s[-1]):
            d[k] = v[a:b]
        else:
            # try last dim
            d[k] = v[..., a:b]
    return d

print('='*70)
print('Loading pre-B4...')
pre_data, pre_slices = load(PRE)
print('Loading post-B4...')
post_data, post_slices = load(POST)

n_mols_pre = len(pre_slices['x']) - 1
n_mols_post = len(post_slices['x']) - 1
print(f'#mols pre={n_mols_pre} post={n_mols_post}')
print(f'Keys pre: {list(pre_slices.keys())}')
print(f'Keys post: {list(post_slices.keys())}')

# ============ Investigation 1: edge_index shape ============
print('\n=== Investigation 1: edge_index shape ===')
print(f'PRE  total edge_index: {pre_data.edge_index.shape}')
print(f'POST total edge_index: {post_data.edge_index.shape}')
print(f'Ratio post/pre = {post_data.edge_index.size(1)/pre_data.edge_index.size(1):.3f}')

m0_pre = get_mol(pre_data, pre_slices, 0)
m0_post = get_mol(post_data, post_slices, 0)
print(f'Mol[0] PRE  edge_index: {m0_pre["edge_index"].shape}')
print(f'Mol[0] POST edge_index: {m0_post["edge_index"].shape}')

# unique pairs
def unique_pairs(ei):
    s = set()
    for i in range(ei.size(1)):
        s.add((int(ei[0,i]), int(ei[1,i])))
    return len(s)
print(f'Mol[0] PRE  unique pairs: {unique_pairs(m0_pre["edge_index"])} / total {m0_pre["edge_index"].size(1)}')
print(f'Mol[0] POST unique pairs: {unique_pairs(m0_post["edge_index"])} / total {m0_post["edge_index"].size(1)}')

mean_pre = pre_data.edge_index.size(1) / n_mols_pre
mean_post = post_data.edge_index.size(1) / n_mols_post
print(f'Per-mol mean edges: PRE={mean_pre:.2f} POST={mean_post:.2f}')

# ============ Investigation 2: Bidirectional check ============
print('\n=== Investigation 2: Bidirectional check (first 5 mols) ===')
for idx in range(5):
    for tag, data, slices in [('PRE', pre_data, pre_slices), ('POST', post_data, post_slices)]:
        m = get_mol(data, slices, idx)
        ei = m['edge_index']
        s = set((int(ei[0,k]), int(ei[1,k])) for k in range(ei.size(1)))
        rev = sum(1 for (a,b) in s if (b,a) in s and a!=b)
        non_self = sum(1 for (a,b) in s if a!=b)
        pct = 100*rev/non_self if non_self else 0
        print(f'  mol[{idx}] {tag}: {ei.size(1)} edges, {len(s)} unique, {rev}/{non_self} have reverse ({pct:.1f}%)')

# ============ Investigation 3: edge_attr shape ============
print('\n=== Investigation 3: edge_attr shape ===')
print(f'PRE  total edge_attr: {pre_data.edge_attr.shape if pre_data.edge_attr is not None else None}')
print(f'POST total edge_attr: {post_data.edge_attr.shape if post_data.edge_attr is not None else None}')
m0_pre_ea = m0_pre.get('edge_attr')
m0_post_ea = m0_post.get('edge_attr')
print(f'Mol[0] PRE  edge_attr: {m0_pre_ea.shape if m0_pre_ea is not None else None}')
print(f'Mol[0] POST edge_attr: {m0_post_ea.shape if m0_post_ea is not None else None}')
# check if reverse edge_attr is duplicate of forward
if m0_post_ea is not None and m0_post['edge_index'].size(1) == 2 * m0_pre['edge_index'].size(1):
    n_fwd = m0_pre['edge_index'].size(1)
    fwd_attr = m0_post_ea[:n_fwd]
    rev_attr = m0_post_ea[n_fwd:]
    print(f'  POST mol[0] first half == second half? {torch.equal(fwd_attr, rev_attr)}')
    # try interleaved pattern
    if m0_post_ea.size(0) % 2 == 0:
        even = m0_post_ea[0::2]; odd = m0_post_ea[1::2]
        print(f'  POST mol[0] even == odd (interleaved)? {torch.equal(even, odd)}')

# ============ Investigation 4: pair_edge_index unchanged? ============
print('\n=== Investigation 4: pair_edge_index ===')
if hasattr(pre_data, 'pair_edge_index'):
    print(f'PRE  total pair_edge_index: {pre_data.pair_edge_index.shape}')
    print(f'POST total pair_edge_index: {post_data.pair_edge_index.shape}')
    print(f'Mol[0] PRE  pair_edge_index: {m0_pre.get("pair_edge_index", "MISSING").shape if "pair_edge_index" in m0_pre else "MISSING"}')
    print(f'Mol[0] POST pair_edge_index: {m0_post.get("pair_edge_index", "MISSING").shape if "pair_edge_index" in m0_post else "MISSING"}')
    same = torch.equal(pre_data.pair_edge_index, post_data.pair_edge_index)
    print(f'PRE pair_edge_index == POST? {same}')

# ============ Investigation 5: pair_repr_edge values ============
print('\n=== Investigation 5: pair_repr_edge values ===')
if hasattr(pre_data, 'pair_repr_edge'):
    print(f'PRE  pair_repr_edge: {pre_data.pair_repr_edge.shape}')
    print(f'POST pair_repr_edge: {post_data.pair_repr_edge.shape}')
    if pre_data.pair_repr_edge.shape == post_data.pair_repr_edge.shape:
        diff = (pre_data.pair_repr_edge - post_data.pair_repr_edge).abs()
        print(f'  max abs diff = {diff.max().item():.6e}')
        print(f'  mean abs diff = {diff.mean().item():.6e}')
        print(f'  identical? {torch.equal(pre_data.pair_repr_edge, post_data.pair_repr_edge)}')

# ============ Investigation 6: Self-loops ============
print('\n=== Investigation 6: Self-loops ===')
for tag, data, slices in [('PRE', pre_data, pre_slices), ('POST', post_data, post_slices)]:
    n = len(slices['x']) - 1
    cnt = 0
    total_self = 0
    for idx in range(n):
        m = get_mol(data, slices, idx)
        ei = m['edge_index']
        sl = (ei[0] == ei[1]).sum().item()
        if sl > 0:
            cnt += 1
            total_self += sl
    print(f'  {tag}: {cnt}/{n} mols have self-loops, total self-loop edges = {total_self}')

# ============ Investigation 7: Spectral norm ============
print('\n=== Investigation 7: Spectral norm of normalized Laplacian (mol[0] and mol[100]) ===')
for idx in [0, 100]:
    print(f' --- mol[{idx}] ---')
    for tag, data, slices in [('PRE', pre_data, pre_slices), ('POST', post_data, post_slices)]:
        m = get_mol(data, slices, idx)
        ei = m['edge_index']
        n = m['x'].size(0)
        try:
            edge_index_l, edge_weight_l = get_laplacian(ei, num_nodes=n, normalization='sym')
            L = to_scipy_sparse_matrix(edge_index_l, edge_weight_l, num_nodes=n).toarray()
            ev = np.linalg.eigvalsh(L)
            print(f'  {tag} N={n} edges={ei.size(1)} eigval range=[{ev.min():.4f}, {ev.max():.4f}] (#>2.001: {(ev>2.001).sum()})')
        except Exception as e:
            print(f'  {tag} ERROR: {e}')

# ============ Investigation 8: Deep look at mol[100] ============
print('\n=== Investigation 8: Deep comparison of mol[100] ===')
for tag, data, slices in [('PRE', pre_data, pre_slices), ('POST', post_data, post_slices)]:
    m = get_mol(data, slices, 100)
    print(f' --- {tag} ---')
    print(f'  x.shape: {m["x"].shape}')
    print(f'  edge_index.shape: {m["edge_index"].shape}')
    print(f'  edge_index[:, :10]:\n{m["edge_index"][:, :10]}')
    if m.get('edge_attr') is not None:
        print(f'  edge_attr.shape: {m["edge_attr"].shape}')
        print(f'  edge_attr[:5]:\n{m["edge_attr"][:5]}')
    if 'pair_edge_index' in m:
        print(f'  pair_edge_index.shape: {m["pair_edge_index"].shape}')
    if 'pair_repr_edge' in m:
        print(f'  pair_repr_edge.shape: {m["pair_repr_edge"].shape}')
        print(f'  pair_repr_edge[:3] (first 5 dims): {m["pair_repr_edge"][:3, :5] if m["pair_repr_edge"].dim()>1 else m["pair_repr_edge"][:3]}')

print('\nDONE')
