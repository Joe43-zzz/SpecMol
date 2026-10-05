"""Verify atom counts changed across mols."""
import torch, os
PRE = os.path.expanduser('~/zhoutianyang/SpecMol/SpecMol-Zip/down_task_v2/processed/bace_all.pt')
POST = os.path.expanduser('~/zhoutianyang/SpecMol/SpecMol-Zip/down_task_bace_v2/processed/bace_all.pt')
def load(p):
    obj = torch.load(p, map_location='cpu')
    if isinstance(obj, tuple): return obj[0], obj[1]
    return obj['data'], obj['slices']
pre_d, pre_s = load(PRE); post_d, post_s = load(POST)
n = len(pre_s['x'])-1
n_diff_atoms = 0
n_diff_bonds = 0
n_pre_atoms = 0; n_post_atoms = 0
for i in range(n):
    pa = int(pre_s['x'][i+1]) - int(pre_s['x'][i])
    poa = int(post_s['x'][i+1]) - int(post_s['x'][i])
    pb = int(pre_s['edge_index'][i+1]) - int(pre_s['edge_index'][i])
    pob = int(post_s['edge_index'][i+1]) - int(post_s['edge_index'][i])
    n_pre_atoms += pa; n_post_atoms += poa
    if pa != poa: n_diff_atoms += 1
    if pob != 2*pb: n_diff_bonds += 1
print(f'Total mols: {n}')
print(f'Mols with different atom count: {n_diff_atoms} ({100*n_diff_atoms/n:.1f}%)')
print(f'Mols where post bonds != 2*pre bonds: {n_diff_bonds} ({100*n_diff_bonds/n:.1f}%)')
print(f'Total atoms PRE={n_pre_atoms} POST={n_post_atoms} (diff={n_post_atoms-n_pre_atoms})')
print(f'Total bonds PRE={pre_d.edge_index.size(1)} POST={post_d.edge_index.size(1)}')

# check mol_id alignment
if hasattr(pre_d, 'mol_id'):
    print(f'mol_id PRE[:10]={pre_d.mol_id[:10].tolist() if hasattr(pre_d.mol_id, "tolist") else pre_d.mol_id[:10]}')
    print(f'mol_id POST[:10]={post_d.mol_id[:10].tolist() if hasattr(post_d.mol_id, "tolist") else post_d.mol_id[:10]}')
    # are they same set?
    pre_ids = set(pre_d.mol_id.tolist() if hasattr(pre_d.mol_id, "tolist") else pre_d.mol_id)
    post_ids = set(post_d.mol_id.tolist() if hasattr(post_d.mol_id, "tolist") else post_d.mol_id)
    print(f'Same mol_id set? {pre_ids == post_ids}')
    print(f'PRE-only ids: {len(pre_ids - post_ids)}, POST-only: {len(post_ids - pre_ids)}')

# y/labels alignment
print(f'y PRE shape={pre_d.y.shape}, POST shape={post_d.y.shape}')
print(f'y PRE sum={pre_d.y.sum().item()}, POST sum={post_d.y.sum().item()}')
print(f'y identical? {torch.equal(pre_d.y, post_d.y)}')

# fps identical?
if hasattr(pre_d, 'fps'):
    print(f'fps PRE shape={pre_d.fps.shape}, POST shape={post_d.fps.shape}')
    print(f'fps identical? {torch.equal(pre_d.fps, post_d.fps)}')

# pair_edge_index - count per mol
print('\n--- pair_edge_index per mol[0..4] ---')
for i in range(5):
    pre_pe = int(pre_s['pair_edge_index'][i+1]) - int(pre_s['pair_edge_index'][i])
    post_pe = int(post_s['pair_edge_index'][i+1]) - int(post_s['pair_edge_index'][i])
    pre_n = int(pre_s['x'][i+1]) - int(pre_s['x'][i])
    post_n = int(post_s['x'][i+1]) - int(post_s['x'][i])
    print(f'  mol[{i}]: PRE n={pre_n} pair_ei={pre_pe} (expected {pre_n*(pre_n-1)}={pre_n*(pre_n-1)}), POST n={post_n} pair_ei={post_pe} (expected {post_n*(post_n-1)}={post_n*(post_n-1)})')
