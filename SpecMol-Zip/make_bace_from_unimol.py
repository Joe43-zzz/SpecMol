import argparse
import os
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

from create_data_DC import smile_to_graph
from utils_fp_downstream import CombinedFingerprintsFeaturizer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "unimol_out" / "splits" / "data_with_folds_scaffold_k10_seed42.csv"
DEFAULT_SDF_DIR = REPO_ROOT / "unimol_out" / "sdf"
DEFAULT_SDF_PATH = DEFAULT_SDF_DIR / "smiles_unimol.sdf"
DEFAULT_PAIR_REP_DIR = REPO_ROOT / "unimol_out" / "pair_rep"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "down_task" / "processed"

FOLD = 0


def env_path(name, default):
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


def resolve_sdf_path(sdf_dir_arg=None, sdf_path_arg=None):
    if sdf_path_arg:
        return Path(sdf_path_arg).expanduser().resolve()
    if os.getenv("SDF_PATH"):
        return Path(os.environ["SDF_PATH"]).expanduser().resolve()
    if sdf_dir_arg:
        sdf_dir = Path(sdf_dir_arg).expanduser().resolve()
    else:
        sdf_dir = env_path("SDF_DIR", DEFAULT_SDF_DIR)
    if sdf_dir.is_file():
        return sdf_dir
    return sdf_dir / "smiles_unimol.sdf"


def resolve_pair_rep_path(pair_rep_dir, sdf_path, mol_index):
    pair_rep_dir = Path(pair_rep_dir)
    sdf_stem = sdf_path.stem
    candidates = [
        pair_rep_dir / sdf_stem / f"{mol_index:06d}" / "pair_rep.pt",
        pair_rep_dir / f"{mol_index:06d}" / "pair_rep.pt",
        pair_rep_dir / sdf_stem / f"{mol_index:06d}.pt",
        pair_rep_dir / f"{mol_index:06d}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"pair_rep.pt not found for molecule index {mol_index}. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def build_all_pairs_edge_index(num_nodes):
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    src = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, dtype=torch.long).repeat(num_nodes)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def validate_all_pairs_edge_index(edge_index, num_nodes):
    expected_edges = num_nodes * (num_nodes - 1)
    if edge_index.shape != (2, expected_edges):
        raise AssertionError(
            f"edge_index shape mismatch: got {tuple(edge_index.shape)}, expected (2, {expected_edges})"
        )

    src = edge_index[0]
    dst = edge_index[1]
    if torch.any(src == dst):
        raise AssertionError("edge_index contains self-loops")

    out_degree = torch.bincount(src, minlength=num_nodes)
    in_degree = torch.bincount(dst, minlength=num_nodes)
    expected_degree = torch.full((num_nodes,), num_nodes - 1, dtype=torch.long)
    if not torch.equal(out_degree, expected_degree):
        raise AssertionError(f"out-degree mismatch: {out_degree.tolist()}")
    if not torch.equal(in_degree, expected_degree):
        raise AssertionError(f"in-degree mismatch: {in_degree.tolist()}")

    edge_pairs = set(map(tuple, edge_index.t().tolist()))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if (i, j) not in edge_pairs:
                raise AssertionError(f"missing directed edge ({i}, {j})")
            if (j, i) not in edge_pairs:
                raise AssertionError(f"missing reverse directed edge ({j}, {i})")


def pair_rep_to_edge_weight(pair_rep, keep_atom_indices, num_nodes):
    if pair_rep.dim() != 3:
        raise ValueError(f"encoder_pair_rep must be 3D, got shape {tuple(pair_rep.shape)}")

    if keep_atom_indices is None:
        keep_atom_indices = torch.arange(pair_rep.size(0), dtype=torch.long)
    else:
        keep_atom_indices = torch.as_tensor(keep_atom_indices, dtype=torch.long)

    pair_rep = pair_rep.index_select(0, keep_atom_indices).index_select(1, keep_atom_indices)
    if pair_rep.size(0) != num_nodes:
        raise ValueError(
            f"Filtered encoder_pair_rep size {pair_rep.size(0)} does not match graph node count {num_nodes}"
        )

    pair_scalar = F.softplus(pair_rep.mean(dim=-1))
    src, dst = build_all_pairs_edge_index(num_nodes)
    edge_weight = pair_scalar[src, dst].to(torch.float32)
    return src, dst, edge_weight


def rebuild_full_edge_attr(original_edge_index, original_edge_attr, full_edge_index):
    # Uni-Mol pair repr
    if original_edge_attr.dim() != 2:
        edge_attr_dim = 0
    else:
        edge_attr_dim = original_edge_attr.size(-1)

    full_edge_attr = torch.zeros((full_edge_index.size(1), edge_attr_dim), dtype=torch.float32)
    if edge_attr_dim == 0 or original_edge_index.numel() == 0:
        return full_edge_attr

    edge_attr_lookup = {}
    for idx in range(original_edge_index.size(1)):
        src = int(original_edge_index[0, idx].item())
        dst = int(original_edge_index[1, idx].item())
        attr = original_edge_attr[idx].to(torch.float32)
        edge_attr_lookup[(src, dst)] = attr
        if (dst, src) not in edge_attr_lookup:
            edge_attr_lookup[(dst, src)] = attr

    for idx in range(full_edge_index.size(1)):
        src = int(full_edge_index[0, idx].item())
        dst = int(full_edge_index[1, idx].item())
        attr = edge_attr_lookup.get((src, dst))
        if attr is not None:
            full_edge_attr[idx] = attr

    return full_edge_attr


def infer_keep_atom_indices(pair_payload, atoms_from_graph):
    if "keep_atom_indices" in pair_payload:
        return torch.as_tensor(pair_payload["keep_atom_indices"], dtype=torch.long)

    atomic_symbols = pair_payload.get("atomic_symbols")
    if atomic_symbols is None:
        if pair_payload["encoder_pair_rep"].size(0) == len(atoms_from_graph):
            return torch.arange(len(atoms_from_graph), dtype=torch.long)
        raise ValueError("pair_rep payload does not contain keep_atom_indices or atomic_symbols")

    heavy_indices = [idx for idx, symbol in enumerate(atomic_symbols) if symbol != "H"]
    if len(heavy_indices) != len(atoms_from_graph):
        raise ValueError(
            "Heavy-atom count from pair_rep does not match graph node count: "
            f"{len(heavy_indices)} vs {len(atoms_from_graph)}"
        )
    return torch.tensor(heavy_indices, dtype=torch.long)


def load_pair_payload(pair_rep_dir, sdf_path, mol_index):
    pair_path = resolve_pair_rep_path(pair_rep_dir, sdf_path, mol_index)
    payload = torch.load(pair_path, map_location="cpu")
    if "encoder_pair_rep" not in payload:
        raise KeyError(f"{pair_path} does not contain encoder_pair_rep")
    return payload


def build_graph_data(smile, label, weight, fingerprint):
    c_size, features, edge_index, edge_features, atoms = smile_to_graph(smile)

    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    elif edge_index.dim() == 2 and edge_index.size(0) != 2:
        edge_index = edge_index.t().contiguous()

    if edge_features is None:
        edge_attr = torch.empty((edge_index.size(1), 0), dtype=torch.float32)
    else:
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(label, dtype=torch.float32),
        edge_attr=edge_attr,
        w=torch.tensor(weight, dtype=torch.float32),
        fps=torch.tensor(fingerprint, dtype=torch.float32),
    )
    return data, atoms


def attach_unimol_pair_to_data(data, atoms, pair_payload):
    num_nodes = data.x.size(0)
    keep_atom_indices = infer_keep_atom_indices(pair_payload, atoms)
    encoder_pair_rep = torch.as_tensor(pair_payload["encoder_pair_rep"], dtype=torch.float32)
    if encoder_pair_rep.dim() == 4 and encoder_pair_rep.size(0) == 1:
        encoder_pair_rep = encoder_pair_rep.squeeze(0)
    if encoder_pair_rep.dim() != 3:
        raise ValueError(f"encoder_pair_rep must be 3D after squeeze, got shape {tuple(encoder_pair_rep.shape)}")

    # Uni-Mol pair repr: [num_nodes, num_nodes, pair_dim]
    pair_repr_full = encoder_pair_rep.index_select(0, keep_atom_indices).index_select(1, keep_atom_indices).contiguous()
    if pair_repr_full.size(0) != num_nodes or pair_repr_full.size(1) != num_nodes:
        raise ValueError(
            f"Filtered pair_repr_full shape {tuple(pair_repr_full.shape)} does not match graph node count {num_nodes}"
        )

    # All-pairs edge index for pair_repr_edge alignment.
    # Stored as data.pair_edge_index (PyG auto-increments fields containing 'index').
    # data.edge_index is kept as the original chem bond graph for ChebNet II.
    full_edge_index = build_all_pairs_edge_index(num_nodes)

    pair_repr_full = pair_repr_full.to(torch.float32)
    data.pair_edge_index = full_edge_index
    data.pair_repr_edge = pair_repr_full[full_edge_index[0], full_edge_index[1]].to(torch.float32)
    data.pair_repr_dim = int(pair_repr_full.size(-1))

    # Validate
    expected_edges = num_nodes * (num_nodes - 1)
    if data.pair_edge_index.size(1) != expected_edges:
        raise AssertionError(
            f"pair_edge_index edge count mismatch: {data.pair_edge_index.size(1)} vs expected {expected_edges}"
        )
    validate_all_pairs_edge_index(data.pair_edge_index, num_nodes)
    if pair_repr_full.shape != (num_nodes, num_nodes, data.pair_repr_dim):
        raise AssertionError(
            f"pair_repr_full shape mismatch: {tuple(pair_repr_full.shape)}"
        )
    if data.pair_repr_edge.shape != (expected_edges, data.pair_repr_dim):
        raise AssertionError(
            f"pair_repr_edge shape mismatch: {tuple(data.pair_repr_edge.shape)}"
        )

    return data


def build_split(
    fold=FOLD,
    csv_path=None,
    sdf_dir=None,
    sdf_path=None,
    pair_rep_dir=None,
    output_dir=None,
):
    csv_path = Path(csv_path).expanduser().resolve() if csv_path else env_path("CSV_PATH", DEFAULT_CSV)
    sdf_path = resolve_sdf_path(sdf_dir_arg=sdf_dir, sdf_path_arg=sdf_path)
    pair_rep_dir = (
        Path(pair_rep_dir).expanduser().resolve()
        if pair_rep_dir
        else env_path("PAIR_REP_DIR", DEFAULT_PAIR_REP_DIR)
    )
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else env_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    smiles_all = df["smiles"].astype(str).tolist()
    labels_all = df["label"].values.astype(np.float32)
    fold_ids = df["fold_id"].values.astype(int)

    fps_all = CombinedFingerprintsFeaturizer().featurize(smiles_all)
    y_all = labels_all.reshape(-1, 1).astype(np.float32)
    w_init = np.ones_like(y_all, dtype=np.float32)
    ds_all = dc.data.NumpyDataset(X=fps_all, y=y_all, w=w_init, ids=np.array(smiles_all))

    kfold = fold_ids.max() + 1
    test_fold = fold
    val_fold = (fold + 1) % kfold

    test_idx = np.where(fold_ids == test_fold)[0]
    val_idx = np.where(fold_ids == val_fold)[0]
    train_idx = np.where(~((fold_ids == test_fold) | (fold_ids == val_fold)))[0]

    train_ds = ds_all.select(train_idx)
    valid_ds = ds_all.select(val_idx)
    test_ds = ds_all.select(test_idx)

    balancer = dc.trans.BalancingTransformer(dataset=train_ds)
    train_ds = balancer.transform(train_ds)
    valid_ds = balancer.transform(valid_ds)
    test_ds = balancer.transform(test_ds)

    w_all = np.zeros_like(y_all, dtype=np.float32)
    w_all[train_idx] = train_ds.w
    w_all[val_idx] = valid_ds.w
    w_all[test_idx] = test_ds.w

    print(f"csv_path={csv_path}")
    print(f"sdf_path={sdf_path}")
    print(f"pair_rep_dir={pair_rep_dir}")
    print(f"output_dir={output_dir}")
    print(f"kfold={kfold}, train={len(train_idx)}, valid={len(val_idx)}, test={len(test_idx)}")

    def build_data_list(indices):
        data_list = []
        for mol_index in indices:
            smile = smiles_all[mol_index]
            label = np.array([labels_all[mol_index]], dtype=np.float32)
            weight = w_all[mol_index].astype(np.float32)
            try:
                data, atoms = build_graph_data(
                    smile=smile,
                    label=label,
                    weight=weight,
                    fingerprint=fps_all[mol_index],
                )
                pair_payload = load_pair_payload(pair_rep_dir, sdf_path, mol_index)
                data = attach_unimol_pair_to_data(data, atoms, pair_payload)
                data.mol_id = int(mol_index)
                data_list.append(data)
            except Exception as exc:
                print(f"skip sample {mol_index}: {smile} -> {exc}")
        return data_list

    train_list = build_data_list(train_idx)
    valid_list = build_data_list(val_idx)
    test_list = build_data_list(test_idx)

    data_train, slices_train = InMemoryDataset.collate(train_list)
    data_valid, slices_valid = InMemoryDataset.collate(valid_list)
    data_test, slices_test = InMemoryDataset.collate(test_list)

    torch.save((data_train, slices_train), output_dir / "bace_train.pt")
    torch.save((data_valid, slices_valid), output_dir / "bace_valid.pt")
    torch.save((data_test, slices_test), output_dir / "bace_test.pt")

    all_list = train_list + valid_list + test_list
    data_all, slices_all = InMemoryDataset.collate(all_list)
    torch.save((data_all, slices_all), output_dir / "bace_all.pt")

    print("saved processed datasets with Uni-Mol pair_repr_edge (chem bond edge_index preserved)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--sdf-dir", type=str, default=None)
    parser.add_argument("--sdf-path", type=str, default=None)
    parser.add_argument("--pair-rep-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_split(
        fold=args.fold,
        csv_path=args.csv_path,
        sdf_dir=args.sdf_dir,
        sdf_path=args.sdf_path,
        pair_rep_dir=args.pair_rep_dir,
        output_dir=args.output_dir,
    )
