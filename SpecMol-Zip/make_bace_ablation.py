import argparse
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from rdkit import Chem

from make_bace_from_unimol import (
    DEFAULT_PAIR_REP_DIR,
    attach_unimol_pair_to_data,
    build_all_pairs_edge_index,
    build_graph_data,
    env_path,
    load_pair_payload,
    resolve_sdf_path,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCH_ROOT = REPO_ROOT / "molecular_benchmarks"
MODE_TO_SUBDIR = {
    "2d": "down_task_2d",
    "ctrl": "down_task_ctrl",
    "bondpair": "down_task_bondpair",
    "unimol": "down_task_unimol",
    "unimolupd": "down_task_unimolupd",
}
TASK_TO_SPLIT_CSV = {
    "bace": REPO_ROOT / "unimol_out" / "splits" / "data_with_folds_scaffold_k10_seed42.csv",
}


def resolve_default_raw_csv(task):
    return REPO_ROOT / "dataset" / task.lower() / "raw" / "smiles.csv"


def resolve_default_split_csv(task):
    default_path = TASK_TO_SPLIT_CSV.get(task.lower())
    if default_path is not None and default_path.exists():
        return default_path
    return None


def load_raw_task_data(raw_csv_path):
    df = pd.read_csv(raw_csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"raw csv must contain at least 2 columns (smiles + labels), got shape={df.shape}")
    smiles_raw = df.iloc[:, 0].astype(str).tolist()
    labels_raw = df.iloc[:, 1:].values.astype(np.float32)

    smiles_all = []
    labels_all = []
    source_indices = []
    invalid_count = 0
    for idx, smile in enumerate(smiles_raw):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            invalid_count += 1
            continue
        smiles_all.append(smile)
        labels_all.append(labels_raw[idx])
        source_indices.append(idx)

    if not smiles_all:
        raise RuntimeError(f"no valid molecules found in raw csv: {raw_csv_path}")

    labels_all = np.asarray(labels_all, dtype=np.float32)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if invalid_count:
        print(f"[build] filtered_invalid_smiles={invalid_count} from {raw_csv_path}")
    return smiles_all, labels_all, source_indices


def build_split_indices(smiles_all, labels_all, split_csv_path=None, split_seed=42):
    if split_csv_path is not None:
        split_df = pd.read_csv(split_csv_path)
        required = {"smiles", "fold_id"}
        if not required.issubset(split_df.columns):
            raise ValueError(f"split csv must contain columns {sorted(required)}, got {split_df.columns.tolist()}")
        split_smiles = split_df["smiles"].astype(str).tolist()
        if split_smiles != smiles_all:
            raise ValueError("split csv smiles order does not match raw csv order")
        fold_ids = split_df["fold_id"].values.astype(int)
        test_idx = np.where(fold_ids == 0)[0]
        val_idx = np.where(fold_ids == 1)[0]
        train_idx = np.where(~((fold_ids == 0) | (fold_ids == 1)))[0]
        return train_idx, val_idx, test_idx, "fold_id"

    dataset = dc.data.NumpyDataset(
        X=np.zeros((len(smiles_all), 1), dtype=np.float32),
        y=labels_all,
        w=np.ones_like(labels_all, dtype=np.float32),
        ids=np.array(smiles_all),
    )
    splitter = dc.splits.ScaffoldSplitter()
    train_idx, val_idx, test_idx = splitter.split(
        dataset,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=split_seed,
    )
    return np.array(train_idx), np.array(val_idx), np.array(test_idx), "scaffold_split"


def build_balanced_weights(smiles_all, labels_all, train_idx, val_idx, test_idx):
    from utils_fp_downstream import CombinedFingerprintsFeaturizer

    fps_all = CombinedFingerprintsFeaturizer().featurize(smiles_all)
    w_init = np.ones_like(labels_all, dtype=np.float32)
    ds_all = dc.data.NumpyDataset(X=fps_all, y=labels_all, w=w_init, ids=np.array(smiles_all))
    train_ds = ds_all.select(train_idx)
    valid_ds = ds_all.select(val_idx)
    test_ds = ds_all.select(test_idx)
    balancer = dc.trans.BalancingTransformer(dataset=train_ds)
    train_ds = balancer.transform(train_ds)
    valid_ds = balancer.transform(valid_ds)
    test_ds = balancer.transform(test_ds)
    w_all = np.zeros_like(labels_all, dtype=np.float32)
    w_all[train_idx] = train_ds.w
    w_all[val_idx] = valid_ds.w
    w_all[test_idx] = test_ds.w
    return fps_all, w_all


def assert_graph_integrity(data):
    num_nodes = data.x.size(0)
    if hasattr(data, "edge_weight_3d"):
        expected_edges = num_nodes * (num_nodes - 1)
        assert data.edge_index.size(1) == expected_edges
        assert data.edge_weight_3d.numel() == expected_edges
        assert torch.isfinite(data.edge_weight_3d).all()
        print(
            f"[build] mol_id={getattr(data, 'mol_id', 'na')} N={num_nodes} E={data.edge_index.size(1)} "
            f"edge_weight_3d_ok=True"
        )
    else:
        assert data.edge_index.dim() == 2 and data.edge_index.size(0) == 2
        print(f"[build] mol_id={getattr(data, 'mol_id', 'na')} N={num_nodes} bond_E={data.edge_index.size(1)}")


def build_data_list(mode, indices, smiles_all, labels_all, fps_all, w_all, source_indices, pair_rep_dir, sdf_path):
    data_list = []
    for mol_index in indices:
        smile = smiles_all[mol_index]
        label = labels_all[mol_index].astype(np.float32)
        weight = w_all[mol_index].astype(np.float32)
        source_index = int(source_indices[mol_index])
        data, atoms = build_graph_data(
            smile=smile,
            label=label,
            weight=weight,
            fingerprint=fps_all[mol_index],
        )
        data.mol_id = source_index
        if mode == "ctrl":
            num_nodes = data.x.size(0)
            edge_index_full = build_all_pairs_edge_index(num_nodes)
            data.edge_index = edge_index_full
            edge_attr_dim = data.edge_attr.size(1) if data.edge_attr.dim() == 2 else 0
            data.edge_attr = torch.zeros((edge_index_full.size(1), edge_attr_dim), dtype=torch.float32)
            data.edge_weight_3d = torch.ones(edge_index_full.size(1), dtype=torch.float32)
        elif mode == "bondpair":
            pair_payload = load_pair_payload(pair_rep_dir, sdf_path, source_index)
            keep_atom_indices = pair_payload["keep_atom_indices"]
            pair_rep = pair_payload["encoder_pair_rep"]
            keep_atom_indices = torch.as_tensor(keep_atom_indices, dtype=torch.long)
            pair_rep = pair_rep.index_select(0, keep_atom_indices).index_select(1, keep_atom_indices)
            data.pair_rep_flat = pair_rep.reshape(pair_rep.size(0) * pair_rep.size(1), pair_rep.size(2)).to(torch.float32)
            data.unimol_num_heads = int(pair_rep.size(-1))
        elif mode in {"unimol", "unimolupd"}:
            pair_payload = load_pair_payload(pair_rep_dir, sdf_path, source_index)
            data = attach_unimol_pair_to_data(data, atoms, pair_payload)
        assert_graph_integrity(data)
        data_list.append(data)
    return data_list


def build_mode(task, mode, raw_csv_path, split_csv_path, sdf_path, pair_rep_dir, out_root, split_seed):
    smiles_all, labels_all, source_indices = load_raw_task_data(raw_csv_path)
    train_idx, val_idx, test_idx, split_source = build_split_indices(
        smiles_all=smiles_all,
        labels_all=labels_all,
        split_csv_path=split_csv_path,
        split_seed=split_seed,
    )
    fps_all, w_all = build_balanced_weights(smiles_all, labels_all, train_idx, val_idx, test_idx)

    processed_dir = out_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] task={task}")
    print(f"[build] mode={mode}")
    print(f"[build] raw_csv_path={raw_csv_path}")
    print(f"[build] split_csv_path={split_csv_path}")
    print(f"[build] split_source={split_source}")
    print(f"[build] sdf_path={sdf_path}")
    print(f"[build] pair_rep_dir={pair_rep_dir}")
    print(f"[build] out_root={out_root}")
    print(f"[build] train={len(train_idx)} valid={len(val_idx)} test={len(test_idx)}")

    split_plan = [
        ("train", train_idx),
        ("valid", val_idx),
        ("test", test_idx),
        ("all", np.arange(len(smiles_all))),
    ]

    for split_name, split_indices in split_plan:
        data_list = build_data_list(
            mode,
            split_indices,
            smiles_all,
            labels_all,
            fps_all,
            w_all,
            source_indices,
            pair_rep_dir,
            sdf_path,
        )
        data_obj, slices = InMemoryDataset.collate(data_list)
        output_path = processed_dir / f"{task}_{split_name}.pt"
        torch.save((data_obj, slices), output_path)
        print(
            f"[build] saved split={split_name} graphs={len(data_list)} "
            f"path={output_path}"
        )

    print(f"[build] finished task={task} mode={mode}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="bace")
    parser.add_argument("--mode", choices=["2d", "ctrl", "bondpair", "unimol", "unimolupd", "all"], default="all")
    parser.add_argument("--raw-csv", type=str, default=None)
    parser.add_argument("--split-csv", type=str, default=None)
    parser.add_argument("--sdf-path", type=str, default=None)
    parser.add_argument("--pair-rep-dir", type=str, default=None)
    parser.add_argument("--base-out-dir", type=str, default=str(DEFAULT_BENCH_ROOT))
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task.lower()
    raw_csv_path = Path(args.raw_csv).expanduser().resolve() if args.raw_csv else resolve_default_raw_csv(task)
    split_csv_default = resolve_default_split_csv(task)
    split_csv_path = Path(args.split_csv).expanduser().resolve() if args.split_csv else split_csv_default
    sdf_path = resolve_sdf_path(sdf_path_arg=args.sdf_path)
    pair_rep_dir = Path(args.pair_rep_dir).expanduser().resolve() if args.pair_rep_dir else env_path("PAIR_REP_DIR", DEFAULT_PAIR_REP_DIR)
    base_out_dir = Path(args.base_out_dir).expanduser().resolve()

    modes = ["2d", "ctrl", "bondpair", "unimol", "unimolupd"] if args.mode == "all" else [args.mode]
    for mode in modes:
        out_root = base_out_dir / task / MODE_TO_SUBDIR[mode]
        build_mode(
            task=task,
            mode=mode,
            raw_csv_path=raw_csv_path,
            split_csv_path=split_csv_path,
            sdf_path=sdf_path,
            pair_rep_dir=pair_rep_dir,
            out_root=out_root,
            split_seed=args.split_seed,
        )


if __name__ == "__main__":
    main()
