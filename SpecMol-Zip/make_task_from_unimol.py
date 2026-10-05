"""Build PyG datasets from Uni-Mol outputs for any classification task.

Generalizes make_bace_from_unimol.py to work with any MolNet classification
task (bbbp, bace, hiv, clintox, tox21).

Modes:
  --mode v2       : Include pair_repr_edge + pair_edge_index (for V2-T5 / T6)
  --mode baseline : 2D-only graphs on the same Uni-Mol fold split (fair V0)

All variants share the same Uni-Mol scaffold k-fold split for fair comparison.

Usage:
    # Step 1: Run export_unimol_task.py + extract_unimol_pair.py first
    # Step 2: Build datasets
    python make_task_from_unimol.py --task bbbp --mode v2
    python make_task_from_unimol.py --task bbbp --mode baseline
"""

import argparse
import os
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from create_data_DC import smile_to_graph
from utils_fp_downstream import CombinedFingerprintsFeaturizer


REPO_ROOT = Path(__file__).resolve().parent
FOLD = 0


def env_path(name, default):
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


def build_all_pairs_edge_index(num_nodes):
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    src = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, dtype=torch.long).repeat(num_nodes)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def resolve_pair_rep_path(pair_rep_dir, sdf_stem, mol_index):
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
        f"pair_rep.pt not found for mol {mol_index}. Tried: {[str(p) for p in candidates]}"
    )


def infer_keep_atom_indices(pair_payload, atoms_from_graph):
    if "keep_atom_indices" in pair_payload:
        return torch.as_tensor(pair_payload["keep_atom_indices"], dtype=torch.long)

    atomic_symbols = pair_payload.get("atomic_symbols")
    if atomic_symbols is None:
        if pair_payload["encoder_pair_rep"].size(0) == len(atoms_from_graph):
            return torch.arange(len(atoms_from_graph), dtype=torch.long)
        raise ValueError("pair_rep payload missing keep_atom_indices or atomic_symbols")

    heavy_indices = [i for i, s in enumerate(atomic_symbols) if s != "H"]
    if len(heavy_indices) != len(atoms_from_graph):
        raise ValueError(
            f"Heavy-atom count mismatch: {len(heavy_indices)} vs {len(atoms_from_graph)}"
        )
    return torch.tensor(heavy_indices, dtype=torch.long)


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


def attach_unimol_pair(data, atoms, pair_payload):
    num_nodes = data.x.size(0)
    keep_atom_indices = infer_keep_atom_indices(pair_payload, atoms)
    encoder_pair_rep = torch.as_tensor(pair_payload["encoder_pair_rep"], dtype=torch.float32)
    if encoder_pair_rep.dim() == 4 and encoder_pair_rep.size(0) == 1:
        encoder_pair_rep = encoder_pair_rep.squeeze(0)
    if encoder_pair_rep.dim() != 3:
        raise ValueError(f"encoder_pair_rep must be 3D, got {tuple(encoder_pair_rep.shape)}")

    pair_repr_full = (
        encoder_pair_rep
        .index_select(0, keep_atom_indices)
        .index_select(1, keep_atom_indices)
        .contiguous()
    )
    if pair_repr_full.size(0) != num_nodes or pair_repr_full.size(1) != num_nodes:
        raise ValueError(
            f"pair_repr shape {tuple(pair_repr_full.shape)} vs num_nodes={num_nodes}"
        )

    full_edge_index = build_all_pairs_edge_index(num_nodes)
    data.pair_edge_index = full_edge_index
    data.pair_repr_edge = pair_repr_full[
        full_edge_index[0], full_edge_index[1]
    ].to(torch.float32)
    data.pair_repr_dim = int(pair_repr_full.size(-1))
    return data


def load_fold_csv(csv_path):
    df = pd.read_csv(csv_path)
    smiles_all = df["smiles"].astype(str).tolist()
    fold_ids = df["fold_id"].values.astype(int)

    # Detect label columns
    label_cols = [c for c in df.columns if c.startswith("label")]
    labels_all = df[label_cols].values.astype(np.float32)

    return smiles_all, labels_all, fold_ids


def split_by_fold(smiles_all, labels_all, fold_ids, fold):
    kfold = fold_ids.max() + 1
    test_fold = fold
    val_fold = (fold + 1) % kfold

    test_idx = np.where(fold_ids == test_fold)[0]
    val_idx = np.where(fold_ids == val_fold)[0]
    train_idx = np.where(~((fold_ids == test_fold) | (fold_ids == val_fold)))[0]

    return train_idx, val_idx, test_idx


def balance_weights(smiles_all, labels_all, train_idx, val_idx, test_idx):
    fps_all = CombinedFingerprintsFeaturizer().featurize(smiles_all)
    y_all = labels_all.copy()
    if y_all.ndim == 1:
        y_all = y_all.reshape(-1, 1)
    w_init = np.ones_like(y_all, dtype=np.float32)
    ds_all = dc.data.NumpyDataset(
        X=fps_all, y=y_all, w=w_init, ids=np.array(smiles_all)
    )

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

    return fps_all, w_all


def build_and_save(task, mode, smiles_all, labels_all, fps_all, w_all,
                   train_idx, val_idx, test_idx, output_dir,
                   pair_rep_dir=None, sdf_stem=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_pair = (mode == "v2")

    def build_data_list(indices):
        data_list = []
        n_fail = 0
        for mol_index in indices:
            smile = smiles_all[mol_index]
            label = labels_all[mol_index]
            if label.ndim == 0:
                label = np.array([label], dtype=np.float32)
            weight = w_all[mol_index].astype(np.float32)
            try:
                data, atoms = build_graph_data(smile, label, weight, fps_all[mol_index])
                if include_pair:
                    pair_path = resolve_pair_rep_path(pair_rep_dir, sdf_stem, mol_index)
                    payload = torch.load(pair_path, map_location="cpu")
                    if "encoder_pair_rep" not in payload:
                        raise KeyError(f"Missing encoder_pair_rep in {pair_path}")
                    data = attach_unimol_pair(data, atoms, payload)
                data.mol_id = int(mol_index)
                data_list.append(data)
            except Exception as exc:
                n_fail += 1
                print(f"  skip mol {mol_index}: {smile[:40]} -> {exc}")
        if n_fail:
            print(f"  ({n_fail} molecules failed)")
        return data_list

    splits = [
        ("train", train_idx),
        ("valid", val_idx),
        ("test", test_idx),
    ]

    all_data_list = []
    for split_name, indices in splits:
        print(f"Building {split_name} ({len(indices)} mols)...")
        data_list = build_data_list(indices)
        if not data_list:
            raise RuntimeError(f"No valid data for split {split_name}")
        collated, slices = InMemoryDataset.collate(data_list)
        save_path = output_dir / f"{task}_{split_name}.pt"
        torch.save((collated, slices), save_path)
        print(f"  saved {len(data_list)} graphs -> {save_path}")
        all_data_list.extend(data_list)

    # Save 'all'
    collated, slices = InMemoryDataset.collate(all_data_list)
    save_path = output_dir / f"{task}_all.pt"
    torch.save((collated, slices), save_path)
    print(f"  saved {len(all_data_list)} graphs -> {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build PyG datasets from Uni-Mol outputs")
    parser.add_argument("--task", required=True, help="Task name (bbbp, bace, hiv, ...)")
    parser.add_argument("--mode", required=True, choices=["v2", "baseline"],
                        help="v2=with pair_repr, baseline=2D only (same fold split)")
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--csv-path", default=None,
                        help="Path to fold CSV (default: unimol_out_{task}/splits/data_with_folds_*.csv)")
    parser.add_argument("--pair-rep-dir", default=None,
                        help="Dir containing pair_rep.pt files (mode=v2 only)")
    parser.add_argument("--sdf-stem", default=None,
                        help="SDF file stem for pair_rep lookup (default: auto-detect)")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: down_task_{task}_{v2|unifold}/processed)")
    return parser.parse_args()


def auto_detect_csv(task):
    unimol_dir = REPO_ROOT / f"unimol_out_{task}" / "splits"
    if not unimol_dir.exists():
        # Fall back to original unimol_out (BACE)
        unimol_dir = REPO_ROOT / "unimol_out" / "splits"
    candidates = sorted(unimol_dir.glob("data_with_folds_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No fold CSV found in {unimol_dir}")
    return candidates[0]


def auto_detect_pair_rep_dir(task):
    unimol_dir = REPO_ROOT / f"unimol_out_{task}" / "pair_rep"
    if not unimol_dir.exists():
        unimol_dir = REPO_ROOT / "unimol_out" / "pair_rep"
    if not unimol_dir.exists():
        raise FileNotFoundError(f"No pair_rep dir found for task={task}")
    return unimol_dir


def auto_detect_sdf_stem(task):
    sdf_dir = REPO_ROOT / f"unimol_out_{task}" / "sdf"
    if not sdf_dir.exists():
        sdf_dir = REPO_ROOT / "unimol_out" / "sdf"
    sdfs = sorted(sdf_dir.glob("*.sdf"))
    if sdfs:
        return sdfs[0].stem
    return f"{task}_for_unimol"


def main():
    args = parse_args()
    task = args.task.lower()
    mode = args.mode

    # Resolve paths
    csv_path = Path(args.csv_path) if args.csv_path else auto_detect_csv(task)

    if mode == "v2":
        pair_rep_dir = Path(args.pair_rep_dir) if args.pair_rep_dir else auto_detect_pair_rep_dir(task)
        sdf_stem = args.sdf_stem or auto_detect_sdf_stem(task)
    else:
        pair_rep_dir = None
        sdf_stem = None

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "v2" if mode == "v2" else "unifold"
        output_dir = REPO_ROOT / f"down_task_{task}_{suffix}" / "processed"

    print(f"task={task}, mode={mode}, fold={args.fold}")
    print(f"csv_path={csv_path}")
    if pair_rep_dir:
        print(f"pair_rep_dir={pair_rep_dir}")
        print(f"sdf_stem={sdf_stem}")
    print(f"output_dir={output_dir}")

    # Load and split
    smiles_all, labels_all, fold_ids = load_fold_csv(csv_path)
    train_idx, val_idx, test_idx = split_by_fold(smiles_all, labels_all, fold_ids, args.fold)

    print(f"n={len(smiles_all)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Compute fingerprints and balance weights
    fps_all, w_all = balance_weights(smiles_all, labels_all, train_idx, val_idx, test_idx)

    # Build and save
    build_and_save(
        task=task,
        mode=mode,
        smiles_all=smiles_all,
        labels_all=labels_all,
        fps_all=fps_all,
        w_all=w_all,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        output_dir=output_dir,
        pair_rep_dir=pair_rep_dir,
        sdf_stem=sdf_stem,
    )

    print(f"\nDone! {mode} dataset for {task} saved to {output_dir}")


if __name__ == "__main__":
    main()
