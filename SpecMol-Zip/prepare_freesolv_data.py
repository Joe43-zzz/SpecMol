"""Prepare FreeSolv data for both baseline and V2-T5 experiments.

Creates:
  - dataset/freesolv/raw/smiles.csv
  - down_task_freesolv_2d/processed/freesolv_{train,valid,test,all}.pt  (baseline)
  - down_task_freesolv_v2/processed/freesolv_{train,valid,test,all}.pt  (V2 with GBF pair_repr)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data, InMemoryDataset

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from rdkit import Chem
from rdkit.Chem import AllChem

import deepchem as dc
from deepchem.splits import ScaffoldSplitter

from create_data_DC import smile_to_graph
from utils_fp_downstream import CombinedFingerprintsFeaturizer


REPO = Path(__file__).resolve().parent
GBF_NUM_GAUSSIANS = 64
GBF_MAX_DIST = 10.0
GBF_SIGMA = 0.5


def gaussian_basis_expansion(distances, num_gaussians=GBF_NUM_GAUSSIANS,
                              max_dist=GBF_MAX_DIST, sigma=GBF_SIGMA):
    """Expand pairwise distances to Gaussian basis features.

    Args:
        distances: [N, N] pairwise distance matrix

    Returns:
        [N, N, num_gaussians] GBF features
    """
    centers = torch.linspace(0, max_dist, num_gaussians)
    # distances: [N, N] -> [N, N, 1], centers: [num_gaussians]
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0).unsqueeze(0)
    return torch.exp(-diff ** 2 / (2 * sigma ** 2))


def generate_3d_coords(smile, n_atoms_expected):
    """Generate 3D coordinates for a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        mol_h = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
        if result == -1:
            result = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=42)
            if result == -1:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        mol_noH = Chem.RemoveHs(mol_h)
        if mol_noH.GetNumAtoms() != n_atoms_expected:
            return None
        conf = mol_noH.GetConformer()
        pos = torch.zeros(n_atoms_expected, 3)
        for idx in range(n_atoms_expected):
            p = conf.GetAtomPosition(idx)
            pos[idx] = torch.tensor([p.x, p.y, p.z])
        return pos
    except Exception:
        return None


def build_all_pairs_edge_index(num_nodes):
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    src = torch.arange(num_nodes).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes).repeat(num_nodes)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def build_graph_data(smile, label, fingerprint, include_pair_repr=False):
    """Build a PyG Data object from SMILES.

    Args:
        smile: SMILES string
        label: float label value
        fingerprint: numpy array of fingerprint features
        include_pair_repr: if True, generate GBF pair_repr from RDKit 3D

    Returns:
        Data object or None on failure
    """
    try:
        c_size, features, edge_index, edge_features, atoms = smile_to_graph(smile)
    except Exception:
        return None

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
        y=torch.tensor([label], dtype=torch.float32),
        edge_attr=edge_attr,
        w=torch.ones(1, dtype=torch.float32),
        fps=torch.tensor(fingerprint, dtype=torch.float32),
    )

    if include_pair_repr:
        num_nodes = c_size
        pos = generate_3d_coords(smile, num_nodes)
        if pos is None:
            return None

        # Compute pairwise distances
        dist_matrix = torch.cdist(pos, pos, p=2)  # [N, N]

        # GBF expansion -> [N, N, 64]
        pair_repr_full = gaussian_basis_expansion(dist_matrix)

        # Build all-pairs edge index
        full_edge_index = build_all_pairs_edge_index(num_nodes)

        data.pair_edge_index = full_edge_index
        data.pair_repr_edge = pair_repr_full[
            full_edge_index[0], full_edge_index[1]
        ].to(torch.float32)
        data.pair_repr_dim = GBF_NUM_GAUSSIANS

    return data


def download_freesolv():
    """Download FreeSolv using DeepChem and save as CSV."""
    csv_dir = REPO / "dataset" / "freesolv" / "raw"
    csv_path = csv_dir / "smiles.csv"

    if csv_path.exists():
        print(f"FreeSolv CSV already exists: {csv_path}")
        df = pd.read_csv(csv_path, header=None)
        return df.iloc[:, 0].tolist(), df.iloc[:, 1].values.astype(np.float32)

    csv_dir.mkdir(parents=True, exist_ok=True)

    # Load via DeepChem (downloads if needed)
    tasks, datasets, _ = dc.molnet.load_freesolv(featurizer='Raw', splitter=None)
    full_ds = datasets[0]

    smiles = [str(s) for s in full_ds.ids]
    labels = full_ds.y[:, 0].astype(np.float32)

    # Filter invalid molecules
    valid_smiles, valid_labels = [], []
    for s, l in zip(smiles, labels):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_smiles.append(s)
            valid_labels.append(l)

    valid_labels = np.array(valid_labels, dtype=np.float32)

    # Save CSV (no header, smiles + label)
    df = pd.DataFrame({'smiles': valid_smiles, 'label': valid_labels})
    df.to_csv(csv_path, index=False, header=False)
    print(f"Saved {len(valid_smiles)} molecules to {csv_path}")

    return valid_smiles, valid_labels


def scaffold_split(smiles, labels, seed=42):
    """Scaffold split into train/valid/test (80/10/10)."""
    fps = np.zeros((len(smiles), 1))  # dummy features for DeepChem
    w = np.ones((len(smiles), 1))
    y = labels.reshape(-1, 1)

    dataset = dc.data.NumpyDataset(X=fps, y=y, w=w, ids=np.array(smiles))
    splitter = ScaffoldSplitter()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
    )

    return train_ds, valid_ds, test_ds


def build_and_save(smiles_list, labels, output_dir, task_name, include_pair_repr=False):
    """Build PyG dataset and save .pt files."""
    output_dir = Path(output_dir)
    proc_dir = output_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Compute fingerprints for all
    featurizer = CombinedFingerprintsFeaturizer()
    fps_all = featurizer.featurize(smiles_list)

    # Scaffold split
    train_ds, valid_ds, test_ds = scaffold_split(smiles_list, labels)

    # Map SMILES to index
    smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}

    splits = {
        'train': train_ds,
        'valid': valid_ds,
        'test': test_ds,
    }

    all_data_list = []

    for split_name, ds in splits.items():
        data_list = []
        n_fail = 0
        for i in range(len(ds)):
            smile = str(ds.ids[i])
            label = float(ds.y[i, 0])
            idx = smiles_to_idx.get(smile)
            if idx is None:
                n_fail += 1
                continue
            fp = fps_all[idx]

            data = build_graph_data(smile, label, fp, include_pair_repr=include_pair_repr)
            if data is None:
                n_fail += 1
                continue
            data_list.append(data)

        if len(data_list) == 0:
            raise RuntimeError(f"No valid data for split {split_name}")

        collated, slices = InMemoryDataset.collate(data_list)
        save_path = proc_dir / f"{task_name}_{split_name}.pt"
        torch.save((collated, slices), save_path)
        print(f"  {split_name}: {len(data_list)} graphs saved to {save_path}" +
              (f" ({n_fail} failed)" if n_fail else ""))
        all_data_list.extend(data_list)

    # Save 'all' (full dataset, no split)
    collated, slices = InMemoryDataset.collate(all_data_list)
    save_path = proc_dir / f"{task_name}_all.pt"
    torch.save((collated, slices), save_path)
    print(f"  all: {len(all_data_list)} graphs saved to {save_path}")


def main():
    print("=" * 60)
    print("Step 1: Download/load FreeSolv data")
    print("=" * 60)
    smiles, labels = download_freesolv()
    print(f"FreeSolv: {len(smiles)} molecules, label range [{labels.min():.2f}, {labels.max():.2f}]")

    print("\n" + "=" * 60)
    print("Step 2: Build baseline (2D-only) dataset")
    print("=" * 60)
    build_and_save(
        smiles, labels,
        output_dir=REPO / "down_task_freesolv_2d",
        task_name="freesolv",
        include_pair_repr=False,
    )

    print("\n" + "=" * 60)
    print("Step 3: Build V2-T5 dataset (with GBF pair_repr)")
    print("=" * 60)
    build_and_save(
        smiles, labels,
        output_dir=REPO / "down_task_freesolv_v2",
        task_name="freesolv",
        include_pair_repr=True,
    )

    print("\nDone! Datasets saved.")


if __name__ == "__main__":
    main()
