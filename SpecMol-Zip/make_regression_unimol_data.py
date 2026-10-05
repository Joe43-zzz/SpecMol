"""Build down_task_{task}_unimol_v2/ using Uni-Mol pair_repr for regression tasks.

Generic version of make_freesolv_unimol_data.py. Routes by --task.

Prereqs (run in order, latter two on HPC):
  1. python prepare_regression_unimol_input.py --task {esol|lipo|freesolv}
  2. python tools/extract_unimol_pair.py \
        --sdf-path unimol_out_{task}/sdf/{task}_for_unimol.sdf \
        --output-dir unimol_out_{task}/pair_rep \
        --ckpt-path $UNIMOL_CKPT --dict-path $UNIMOL_DICT
  3. python make_regression_unimol_data.py --task {esol|lipo|freesolv}

Output:
  down_task_{task}_unimol_v2/processed/{task}_{train,valid,test,all}.pt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from create_data_DC import smile_to_graph
from utils_fp_downstream import CombinedFingerprintsFeaturizer

from make_task_from_unimol import (
    attach_unimol_pair,
    resolve_pair_rep_path,
)


REPO = Path(__file__).resolve().parent

REGRESSION_TASKS = ("freesolv", "esol", "lipo")


def build_graph_data_regression(smile, label, fingerprint):
    """Mirror prepare_freesolv_data.build_graph_data but with w=1.0 (regression)."""
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
        y=torch.tensor([float(label)], dtype=torch.float32),
        edge_attr=edge_attr,
        w=torch.ones(1, dtype=torch.float32),
        fps=torch.tensor(fingerprint, dtype=torch.float32),
    )
    return data, atoms


def build_split(manifest_split, fps_by_smile, pair_rep_dir, sdf_stem):
    data_list = []
    n_fail = 0
    for row in manifest_split.itertuples():
        if not bool(row.embed_ok):
            n_fail += 1
            continue
        smile = str(row.smiles)
        label = float(row.label)
        sdf_index = int(row.sdf_index)
        fp = fps_by_smile[smile]
        try:
            data, atoms = build_graph_data_regression(smile, label, fp)
            pair_path = resolve_pair_rep_path(pair_rep_dir, sdf_stem, sdf_index)
            payload = torch.load(pair_path, map_location="cpu")
            data = attach_unimol_pair(data, atoms, payload)
            data.mol_id = sdf_index
            data_list.append(data)
        except Exception as exc:
            n_fail += 1
            print(f"  skip mol idx={sdf_index} smiles={smile[:40]}: {exc}")
    return data_list, n_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=REGRESSION_TASKS)
    args = parser.parse_args()
    task = args.task

    unimol_out = REPO / f"unimol_out_{task}"
    manifest_csv = unimol_out / "manifest.csv"
    pair_rep_dir = unimol_out / "pair_rep"
    sdf_stem = f"{task}_for_unimol"
    output_dir = REPO / f"down_task_{task}_unimol_v2" / "processed"

    if not manifest_csv.exists():
        sys.exit(
            f"Missing {manifest_csv}. Run "
            f"`python prepare_regression_unimol_input.py --task {task}` first."
        )
    if not pair_rep_dir.exists():
        sys.exit(
            f"Missing {pair_rep_dir}. Run tools/extract_unimol_pair.py on HPC first."
        )

    manifest = pd.read_csv(manifest_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_all = manifest["smiles"].astype(str).tolist()
    fps_all = CombinedFingerprintsFeaturizer().featurize(smiles_all)
    fps_by_smile = {s: fps_all[i] for i, s in enumerate(smiles_all)}

    all_data_list = []
    for split in ("train", "valid", "test"):
        sub = manifest[manifest["split"] == split]
        print(f"[{task}] Building {split}: {len(sub)} mols (manifest)")
        data_list, n_fail = build_split(sub, fps_by_smile, pair_rep_dir, sdf_stem)
        if not data_list:
            sys.exit(f"No data for split {split}")
        if n_fail:
            print(f"  ({n_fail} failed)")
        collated, slices = InMemoryDataset.collate(data_list)
        save_path = output_dir / f"{task}_{split}.pt"
        torch.save((collated, slices), save_path)
        print(f"  saved {len(data_list)} graphs -> {save_path}")
        all_data_list.extend(data_list)

    collated, slices = InMemoryDataset.collate(all_data_list)
    save_path = output_dir / f"{task}_all.pt"
    torch.save((collated, slices), save_path)
    print(f"  saved {len(all_data_list)} graphs -> {save_path}")

    sample = all_data_list[0]
    pair_dim = int(sample.pair_repr_edge.size(-1)) if all_data_list else -1
    print(
        f"\n[{task}] Done. pair_repr_dim={pair_dim} (Uni-Mol encoder_pair_rep dim). "
        f"Use --pair-dim {pair_dim} for run_regression_v2t5.py."
    )


if __name__ == "__main__":
    main()
