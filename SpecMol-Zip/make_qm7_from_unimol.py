"""Build down_task_qm7_unimol_v2/ using Uni-Mol pair_repr (mirror of make_freesolv_unimol_data.py).

Prereqs (in order):
  1. python prepare_qm7_unimol_input.py
  2. python tools/extract_unimol_pair.py \
        --sdf-path unimol_out_qm7/sdf/qm7_for_unimol.sdf \
        --output-dir unimol_out_qm7/pair_rep \
        --ckpt-path $UNIMOL_CKPT --dict-path $UNIMOL_DICT

This reads the manifest + pair_rep payloads and produces:
  down_task_qm7_unimol_v2/processed/qm7_{train,valid,test,all}.pt

Atom-order alignment is automatic: the SDF was embedded from the SAME SMILES that
smile_to_graph parses here, so the graph heavy-atom order matches the Uni-Mol
heavy-atom order (keep_atom_indices), same as the FreeSolv pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from create_data_DC import smile_to_graph
from utils_fp_downstream import CombinedFingerprintsFeaturizer
from make_task_from_unimol import attach_unimol_pair, resolve_pair_rep_path

REPO = Path(__file__).resolve().parent
UNIMOL_OUT = REPO / "unimol_out_qm7"
MANIFEST_CSV = UNIMOL_OUT / "manifest.csv"
PAIR_REP_DIR = UNIMOL_OUT / "pair_rep"
SDF_STEM = "qm7_for_unimol"
OUTPUT_DIR = REPO / "down_task_qm7_unimol_v2" / "processed"


def build_graph_data_regression(smile, label, fingerprint):
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


def build_split(manifest_split, fps_by_smile):
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
            pair_path = resolve_pair_rep_path(PAIR_REP_DIR, SDF_STEM, sdf_index)
            payload = torch.load(pair_path, map_location="cpu")
            data = attach_unimol_pair(data, atoms, payload)
            data.mol_id = sdf_index
            data_list.append(data)
        except Exception as exc:
            n_fail += 1
            print(f"  skip mol idx={sdf_index} smiles={smile[:40]}: {exc}")
    return data_list, n_fail


def main():
    if not MANIFEST_CSV.exists():
        sys.exit(f"Missing {MANIFEST_CSV}. Run prepare_qm7_unimol_input.py first.")
    if not PAIR_REP_DIR.exists():
        sys.exit(f"Missing {PAIR_REP_DIR}. Run tools/extract_unimol_pair.py first.")

    manifest = pd.read_csv(MANIFEST_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    smiles_all = manifest["smiles"].astype(str).tolist()
    featurizer = CombinedFingerprintsFeaturizer()
    fps_all = featurizer.featurize(smiles_all)
    fps_by_smile = {s: fps_all[i] for i, s in enumerate(smiles_all)}

    all_data_list = []
    for split in ("train", "valid", "test"):
        sub = manifest[manifest["split"] == split]
        print(f"Building {split}: {len(sub)} mols (manifest)")
        data_list, n_fail = build_split(sub, fps_by_smile)
        if not data_list:
            sys.exit(f"No data for split {split}")
        if n_fail:
            print(f"  ({n_fail} failed)")
        collated, slices = InMemoryDataset.collate(data_list)
        save_path = OUTPUT_DIR / f"qm7_{split}.pt"
        torch.save((collated, slices), save_path)
        print(f"  saved {len(data_list)} graphs -> {save_path}")
        all_data_list.extend(data_list)

    collated, slices = InMemoryDataset.collate(all_data_list)
    torch.save((collated, slices), OUTPUT_DIR / "qm7_all.pt")
    print(f"  saved {len(all_data_list)} graphs -> {OUTPUT_DIR / 'qm7_all.pt'}")

    sample = all_data_list[0]
    pair_dim = int(sample.pair_repr_edge.size(-1))
    print(f"\nDone. pair_repr_dim={pair_dim}. "
          f"n_total={len(all_data_list)}")


if __name__ == "__main__":
    main()
