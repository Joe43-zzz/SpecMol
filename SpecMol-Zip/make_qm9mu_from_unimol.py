"""Build down_task_qm9mu_unimol_v2/ (QM9 dipole subset) -- mirror of make_qm7_from_unimol.py.

Prereqs:
  1. python prepare_qm9mu_unimol_input.py
  2. python tools/extract_unimol_pair.py --sdf-path unimol_out_qm9mu/sdf/qm9mu_for_unimol.sdf \
       --output-dir unimol_out_qm9mu/pair_rep --ckpt-path $UNIMOL_CKPT --dict-path $UNIMOL_DICT
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from create_data_DC import smile_to_graph
from utils_fp_downstream import CombinedFingerprintsFeaturizer
from make_task_from_unimol import attach_unimol_pair, resolve_pair_rep_path

REPO = Path(__file__).resolve().parent
# Dirs overridable via env so the SAME builder serves the MMFF cell (default) and
# the Stage-2 DFT cell (QM9MU_UNIMOL_OUT=unimol_out_qm9mu_dft
# QM9MU_OUTPUT_DIR=down_task_qm9mu_unimol_v2_dft). Graph build is unchanged
# (from SMILES), so DFT and MMFF cells share identical 2D featurization.
UNIMOL_OUT = REPO / os.environ.get("QM9MU_UNIMOL_OUT", "unimol_out_qm9mu")
MANIFEST_CSV = UNIMOL_OUT / "manifest.csv"
PAIR_REP_DIR = UNIMOL_OUT / "pair_rep"
SDF_STEM = "qm9mu_for_unimol"
OUTPUT_DIR = REPO / os.environ.get("QM9MU_OUTPUT_DIR", "down_task_qm9mu_unimol_v2") / "processed"


def build_graph_data_regression(smile, label, fingerprint):
    c_size, features, edge_index, edge_features, atoms = smile_to_graph(smile)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    elif edge_index.dim() == 2 and edge_index.size(0) != 2:
        edge_index = edge_index.t().contiguous()
    edge_attr = (torch.empty((edge_index.size(1), 0), dtype=torch.float32)
                 if edge_features is None else torch.tensor(edge_features, dtype=torch.float32))
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
        smile = str(row.smiles); label = float(row.label); sdf_index = int(row.sdf_index)
        try:
            data, atoms = build_graph_data_regression(smile, label, fps_by_smile[smile])
            payload = torch.load(resolve_pair_rep_path(PAIR_REP_DIR, SDF_STEM, sdf_index), map_location="cpu")
            data = attach_unimol_pair(data, atoms, payload)
            data.mol_id = sdf_index
            data_list.append(data)
        except Exception as exc:
            n_fail += 1
            print(f"  skip idx={sdf_index} {smile[:40]}: {exc}")
    return data_list, n_fail


def main():
    if not MANIFEST_CSV.exists():
        sys.exit(f"Missing {MANIFEST_CSV}. Run prepare_qm9mu_unimol_input.py first.")
    if not PAIR_REP_DIR.exists():
        sys.exit(f"Missing {PAIR_REP_DIR}. Run tools/extract_unimol_pair.py first.")
    manifest = pd.read_csv(MANIFEST_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    smiles_all = manifest["smiles"].astype(str).tolist()
    fps_all = CombinedFingerprintsFeaturizer().featurize(smiles_all)
    fps_by_smile = {s: fps_all[i] for i, s in enumerate(smiles_all)}

    all_data = []
    for split in ("train", "valid", "test"):
        sub = manifest[manifest["split"] == split]
        print(f"Building {split}: {len(sub)} mols")
        dl, nf = build_split(sub, fps_by_smile)
        if not dl:
            sys.exit(f"No data for split {split}")
        if nf:
            print(f"  ({nf} failed)")
        collated, slices = InMemoryDataset.collate(dl)
        torch.save((collated, slices), OUTPUT_DIR / f"qm9mu_{split}.pt")
        print(f"  saved {len(dl)} -> qm9mu_{split}.pt")
        all_data.extend(dl)
    collated, slices = InMemoryDataset.collate(all_data)
    torch.save((collated, slices), OUTPUT_DIR / "qm9mu_all.pt")
    print(f"  saved {len(all_data)} -> qm9mu_all.pt; pair_dim={int(all_data[0].pair_repr_edge.size(-1))}")


if __name__ == "__main__":
    main()
