"""Export a MolNet classification task for Uni-Mol processing.

For each task, this script:
1. Reads the raw CSV (no header: smiles, label(s))
2. Feeds to Uni-Mol DataHub for 3D conformer generation + scaffold k-fold split
3. Saves: fold CSV, fold_ids npy, SDF, unimol_input npy

Prerequisites:
  - dataset/{task}/raw/smiles.csv must exist (run prepare_molnet_dataset.py first)
  - unimol_tools must be installed in the conda env

Usage:
    python export_unimol_task.py --task bbbp
    python export_unimol_task.py --task bbbp --kfold 10 --split-seed 42
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from unimol_tools.data.datahub import DataHub

RDLogger.logger().setLevel(RDLogger.CRITICAL)


REPO_ROOT = Path(__file__).resolve().parent

SINGLE_LABEL_TASKS = {"bace", "bbbp", "hiv"}
MULTI_LABEL_TASKS = {"tox21": 12, "clintox": 2, "sider": 27}
ALL_TASKS = SINGLE_LABEL_TASKS | set(MULTI_LABEL_TASKS)


def parse_args():
    parser = argparse.ArgumentParser(description="Export MolNet task for Uni-Mol")
    parser.add_argument("--task", required=True, choices=sorted(ALL_TASKS))
    parser.add_argument("--kfold", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-name", default="unimolv2")
    parser.add_argument("--output-root", default=None,
                        help="Base output dir (default: unimol_out_{task})")
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task.lower()

    raw_csv = REPO_ROOT / "dataset" / task / "raw" / "smiles.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run: python prepare_molnet_dataset.py --task {task}"
        )

    output_root = (
        Path(args.output_root) if args.output_root
        else REPO_ROOT / f"unimol_out_{task}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # Read raw CSV (no header: smiles, labels...)
    df_raw = pd.read_csv(raw_csv, header=None)

    if task in SINGLE_LABEL_TASKS:
        df_raw.columns = ["smiles", "label"]
        target_cols = ["label"]
    else:
        n_labels = MULTI_LABEL_TASKS[task]
        cols = ["smiles"] + [f"label_{i}" for i in range(n_labels)]
        df_raw.columns = cols
        target_cols = cols[1:]

    # Filter RDKit-parseable SMILES. Unimol_tools' datareader silently drops
    # "illegal" SMILES (e.g. Tox21 contains 8 [AlH3] organometallics) but does
    # not update raw_data, causing a downstream length mismatch in save_mol2sdf
    # ("Length of values (7823) does not match length of index (7831)").
    # Pre-filtering here keeps the DataFrame and the conformer list in sync.
    n_before = len(df_raw)
    rdkit_ok = df_raw["smiles"].apply(
        lambda s: Chem.MolFromSmiles(str(s)) is not None
    )
    df_raw = df_raw[rdkit_ok].reset_index(drop=True)
    n_after = len(df_raw)
    if n_after < n_before:
        print(f"[filter] dropped {n_before - n_after}/{n_before} SMILES "
              f"that RDKit cannot parse")

    # Save as proper CSV with header for DataHub
    csv_for_unimol = output_root / f"{task}_for_unimol.csv"
    df_raw.to_csv(csv_for_unimol, index=False)
    print(f"Saved {len(df_raw)} rows to {csv_for_unimol}")

    # Multi-label tasks (clintox/tox21/sider) trip unimol_tools' internal
    # StratifiedKFold splitter (sklearn flattens 2D y to (n*k,) which then
    # mismatches X length). Scaffold split itself only needs the SMILES;
    # the label is just along for the ride. So for multi-label we hand
    # DataHub a single representative label column for the split, then
    # save the full multi-label df_raw with fold IDs ourselves below.
    split_target_cols = (
        [target_cols[0]] if task in MULTI_LABEL_TASKS else target_cols
    )

    # Run DataHub: generates 3D conformers + scaffold split
    sdf_save_path = str(output_root / "sdf")
    hub = DataHub(
        data=str(csv_for_unimol),
        is_train=True,
        save_path=str(output_root),
        task="classification",
        data_type="molecule",
        smiles_col="smiles",
        target_cols=split_target_cols,
        split="scaffold",
        kfold=args.kfold,
        split_seed=args.split_seed,
        model_name=args.model_name,
        conf_cache_level=2,
        sdf_save_path=sdf_save_path,
    )

    # Save fold assignments
    splits_dir = output_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_nfolds = hub.data["split_nfolds"]
    kfold = hub.kfold
    method = hub.method
    split_seed = hub.split_seed

    n_samples = len(df_raw)
    fold_ids = np.zeros(n_samples, dtype=int)
    for fold, (tr_idx, te_idx) in enumerate(split_nfolds):
        fold_ids[te_idx] = fold

    np.save(
        splits_dir / f"fold_ids_{method}_k{kfold}_seed{split_seed}.npy",
        fold_ids,
    )

    df_raw["fold_id"] = fold_ids
    csv_out = splits_dir / f"data_with_folds_{method}_k{kfold}_seed{split_seed}.csv"
    df_raw.to_csv(csv_out, index=False)

    # Save unimol input features (for potential re-use)
    unimol_input_arr = np.array(hub.data["unimol_input"], dtype=object)
    np.save(
        splits_dir / f"unimol_input_{method}_k{kfold}_seed{split_seed}.npy",
        unimol_input_arr,
    )

    print(f"\nDone! task={task}, n={n_samples}, kfold={kfold}")
    print(f"  Fold CSV: {csv_out}")
    print(f"  SDF dir:  {sdf_save_path}")


if __name__ == "__main__":
    main()
