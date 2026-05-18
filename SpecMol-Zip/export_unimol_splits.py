"""Export Uni-Mol fold splits + 3D inputs for a chosen MoleculeNet task.

Parametrized rewrite of the original BACE-only script. Supports binary
single-label (BACE, BBBP, HIV) and multi-label classification
(ClinTox 2 tasks, Tox21 12 tasks). Mirrors the DATASETS dict used by
`baselines_ml.py` / `baselines_chemprop.py` so the resulting fold CSV
plugs into the same downstream pipeline (see `make_*_from_unimol.py`).

Backward compat: `python export_unimol_splits.py` (no args) reproduces
the original BACE behavior exactly, writing to `unimol_out/splits/`.

Usage:
    python export_unimol_splits.py --task bace
    python export_unimol_splits.py --task clintox
    python export_unimol_splits.py --task tox21
    python export_unimol_splits.py --task bbbp --out-root unimol_out_bbbp

Output layout (per task):
    <out_root>/splits/data_with_folds_scaffold_k10_seed42.csv
        cols: smiles, <label_cols...>, fold_id
    <out_root>/splits/fold_ids_scaffold_k10_seed42.npy
        per-sample fold_id integers in [0, kfold)
    <out_root>/splits/unimol_input_scaffold_k10_seed42.npy
        UniMol DataHub 3D input objects (one per molecule, object dtype)
    <out_root>/sdf/smiles_unimol.sdf
        SDF of generated conformers (consumed by make_*_from_unimol.py)

Note: requires HPC conda env `specmol` for the `unimol_tools.data.datahub`
import. Local test of the prepare_csv step is fine; UniMol DataHub call
must run on HPC.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class TaskSpec:
    name: str
    raw_csv: str                       # path relative to repo root
    has_header: bool                   # True if raw CSV has a header row
    smiles_col: Union[int, str]        # int (column index) or str (column name)
    label_cols: List[str]              # output column names emitted into rewritten CSV
    raw_label_cols: Optional[List[Union[int, str]]] = None  # raw selector; defaults to label_cols
    task: str = "classification"
    out_root: Optional[str] = None     # defaults to "unimol_out" for bace, "unimol_out_<name>" else

    def resolve_out_root(self) -> str:
        if self.out_root is not None:
            return self.out_root
        return "unimol_out" if self.name == "bace" else f"unimol_out_{self.name}"


# Canonical MoleculeNet task-column names. These are emitted into the
# rewritten CSV; downstream `baselines_*.py` already reference them by index,
# so the names are mostly for human readability.
TOX21_TASK_NAMES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
CLINTOX_TASK_NAMES = ["FDA_APPROVED", "CT_TOX"]

TASK_SPECS = {
    "bace": TaskSpec(
        name="bace",
        raw_csv="dataset/bace/raw/smiles.csv",
        has_header=False,
        smiles_col=0,
        label_cols=["label"],
        raw_label_cols=[1],
    ),
    "bbbp": TaskSpec(
        name="bbbp",
        raw_csv="dataset/BBBP.csv",
        has_header=True,
        smiles_col="smiles",
        label_cols=["label"],
        raw_label_cols=["p_np"],
    ),
    "freesolv": TaskSpec(
        name="freesolv",
        raw_csv="dataset/freesolv/raw/smiles.csv",
        has_header=False,
        smiles_col=0,
        label_cols=["label"],
        raw_label_cols=[1],
        task="regression",
    ),
    "clintox": TaskSpec(
        name="clintox",
        raw_csv="dataset/clintox/raw/smiles.csv",
        has_header=False,
        smiles_col=0,
        label_cols=CLINTOX_TASK_NAMES,
        raw_label_cols=[1, 2],
    ),
    "tox21": TaskSpec(
        name="tox21",
        raw_csv="dataset/tox21/raw/smiles.csv",
        has_header=False,
        smiles_col=0,
        label_cols=TOX21_TASK_NAMES,
        raw_label_cols=list(range(1, 13)),
    ),
}


def prepare_csv(spec: TaskSpec, repo_root: Path) -> Path:
    """Read raw CSV, normalize headers/columns, write a headered copy with
    canonical (smiles, <label_cols...>) layout. Returns the rewritten path.

    Headerless -> assume positional cols; headered -> select by name.
    NaN label cells are preserved (UniMol DataHub masks missing labels
    in its multi-task pipeline).
    """
    raw_path = repo_root / spec.raw_csv
    if not raw_path.exists():
        raise FileNotFoundError(f"raw CSV not found: {raw_path}")
    raw_label_cols = spec.raw_label_cols or spec.label_cols

    if spec.has_header:
        df_raw = pd.read_csv(raw_path)
        smi = df_raw[spec.smiles_col].astype(str)
        labels = df_raw[list(raw_label_cols)]
    else:
        df_raw = pd.read_csv(raw_path, header=None)
        smi = df_raw.iloc[:, spec.smiles_col].astype(str)
        labels = df_raw.iloc[:, list(raw_label_cols)]
    labels.columns = spec.label_cols
    df_out = pd.concat([smi.rename("smiles"), labels.reset_index(drop=True)], axis=1)

    out_root = repo_root / spec.resolve_out_root()
    out_csv = out_root / "raw" / "smiles_unimol.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[{spec.name}] prepared CSV at {out_csv}")
    print(f"[{spec.name}]   columns: {list(df_out.columns)}")
    print(f"[{spec.name}]   rows: {len(df_out)}  label_cols: {spec.label_cols}")
    return out_csv


def run_datahub(spec: TaskSpec, csv_path: Path, repo_root: Path, kfold: int, split_seed: int,
                method: str):
    """Run UniMol DataHub on the prepared CSV. Returns the hub object."""
    from unimol_tools.data.datahub import DataHub  # HPC-only import
    out_root = repo_root / spec.resolve_out_root()
    sdf_dir = out_root / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    hub = DataHub(
        data=str(csv_path),
        is_train=True,
        save_path=str(out_root),
        task=spec.task,
        data_type="molecule",
        smiles_col="smiles",
        target_cols=spec.label_cols,
        split=method,
        kfold=kfold,
        split_seed=split_seed,
        model_name="unimolv2",
        conf_cache_level=2,
        sdf_save_path=str(sdf_dir),
    )
    return hub


def dump_splits(hub, spec: TaskSpec, df: pd.DataFrame, repo_root: Path):
    """Emit fold_ids.npy, data_with_folds_*.csv, unimol_input_*.npy."""
    out_root = repo_root / spec.resolve_out_root()
    out_dir = out_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_nfolds = hub.data["split_nfolds"]
    kfold = hub.kfold
    method = hub.method
    split_seed = hub.split_seed

    n_samples = len(df)
    fold_ids = np.zeros(n_samples, dtype=int)
    for fold, (_tr_idx, te_idx) in enumerate(split_nfolds):
        fold_ids[te_idx] = fold

    np.save(out_dir / f"fold_ids_{method}_k{kfold}_seed{split_seed}.npy", fold_ids)

    df_with_folds = df.copy()
    df_with_folds["fold_id"] = fold_ids
    csv_out = out_dir / f"data_with_folds_{method}_k{kfold}_seed{split_seed}.csv"
    df_with_folds.to_csv(csv_out, index=False)

    unimol_input_arr = np.array(hub.data["unimol_input"], dtype=object)
    np.save(out_dir / f"unimol_input_{method}_k{kfold}_seed{split_seed}.npy", unimol_input_arr)

    print(f"[{spec.name}] saved splits/ to {out_dir}")
    print(f"[{spec.name}]   data_with_folds: {csv_out.name}")
    print(f"[{spec.name}]   unimol_input:    unimol_input_{method}_k{kfold}_seed{split_seed}.npy")
    print(f"[{spec.name}]   fold sizes:      {dict(zip(*np.unique(fold_ids, return_counts=True)))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="bace", choices=sorted(TASK_SPECS.keys()))
    p.add_argument("--out-root", default=None,
                   help="Override default output directory. Defaults to 'unimol_out' for "
                        "bace (backward compat) or 'unimol_out_<task>' otherwise.")
    p.add_argument("--kfold", type=int, default=10)
    p.add_argument("--seed", type=int, default=42, dest="split_seed")
    p.add_argument("--method", default="scaffold")
    p.add_argument("--prepare-only", action="store_true",
                   help="Only run prepare_csv, skip DataHub call. Useful for local smoke tests "
                        "(DataHub requires HPC env).")
    args = p.parse_args()

    spec = TASK_SPECS[args.task]
    if args.out_root is not None:
        spec = TaskSpec(**{**spec.__dict__, "out_root": args.out_root})

    repo_root = Path(__file__).resolve().parent
    csv_path = prepare_csv(spec, repo_root)

    if args.prepare_only:
        print(f"[{spec.name}] --prepare-only set; skipping UniMol DataHub call. "
              f"Run again without that flag on HPC to produce the 3D inputs + fold split.")
        return 0

    hub = run_datahub(spec, csv_path, repo_root, args.kfold, args.split_seed, args.method)

    # Reload the rewritten CSV (DataHub may reorder / drop invalid rows internally;
    # we want the splits aligned with the CSV the hub actually ingested).
    df = pd.read_csv(csv_path)
    dump_splits(hub, spec, df, repo_root)
    print(f"\n[ok] {spec.name} export complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
