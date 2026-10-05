#!/usr/bin/env python
"""Prepare all V0 baseline datasets for the HPC run."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_TASKS = ("bace", "bbbp", "hiv", "clintox", "tox21")
DEFAULT_SEEDS = (9, 19, 29)


def parse_csv(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_int_csv(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_command(args):
    print("[prepare]", " ".join(str(part) for part in args), flush=True)
    subprocess.run(args, cwd=REPO, check=True)


def prepare_molnet_csv(task, force_csv):
    cmd = [sys.executable, "prepare_molnet_dataset.py", "--task", task]
    if force_csv:
        cmd.append("--force")
    run_command(cmd)


def build_classification_processed(task, data_root, seeds):
    from utils_fp_downstream import TestbedDataset

    (data_root / "processed").mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        for split in ("train", "valid", "test"):
            TestbedDataset(root=str(data_root), dataset=split, task=task, type="tri", seed=seed)

    all_seed = seeds[0]
    TestbedDataset(root=str(data_root), dataset="all", task=task, type="tri", seed=all_seed)


def verify_classification_outputs(task, data_root, seeds):
    processed = data_root / "processed"
    expected = [processed / f"{task}_tri_{seeds[0]}_all.pt"]
    for seed in seeds:
        expected.extend(
            processed / f"{task}_tri_{seed}_{split}.pt"
            for split in ("train", "valid", "test")
        )

    missing = [path for path in expected if not path.exists()]
    if missing:
        lines = "\n".join(str(path) for path in missing)
        raise RuntimeError(f"Missing processed files for task={task}:\n{lines}")

    print(f"[prepare] verified task={task} root={data_root}", flush=True)


def prepare_freesolv(skip_freesolv):
    if skip_freesolv:
        print("[prepare] skipping FreeSolv by request", flush=True)
        return
    run_command([sys.executable, "prepare_freesolv_data.py"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--data-root", default="down_task_v0")
    parser.add_argument("--force-csv", action="store_true")
    parser.add_argument("--skip-freesolv", action="store_true")
    args = parser.parse_args()

    tasks = parse_csv(args.tasks)
    seeds = parse_int_csv(args.seeds)
    if not tasks:
        raise ValueError("At least one classification task is required")
    if not seeds:
        raise ValueError("At least one seed is required")

    data_root = (REPO / args.data_root).resolve()
    print(f"[prepare] repo={REPO}", flush=True)
    print(f"[prepare] classification_data_root={data_root}", flush=True)
    print(f"[prepare] tasks={tasks}", flush=True)
    print(f"[prepare] seeds={seeds}", flush=True)

    for task in tasks:
        prepare_molnet_csv(task, force_csv=args.force_csv)
        build_classification_processed(task, data_root=data_root, seeds=seeds)
        verify_classification_outputs(task, data_root=data_root, seeds=seeds)

    prepare_freesolv(skip_freesolv=args.skip_freesolv)
    print("[prepare] all requested V0 datasets are ready", flush=True)


if __name__ == "__main__":
    main()
