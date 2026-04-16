import argparse
import csv
from pathlib import Path
from urllib.error import URLError

import deepchem as dc


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "dataset"
DEFAULT_CACHE_ROOT = REPO_ROOT / "molnet_cache"


TASK_TO_LOADER = {
    "bace": dc.molnet.load_bace_classification,
    "bbbp": dc.molnet.load_bbbp,
    "hiv": dc.molnet.load_hiv,
    "clintox": dc.molnet.load_clintox,
    "tox21": dc.molnet.load_tox21,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_LOADER))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-root", type=str, default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_molnet_dataset(task, cache_root):
    loader = TASK_TO_LOADER[task]
    cache_dir = cache_root / task
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        tasks, datasets, _ = loader(
            featurizer="Raw",
            splitter=None,
            data_dir=str(cache_dir),
            save_dir=str(cache_dir),
            reload=True,
        )
    except URLError as exc:
        raise RuntimeError(
            f"Failed to download MolNet dataset for task={task}. "
            f"Check network access. Original error: {exc}"
        ) from exc

    dataset_obj = datasets[0] if isinstance(datasets, tuple) else datasets
    return tasks, dataset_obj


def write_raw_csv(task, output_root, tasks, dataset_obj):
    raw_dir = output_root / task / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "smiles.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for smile, label_row in zip(dataset_obj.ids, dataset_obj.y):
            writer.writerow([str(smile), *[float(value) for value in label_row.tolist()]])

    print(f"[molnet-prepare] task={task}")
    print(f"[molnet-prepare] tasks={list(tasks)}")
    print(f"[molnet-prepare] rows={len(dataset_obj)}")
    print(f"[molnet-prepare] label_dim={dataset_obj.y.shape[1]}")
    print(f"[molnet-prepare] output_path={output_path}")
    return output_path


def main():
    args = parse_args()
    task = args.task.lower()
    output_root = Path(args.output_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    output_path = output_root / task / "raw" / "smiles.csv"

    if output_path.exists() and not args.force:
        print(f"[molnet-prepare] task={task} already exists at {output_path}, skipping")
        return

    tasks, dataset_obj = load_molnet_dataset(task=task, cache_root=cache_root)
    write_raw_csv(task=task, output_root=output_root, tasks=tasks, dataset_obj=dataset_obj)


if __name__ == "__main__":
    main()
