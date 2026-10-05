import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "down_task"


def build_all_pairs_edge_index(num_nodes):
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    src = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, dtype=torch.long).repeat(num_nodes)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def parse_float(value, field_name, row_index):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} at row {row_index}: {value}") from exc


def load_labels_csv(labels_csv):
    labels_by_id = {}
    with Path(labels_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"spectrum_id", "label"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"labels csv must contain columns {sorted(required)}; got {reader.fieldnames}"
            )

        for row_index, row in enumerate(reader, start=2):
            spectrum_id = str(row["spectrum_id"]).strip()
            if not spectrum_id:
                raise ValueError(f"Missing spectrum_id in labels csv row {row_index}")
            label = parse_float(row["label"], "label", row_index)
            weight = parse_float(row.get("weight", 1.0), "weight", row_index) if row.get("weight") not in {None, ""} else 1.0
            split = str(row.get("split", "")).strip().lower() or None
            if split is not None and split not in {"train", "valid", "test"}:
                raise ValueError(f"Invalid split={split} in labels csv row {row_index}")
            labels_by_id[spectrum_id] = {
                "label": label,
                "weight": weight,
                "split": split,
            }
    if not labels_by_id:
        raise ValueError("labels csv is empty")
    return labels_by_id


def load_peaks_csv(peaks_csv):
    peaks_by_id = defaultdict(list)
    with Path(peaks_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"spectrum_id", "mz", "intensity"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"peaks csv must contain columns {sorted(required)}; got {reader.fieldnames}"
            )

        for row_index, row in enumerate(reader, start=2):
            spectrum_id = str(row["spectrum_id"]).strip()
            if not spectrum_id:
                raise ValueError(f"Missing spectrum_id in peaks csv row {row_index}")
            mz = parse_float(row["mz"], "mz", row_index)
            intensity = parse_float(row["intensity"], "intensity", row_index)
            peaks_by_id[spectrum_id].append({"mz": mz, "intensity": intensity})
    if not peaks_by_id:
        raise ValueError("peaks csv is empty")
    return peaks_by_id


def select_peaks(peaks, max_peaks=None):
    if max_peaks is not None and max_peaks > 0 and len(peaks) > max_peaks:
        peaks = sorted(peaks, key=lambda item: item["intensity"], reverse=True)[:max_peaks]
    return sorted(peaks, key=lambda item: item["mz"])


def build_samples_from_csv(peaks_csv, labels_csv, max_peaks=None):
    labels_by_id = load_labels_csv(labels_csv)
    peaks_by_id = load_peaks_csv(peaks_csv)

    sample_ids = sorted(labels_by_id.keys())
    samples = []
    for spectrum_id in sample_ids:
        if spectrum_id not in peaks_by_id:
            raise ValueError(f"spectrum_id={spectrum_id} exists in labels csv but not in peaks csv")
        peaks = select_peaks(peaks_by_id[spectrum_id], max_peaks=max_peaks)
        if not peaks:
            raise ValueError(f"spectrum_id={spectrum_id} has no usable peaks")
        meta = labels_by_id[spectrum_id]
        samples.append(
            {
                "id": spectrum_id,
                "label": meta["label"],
                "weight": meta["weight"],
                "split": meta["split"],
                "peaks": peaks,
            }
        )
    return samples


def build_peak_features(mz, intensity, node_feat_dim):
    if mz.numel() == 0:
        raise ValueError("Spectrum must contain at least one peak")
    if node_feat_dim < 3:
        raise ValueError(f"node_feat_dim={node_feat_dim} must be at least 3")

    max_intensity = torch.clamp(intensity.max(), min=1e-8)
    norm_intensity = intensity / max_intensity
    base_x = torch.stack([mz, intensity, norm_intensity], dim=-1)

    if node_feat_dim == 3:
        return base_x
    pad = base_x.new_zeros(base_x.size(0), node_feat_dim - 3)
    return torch.cat([base_x, pad], dim=-1)


def build_pair_attr(mz, intensity, edge_index):
    if edge_index.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    src = edge_index[0]
    dst = edge_index[1]
    delta_mz = mz[dst] - mz[src]
    intensity_ratio = intensity[dst] / torch.clamp(intensity[src], min=1e-8)
    return torch.stack([delta_mz, intensity_ratio], dim=-1).to(torch.float32)


def build_spectrum_fingerprint(mz, intensity, fps_dim):
    fps = torch.zeros(fps_dim, dtype=torch.float32)
    if fps_dim <= 0 or mz.numel() == 0:
        return fps
    norm_intensity = intensity / torch.clamp(intensity.max(), min=1e-8)
    bin_index = torch.remainder(torch.round(mz).long().abs(), fps_dim)
    fps.index_add_(0, bin_index, norm_intensity)
    max_value = torch.clamp(fps.max(), min=1e-8)
    return fps / max_value


def spectrum_to_data(sample, node_feat_dim, fps_dim):
    peaks = sample["peaks"]
    mz = torch.tensor([float(peak["mz"]) for peak in peaks], dtype=torch.float32)
    intensity = torch.tensor([float(peak["intensity"]) for peak in peaks], dtype=torch.float32)
    x = build_peak_features(mz, intensity, node_feat_dim=node_feat_dim)
    edge_index = build_all_pairs_edge_index(x.size(0))
    pair_attr = build_pair_attr(mz, intensity, edge_index)
    edge_attr = torch.empty((edge_index.size(1), 0), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        pair_attr=pair_attr,
        edge_attr=edge_attr,
        y=torch.tensor([float(sample["label"])], dtype=torch.float32),
        w=torch.tensor([float(sample.get("weight", 1.0))], dtype=torch.float32),
        fps=build_spectrum_fingerprint(mz, intensity, fps_dim=fps_dim),
    )
    data.sample_id = str(sample["id"])
    return data


def split_samples(samples):
    has_explicit_split = all(sample.get("split") in {"train", "valid", "test"} for sample in samples)
    if has_explicit_split:
        train_samples = [sample for sample in samples if sample["split"] == "train"]
        valid_samples = [sample for sample in samples if sample["split"] == "valid"]
        test_samples = [sample for sample in samples if sample["split"] == "test"]
        if not train_samples or not valid_samples or not test_samples:
            raise ValueError("Explicit split column must include at least one sample in train/valid/test")
        return {
            "train": train_samples,
            "valid": valid_samples,
            "test": test_samples,
            "all": samples,
        }

    total = len(samples)
    if total < 3:
        raise ValueError("Need at least 3 spectra to create deterministic train/valid/test splits")

    train_end = max(1, int(total * 0.6))
    valid_end = max(train_end + 1, int(total * 0.8))
    valid_end = min(valid_end, total - 1)
    return {
        "train": samples[:train_end],
        "valid": samples[train_end:valid_end],
        "test": samples[valid_end:],
        "all": samples,
    }


def save_split(data_list, output_path):
    data_obj, slices = InMemoryDataset.collate(data_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((data_obj, slices), output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peaks-csv", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--task", type=str, default="bace")
    parser.add_argument("--node-feat-dim", type=int, default=93)
    parser.add_argument("--fps-dim", type=int, default=1489)
    parser.add_argument("--max-peaks", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = build_samples_from_csv(
        peaks_csv=args.peaks_csv,
        labels_csv=args.labels_csv,
        max_peaks=args.max_peaks,
    )
    split_to_samples = split_samples(samples)
    output_root = Path(args.output_root).expanduser().resolve()
    processed_dir = output_root / "processed"

    print(f"[spectrum] loaded spectra={len(samples)}")
    print(f"[spectrum] peaks_csv={Path(args.peaks_csv).expanduser().resolve()}")
    print(f"[spectrum] labels_csv={Path(args.labels_csv).expanduser().resolve()}")
    print(f"[spectrum] output_root={output_root}")
    print(f"[spectrum] node_feat_dim={args.node_feat_dim} fps_dim={args.fps_dim} max_peaks={args.max_peaks}")

    for split_name, split_samples_list in split_to_samples.items():
        data_list = [
            spectrum_to_data(
                sample=sample,
                node_feat_dim=args.node_feat_dim,
                fps_dim=args.fps_dim,
            )
            for sample in split_samples_list
        ]
        output_path = processed_dir / f"{args.task}_{split_name}.pt"
        save_split(data_list, output_path)
        print(
            f"[spectrum] saved split={split_name} graphs={len(data_list)} "
            f"path={output_path}"
        )


if __name__ == "__main__":
    main()
