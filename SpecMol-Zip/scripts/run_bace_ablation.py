import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_gnn_pre import LH_Direct, LogReg  # noqa: E402
from train_utils import calculate_auc, ours_loss, set_seed  # noqa: E402
from utils_fp_downstream import TestbedDataset  # noqa: E402


DEFAULT_BENCH_ROOT = REPO_ROOT / "molecular_benchmarks"
VARIANT_TO_SUBDIR = {
    "V0": "down_task_2d",
    "V0-pairlearn": "down_task_2d",
    "V0-pairlearn-v2": "down_task_2d",
    "V0-ctrl": "down_task_ctrl",
    "V1-bondpair": "down_task_bondpair",
    "V1": "down_task_unimol",
    "V2-stable": "down_task_unimolupd",
}
VARIANT_CONFIGS = {
    "V0": {"use_pair_update": 0, "pair_update_alpha": None, "pair_update_heads": 1, "pair_update_mode": "legacy"},
    "V0-pairlearn": {"use_pair_update": 1, "pair_update_alpha": 0.1, "pair_update_heads": 3, "pair_update_mode": "legacy"},
    "V0-pairlearn-v2": {"use_pair_update": 1, "pair_update_alpha": 0.1, "pair_update_heads": 4, "pair_update_mode": "v2"},
    "V0-ctrl": {"use_pair_update": 0, "pair_update_alpha": None, "pair_update_heads": 1, "pair_update_mode": "legacy"},
    "V1-bondpair": {"use_pair_update": 1, "pair_update_alpha": 0.0, "pair_update_heads": 3, "pair_update_mode": "legacy"},
    "V1": {"use_pair_update": 0, "pair_update_alpha": None, "pair_update_heads": 1, "pair_update_mode": "legacy"},
    "V2-stable": {"use_pair_update": 1, "pair_update_alpha": 0.1, "pair_update_heads": 1, "pair_update_mode": "legacy"},
}
FIXED_SEEDS = [9, 19, 29]
VARIANT_ORDER = ["V0", "V0-pairlearn", "V0-pairlearn-v2", "V0-ctrl", "V1-bondpair", "V1", "V2-stable"]
TASK_NUMS = {
    "bace": 1,
    "bbbp": 1,
    "hiv": 1,
    "tox21": 12,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bace")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logreg_epochs", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--dprate", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--is_bns", type=bool, default=False)
    parser.add_argument("--act_fn", default="relu")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--logreg_lr", type=float, default=1e-3)
    parser.add_argument("--type", default="tri")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--pair_update_layers", type=int, default=2)
    parser.add_argument("--pair_update_alpha", type=float, default=0.1)
    parser.add_argument("--pair_update_clamp_min", type=float, default=-5.0)
    parser.add_argument("--pair_update_clamp_max", type=float, default=5.0)
    parser.add_argument("--pair_edge_attr_dim", type=int, default=11)
    parser.add_argument("--pair_input_attr_name", type=str, default="edge_attr")
    parser.add_argument("--pair_update_heads_override", type=int, default=None)
    parser.add_argument("--pair_update_alpha_override", type=float, default=None)
    parser.add_argument("--pair_update_mode", type=str, default=None)
    parser.add_argument("--use_pair_update", type=int, default=None)
    parser.add_argument("--debug_pair_path", action="store_true")
    parser.add_argument("--results_dir", type=str, default=str(REPO_ROOT / "results"))
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    return parser.parse_args()


def verify_dynamic_pair_path(batch, spec_model, expected_attr_name, stage, variant_name, seed):
    if not hasattr(batch, expected_attr_name):
        raise AssertionError(
            f"[pair-debug] variant={variant_name} seed={seed} stage={stage} missing batch.{expected_attr_name}"
        )
    pair_attr = getattr(batch, expected_attr_name)
    if pair_attr is None or pair_attr.numel() == 0:
        raise AssertionError(
            f"[pair-debug] variant={variant_name} seed={seed} stage={stage} batch.{expected_attr_name} is empty"
        )
    if not spec_model.use_pair_update:
        raise AssertionError(
            f"[pair-debug] variant={variant_name} seed={seed} stage={stage} pair update is disabled"
        )

    stats = spec_model.latest_pair_update_stats
    if stats is None:
        raise AssertionError(
            f"[pair-debug] variant={variant_name} seed={seed} stage={stage} no pair-update stats were recorded"
        )

    expected_source = f"{expected_attr_name}_proj"
    if stats["pair_source"] != expected_source:
        raise AssertionError(
            f"[pair-debug] variant={variant_name} seed={seed} stage={stage} pair_source="
            f"{stats['pair_source']} expected {expected_source}"
        )
    if stats["pair_input_attr_name"] != expected_attr_name:
        raise AssertionError(
            f"[pair-debug] variant={variant_name} seed={seed} stage={stage} pair_input_attr_name="
            f"{stats['pair_input_attr_name']} expected {expected_attr_name}"
        )
    if stats["original_edge_weight_present"]:
        raise AssertionError(
            f"[pair-debug] variant={variant_name} seed={seed} stage={stage} unexpectedly found static edge_weight_3d"
        )

    expected_edge_weight_shape = (int(batch.edge_index.size(1)),)
    if tuple(stats["updated_edge_weight_shape"]) != expected_edge_weight_shape:
        raise AssertionError(
            f"[pair-debug] variant={variant_name} seed={seed} stage={stage} updated_edge_weight_shape="
            f"{stats['updated_edge_weight_shape']} expected {expected_edge_weight_shape}"
        )

    print(
        f"[pair-debug] variant={variant_name} seed={seed} stage={stage} "
        f"batch_pair_attr_shape={tuple(pair_attr.shape)}"
    )
    print(
        f"[pair-debug] variant={variant_name} seed={seed} stage={stage} "
        f"use_pair_update={int(spec_model.use_pair_update)} pair_source={stats['pair_source']}"
    )
    print(
        f"[pair-debug] variant={variant_name} seed={seed} stage={stage} "
        f"updated_edge_weight_shape={stats['updated_edge_weight_shape']} edge_count={expected_edge_weight_shape[0]}"
    )
    print(
        f"[pair-debug] variant={variant_name} seed={seed} stage={stage} "
        f"delta_pair_mean={stats['delta_pair_mean']:.6f} delta_pair_var={stats['delta_pair_var']:.6f}"
    )


def train_epoch(spec_model, data_loader, train_optimizer, device, alpha, debug_pair_path=False, variant_name="NA", seed=-1, expected_attr_name="edge_attr"):
    spec_model.train()
    total_loss = 0.0
    total_num = 0
    debug_checked = False
    for batch in data_loader:
        low_x, high_x, spec_x, x_fp = spec_model(batch, device)
        if debug_pair_path and not debug_checked:
            verify_dynamic_pair_path(
                batch=batch,
                spec_model=spec_model,
                expected_attr_name=expected_attr_name,
                stage="pretrain",
                variant_name=variant_name,
                seed=seed,
            )
            debug_checked = True
        loss = ours_loss(low_x, high_x, spec_x, x_fp, alpha)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        total_num += len(batch)
        total_loss += float(loss.item()) * len(batch)
    return total_loss / max(total_num, 1)


def refine_logreg(batch, device, logreg, n_task, spec_model):
    low_x_mean, high_x_mean, spec_x_mean, x_fp, y = spec_model.get_embedding(batch, device)
    embed = torch.concat([spec_x_mean, x_fp], dim=1)
    logits = logreg(embed)
    y = y.reshape(-1, n_task)

    non_999_indices = y != 999
    logits = logits[non_999_indices]
    y = y[non_999_indices]
    return logits, y


def load_variant_datasets(root, task, data_type):
    dataset_all = TestbedDataset(root=str(root), dataset="all", task=task, type=data_type)
    dataset_train = TestbedDataset(root=str(root), dataset="train", task=task, type=data_type)
    dataset_valid = TestbedDataset(root=str(root), dataset="valid", task=task, type=data_type)
    dataset_test = TestbedDataset(root=str(root), dataset="test", task=task, type=data_type)
    return dataset_all, dataset_train, dataset_valid, dataset_test


def run_single_variant(args, variant_name, variant_root, seed, device):
    set_seed(seed)
    task_num = TASK_NUMS.get(args.task.lower())
    if task_num is None:
        raise ValueError(f"Unsupported task={args.task}. Add it to TASK_NUMS first.")
    variant_cfg = VARIANT_CONFIGS[variant_name]
    use_pair_update = variant_cfg["use_pair_update"] if args.use_pair_update is None else int(args.use_pair_update)
    pair_update_alpha = args.pair_update_alpha if variant_cfg["pair_update_alpha"] is None else variant_cfg["pair_update_alpha"]
    if args.pair_update_alpha_override is not None:
        pair_update_alpha = args.pair_update_alpha_override
    pair_update_heads = variant_cfg["pair_update_heads"] if args.pair_update_heads_override is None else int(args.pair_update_heads_override)
    pair_update_mode = variant_cfg["pair_update_mode"] if args.pair_update_mode is None else args.pair_update_mode

    dataset_all, dataset_train, dataset_valid, dataset_test = load_variant_datasets(
        root=variant_root,
        task=args.task,
        data_type=args.type,
    )
    pretrain_loader = DataLoader(dataset_all, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    spec_model = LH_Direct(
        in_dim=93,
        hid_dim=args.hid_dim,
        K=args.K,
        dprate=args.dprate,
        dropout=args.dropout,
        is_bns=args.is_bns,
        act_fn=args.act_fn,
        type=args.type,
        use_pair_update=bool(use_pair_update),
        pair_update_heads=pair_update_heads,
        pair_update_layers=args.pair_update_layers,
        pair_update_alpha=pair_update_alpha,
        pair_update_clamp_min=args.pair_update_clamp_min,
        pair_update_clamp_max=args.pair_update_clamp_max,
        pair_edge_attr_dim=args.pair_edge_attr_dim,
        pair_input_attr_name=args.pair_input_attr_name,
        pair_update_mode=pair_update_mode,
    ).to(device)
    optimizer = optim.Adam(spec_model.parameters(), lr=args.lr, weight_decay=1e-7)

    best_pretrain_loss = float("inf")
    debug_embedding_checked = False
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            spec_model,
            pretrain_loader,
            optimizer,
            device,
            args.alpha,
            debug_pair_path=args.debug_pair_path,
            variant_name=variant_name,
            seed=seed,
            expected_attr_name=args.pair_input_attr_name,
        )
        best_pretrain_loss = min(best_pretrain_loss, train_loss)
        print(
            f"[ablation] variant={variant_name} seed={seed} pretrain_epoch={epoch} "
            f"loss={train_loss:.6f}"
        )

    spec_model.eval()
    logreg = LogReg(hid_dim=args.hid_dim, n_classes=task_num).to(device)
    opt = optim.Adam(logreg.parameters(), lr=args.logreg_lr, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    best_valid_auc = -1.0
    best_test_auc = -1.0
    best_epoch = -1

    for epoch in range(args.logreg_epochs):
        logreg.train()
        for batch in train_loader:
            opt.zero_grad()
            logits, y = refine_logreg(batch, device, logreg, task_num, spec_model)
            if args.debug_pair_path and not debug_embedding_checked:
                verify_dynamic_pair_path(
                    batch=batch,
                    spec_model=spec_model,
                    expected_attr_name=args.pair_input_attr_name,
                    stage="logreg_train",
                    variant_name=variant_name,
                    seed=seed,
                )
                debug_embedding_checked = True
            if logits.numel() == 0:
                continue
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            logreg.eval()
            valid_logits = torch.empty((0,), device=device)
            valid_y = torch.empty((0,), device=device)
            for batch in valid_loader:
                logits, y = refine_logreg(batch, device, logreg, task_num, spec_model)
                valid_logits = torch.cat((valid_logits, logits.reshape(-1)), dim=0)
                valid_y = torch.cat((valid_y, y.reshape(-1)), dim=0)
            valid_auc = calculate_auc(valid_y.cpu().numpy(), valid_logits.cpu().numpy(), task_num)

            if valid_auc > best_valid_auc:
                test_logits = torch.empty((0,), device=device)
                test_y = torch.empty((0,), device=device)
                for batch in test_loader:
                    logits, y = refine_logreg(batch, device, logreg, task_num, spec_model)
                    test_logits = torch.cat((test_logits, logits.reshape(-1)), dim=0)
                    test_y = torch.cat((test_y, y.reshape(-1)), dim=0)
                test_auc = calculate_auc(test_y.cpu().numpy(), test_logits.cpu().numpy(), task_num)
                best_valid_auc = valid_auc
                best_test_auc = test_auc
                best_epoch = epoch
                print(
                    f"[ablation] variant={variant_name} seed={seed} "
                    f"logreg_epoch={epoch} valid_auc={valid_auc:.6f} test_auc={test_auc:.6f}"
                )

    return {
        "variant": variant_name,
        "seed": seed,
        "data_root": str(variant_root),
        "use_pair_update": int(use_pair_update),
        "pair_update_layers": int(args.pair_update_layers),
        "pair_update_heads": int(pair_update_heads),
        "pair_update_alpha": float(pair_update_alpha),
        "pair_update_mode": str(pair_update_mode),
        "pair_update_clamp_min": float(args.pair_update_clamp_min),
        "pair_update_clamp_max": float(args.pair_update_clamp_max),
        "pair_edge_attr_dim": int(args.pair_edge_attr_dim),
        "pair_input_attr_name": str(args.pair_input_attr_name),
        "pretrain_epochs": int(args.epochs),
        "logreg_epochs": int(args.logreg_epochs),
        "best_pretrain_loss": float(best_pretrain_loss),
        "best_valid_auc": float(best_valid_auc),
        "best_test_auc": float(best_test_auc),
        "best_epoch": int(best_epoch),
    }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows):
    summary_rows = []
    for variant_name in VARIANT_ORDER:
        variant_rows = [row for row in rows if row["variant"] == variant_name]
        if not variant_rows:
            continue
        seed_values = [int(row["seed"]) for row in variant_rows]
        valid_values = np.array([row["best_valid_auc"] for row in variant_rows], dtype=np.float64)
        test_values = np.array([row["best_test_auc"] for row in variant_rows], dtype=np.float64)
        summary_rows.append(
            {
                "variant": variant_name,
                "seeds": ",".join(str(seed) for seed in seed_values),
                "runs": len(variant_rows),
                "valid_auc_mean": float(valid_values.mean()),
                "valid_auc_std": float(valid_values.std(ddof=0)),
                "test_auc_mean": float(test_values.mean()),
                "test_auc_std": float(test_values.std(ddof=0)),
            }
        )
    return summary_rows


def resolve_variant_roots(data_root_arg, task):
    if data_root_arg:
        base_root = Path(data_root_arg).expanduser().resolve()
        if (base_root / "processed").exists():
            return {variant_name: base_root for variant_name in VARIANT_ORDER}
        return {
            variant_name: base_root / VARIANT_TO_SUBDIR[variant_name]
            for variant_name in VARIANT_ORDER
        }
    task_root = DEFAULT_BENCH_ROOT / task.lower()
    return {
        variant_name: task_root / VARIANT_TO_SUBDIR[variant_name]
        for variant_name in VARIANT_ORDER
    }


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if args.gpu != -1 and torch.cuda.is_available() else "cpu"
    variant_roots = resolve_variant_roots(args.data_root, args.task)
    selected_variants = VARIANT_ORDER if not args.variants else [item.strip() for item in args.variants.split(",") if item.strip()]
    selected_seeds = FIXED_SEEDS if not args.seeds else [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    print(f"[ablation] task={args.task}")
    print(f"[ablation] device={device}")
    print(f"[ablation] seeds={selected_seeds}")
    print(f"[ablation] pair_update_layers={args.pair_update_layers}")
    print(
        "[ablation] note: V0-pairlearn keeps the original sparse bond graph, initializes pair bias from "
        "bond edge_attr, and then learns dynamic edge weights during training."
    )
    print(
        "[ablation] note: V0-ctrl uses all-pairs edge_index with edge_weight_3d=1 to isolate the effect "
        "of dense all-pairs connectivity from actual Uni-Mol information."
    )
    print(
        "[ablation] note: V2-stable uses residual-alpha pair updates plus clamp to test whether the "
        "original update mechanism was too strong and hurt generalization."
    )
    print(
        "[ablation] note: V1-bondpair keeps the original bond graph and feeds full Uni-Mol pair heads "
        "into attention, avoiding the previous mean-to-scalar edge-weight compression."
    )
    for variant_name in selected_variants:
        variant_root = variant_roots[variant_name]
        print(
            f"[ablation] variant={variant_name} root={variant_root} "
            f"use_pair_update={VARIANT_CONFIGS[variant_name]['use_pair_update'] if args.use_pair_update is None else int(args.use_pair_update)} "
            f"pair_update_mode={VARIANT_CONFIGS[variant_name]['pair_update_mode'] if args.pair_update_mode is None else args.pair_update_mode}"
        )

    run_rows = []
    for variant_name in selected_variants:
        for seed in selected_seeds:
            run_rows.append(run_single_variant(args, variant_name, variant_roots[variant_name], seed, device))

    results_dir = Path(args.results_dir).expanduser().resolve()
    runs_path = results_dir / f"{args.task.lower()}_ablation_runs.csv"
    summary_path = results_dir / f"{args.task.lower()}_ablation_summary.csv"

    write_csv(
        runs_path,
        [
            "variant",
            "seed",
            "data_root",
            "use_pair_update",
            "pair_update_layers",
            "pair_update_heads",
            "pair_update_alpha",
            "pair_update_mode",
            "pair_update_clamp_min",
            "pair_update_clamp_max",
            "pair_edge_attr_dim",
            "pair_input_attr_name",
            "pretrain_epochs",
            "logreg_epochs",
            "best_pretrain_loss",
            "best_valid_auc",
            "best_test_auc",
            "best_epoch",
        ],
        run_rows,
    )

    summary_rows = summarize_rows(run_rows)
    write_csv(
        summary_path,
        [
            "variant",
            "seeds",
            "runs",
            "valid_auc_mean",
            "valid_auc_std",
            "test_auc_mean",
            "test_auc_std",
        ],
        summary_rows,
    )

    print(f"[ablation] wrote_runs_csv={runs_path}")
    print(f"[ablation] wrote_summary_csv={summary_path}")
    for row in summary_rows:
        print(
            f"[ablation-summary] variant={row['variant']} runs={row['runs']} "
            f"valid_auc={row['valid_auc_mean']:.6f}±{row['valid_auc_std']:.6f} "
            f"test_auc={row['test_auc_mean']:.6f}±{row['test_auc_std']:.6f}"
        )


if __name__ == "__main__":
    main()
