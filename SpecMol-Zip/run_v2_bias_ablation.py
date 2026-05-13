"""Bias-init ablation: test whether lowering PairToEdgeWeight bias_init
allows the model to actually learn differentiated edge weights from 3D signal.

Key question: with bias=5 the edge weights stay at ~0.993 (saturated sigmoid).
Does bias=0 let them differentiate, and does that help or hurt?

Usage:
    python run_v2_bias_ablation.py --bias_init 0.0 --seed 9   # quick diagnostic
    python run_v2_bias_ablation.py --bias_init 0.0             # all 3 seeds
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader

from model_gnn_pre import LogReg
from model_gnn_pre_v2 import LH_Direct_V2
from train_utils import calculate_auc, load_model_state_dict, ours_loss, set_seed
from utils_fp_downstream import TestbedDataset


def log_edge_weight_stats(model, data_loader, device):
    """Compute edge weight distribution across one pass of data."""
    model.eval()
    all_weights = []
    with torch.no_grad():
        for batch in data_loader:
            w = model._compute_edge_weight(batch, device)
            all_weights.append(w.cpu())
    w_cat = torch.cat(all_weights)
    stats = {
        "mean": round(w_cat.mean().item(), 6),
        "std": round(w_cat.std().item(), 6),
        "min": round(w_cat.min().item(), 6),
        "max": round(w_cat.max().item(), 6),
    }
    model.train()
    return stats


def run_one_seed(pretrain_seed, eval_splits, device, bias_init, path="down_task_v2"):
    set_seed(pretrain_seed)

    batch_size = 512
    epochs = 1000
    patience = 100
    lr = 0.0001
    hid_dim = 512
    K = 10
    eval_epochs = 2000
    task = "bace"
    fp_type = "tri"
    alpha = 1

    print(f"\n{'='*60}")
    print(f"PRETRAIN seed={pretrain_seed}, bias_init={bias_init}")
    print(f"{'='*60}")

    data = TestbedDataset(root=path, dataset="all", task=task, type=fp_type)
    model = LH_Direct_V2(
        in_dim=93, hid_dim=hid_dim, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type=fp_type, bias_init=bias_init,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    actual_bias = model.pair_to_edge_weight.mlp[2].bias.item()
    print(f"  PairToEdgeWeight bias init: {actual_bias:.4f}")

    # Log initial edge weight distribution
    init_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    ew_stats_init = log_edge_weight_stats(model, init_loader, device)
    print(f"  Edge weights @ epoch 0: {ew_stats_init}")
    ew_history = [{"epoch": 0, **ew_stats_init}]

    best_loss = float("inf")
    cnt_wait = 0
    best_t = 0
    tag = f"v2_bias{bias_init}_{pretrain_seed}_{int(time.time())}"
    ckpt_path = f"pkl/best_spec_model_bace_{tag}.pkl"

    for epoch in range(epochs + 1):
        model.train()
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        total_loss, total_num = 0.0, 0
        for batch in loader:
            low_x, high_x, spec_x, x_fp = model(batch, device)
            loss = ours_loss(low_x, high_x, spec_x, x_fp, alpha=alpha)
            total_num += len(batch)
            total_loss += loss.item() * len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / total_num

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            cnt_wait += 1

        # Log edge weight stats at key epochs
        if epoch in [1, 5, 10, 50, 100, 200, 500] or epoch == best_t:
            stats_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
            ew_stats = log_edge_weight_stats(model, stats_loader, device)
            ew_history.append({"epoch": epoch, **ew_stats})
            bias_now = model.pair_to_edge_weight.mlp[2].bias.item()
            if epoch % 50 == 0 or epoch <= 10:
                print(f"  Epoch {epoch}: loss={avg_loss:.6f}, bias={bias_now:.4f}, "
                      f"ew_mean={ew_stats['mean']:.4f}, ew_std={ew_stats['std']:.4f}")

        if cnt_wait == patience:
            print(f"  Early stopping at epoch {epoch}")
            # Log final stats
            stats_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
            ew_stats = log_edge_weight_stats(model, stats_loader, device)
            ew_history.append({"epoch": epoch, "final": True, **ew_stats})
            break

    bias_final = model.pair_to_edge_weight.mlp[2].bias.item()
    print(f"  Best epoch: {best_t}, best loss: {best_loss:.6f}")
    print(f"  PairToEdgeWeight bias final: {bias_final:.4f}")

    del optimizer, data, loader
    import gc; gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(load_model_state_dict(ckpt_path))
    model.eval()

    result = {
        "pretrain_seed": pretrain_seed,
        "bias_init": bias_init,
        "pretrain_best_epoch": best_t,
        "pretrain_final_loss": round(best_loss, 6),
        "bias_final": round(bias_final, 4),
        "edge_weight_history": ew_history,
        "eval_splits": {},
    }

    for split_seed in eval_splits:
        print(f"  EVAL split_seed={split_seed}")
        logreg = LogReg(hid_dim=hid_dim, n_classes=1).to(device)
        opt = torch.optim.Adam(logreg.parameters(), lr=0.001, weight_decay=0.0)
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        train_data = TestbedDataset(root=path, dataset="train", task=task, type=fp_type, seed=split_seed)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_data = TestbedDataset(root=path, dataset="valid", task=task, type=fp_type, seed=split_seed)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_data = TestbedDataset(root=path, dataset="test", task=task, type=fp_type, seed=split_seed)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        val_auc_best = 0
        best_test_auc = 0
        best_eval_epoch = 0

        for ep in range(eval_epochs):
            logreg.train()
            for batch in train_loader:
                opt.zero_grad()
                with torch.no_grad():
                    low_x, high_x, spec_x, x_fp, y = model.get_embedding(batch, device)
                embed = torch.cat([spec_x.detach(), x_fp.detach()], dim=1)
                logits = logreg(embed)
                y = y.reshape(-1, 1)
                mask = y != 999
                loss = loss_fn(logits[mask], y[mask])
                loss.backward()
                opt.step()

            with torch.no_grad():
                val_logits = torch.Tensor().to(device)
                val_y = torch.Tensor().to(device)
                for batch in val_loader:
                    low_x, high_x, spec_x, x_fp, y = model.get_embedding(batch, device)
                    embed = torch.cat([spec_x, x_fp], dim=1)
                    logits = logreg(embed)
                    y = y.reshape(-1, 1)
                    mask = y != 999
                    val_logits = torch.cat((val_logits, logits[mask]), 0)
                    val_y = torch.cat((val_y, y[mask]), 0)
                val_auc = calculate_auc(val_y.cpu().numpy(), val_logits.cpu().numpy(), 1)

                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    test_logits = torch.Tensor().to(device)
                    test_y = torch.Tensor().to(device)
                    for batch in test_loader:
                        low_x, high_x, spec_x, x_fp, y = model.get_embedding(batch, device)
                        embed = torch.cat([spec_x, x_fp], dim=1)
                        logits = logreg(embed)
                        y = y.reshape(-1, 1)
                        mask = y != 999
                        test_logits = torch.cat((test_logits, logits[mask]), 0)
                        test_y = torch.cat((test_y, y[mask]), 0)
                    best_test_auc = calculate_auc(test_y.cpu().numpy(), test_logits.cpu().numpy(), 1)
                    best_eval_epoch = ep

        print(f"    split_{split_seed}: test_auc={best_test_auc:.4f} (eval epoch {best_eval_epoch})")
        result["eval_splits"][f"split_{split_seed}"] = {
            "test_auc": round(best_test_auc, 4),
            "best_eval_epoch": best_eval_epoch,
        }

    split_aucs = [v["test_auc"] for v in result["eval_splits"].values()]
    result["mean_test_auc"] = round(np.mean(split_aucs), 4)
    print(f"  Mean AUC for seed {pretrain_seed}: {result['mean_test_auc']:.4f}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias_init", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed for quick diagnostic; omit for all 3")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs("pkl", exist_ok=True)

    pretrain_seeds = [args.seed] if args.seed is not None else [9, 19, 29]
    eval_splits = [9, 19, 29]
    bias_tag = str(args.bias_init).replace(".", "p")
    results_path = f"v2_bias{bias_tag}_bace_results.json"

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "experiment": f"V2 Bias-init ablation (bias={args.bias_init})",
            "bias_init": args.bias_init,
            "data_root": "down_task_v2",
            "comparison": "V2-T5 (bias=5): 0.837, FP-only: 0.846, Baseline: 0.797",
            "results_per_seed": {},
        }

    for seed in pretrain_seeds:
        key = f"seed_{seed}"
        if key in all_results.get("results_per_seed", {}):
            print(f"\nSkipping seed {seed} (already done)")
            continue
        result = run_one_seed(seed, eval_splits, device, args.bias_init)
        all_results["results_per_seed"][key] = result
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    all_aucs = []
    for key in sorted(all_results["results_per_seed"].keys()):
        r = all_results["results_per_seed"][key]
        all_aucs.extend([v["test_auc"] for v in r["eval_splits"].values()])

    if all_aucs:
        all_results["summary"] = {
            "all_test_aucs": all_aucs,
            "grand_mean": round(np.mean(all_aucs), 4),
            "grand_std": round(np.std(all_aucs), 4),
        }
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"BIAS={args.bias_init} RESULT: {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
        print(f"Compare: V2-T5 (bias=5): 0.837 | FP-only: 0.846 | Baseline: 0.797")
        print(f"{'='*60}")
