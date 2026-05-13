"""Run FreeSolv baseline (2D-only LH_Direct) with regression downstream.

Usage:
    python run_freesolv_baseline.py --seed 9
    python run_freesolv_baseline.py  # runs all 3 seeds
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

from model_gnn_pre import LH_Direct, LogReg
from train_utils import load_model_state_dict, load_pyg_inmemory_split, ours_loss, set_seed

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)


TASK = "freesolv"
DATA_ROOT = "down_task_freesolv_2d"
RESULTS_PATH = "freesolv_baseline_results.json"


def load_freesolv_split(root, split, seed=9):
    """Load a FreeSolv .pt split directly (bypass TestbedDataset which expects classification)."""
    return load_pyg_inmemory_split(root, "freesolv", split)


def run_one_seed(pretrain_seed, device):
    set_seed(pretrain_seed)

    batch_size = 256
    epochs = 1000
    patience = 100
    lr = 0.0001
    hid_dim = 512
    K = 10
    eval_epochs = 2000
    fp_type = "tri"
    alpha = 1

    print(f"\n{'='*60}")
    print(f"BASELINE PRETRAIN seed={pretrain_seed}")
    print(f"{'='*60}")

    data = load_freesolv_split(DATA_ROOT, "all")
    model = LH_Direct(
        in_dim=93, hid_dim=hid_dim, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type=fp_type,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    best_loss = float("inf")
    cnt_wait = 0
    best_t = 0
    tag = f"freesolv_base_{pretrain_seed}_{int(time.time())}"
    ckpt_path = f"pkl/best_spec_model_{tag}.pkl"

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
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: loss={avg_loss:.6f} (new best)")
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"  Best epoch: {best_t}, best loss: {best_loss:.6f}")

    del optimizer, data, loader
    import gc; gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(load_model_state_dict(ckpt_path))
    model.eval()

    # Regression downstream eval (MSE loss, RMSE metric)
    print(f"  EVAL regression downstream")
    logreg = LogReg(hid_dim=hid_dim, n_classes=1).to(device)
    opt = torch.optim.Adam(logreg.parameters(), lr=0.001, weight_decay=0.0)
    loss_fn = nn.MSELoss(reduction="mean")

    train_data = load_freesolv_split(DATA_ROOT, "train")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = load_freesolv_split(DATA_ROOT, "valid")
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_data = load_freesolv_split(DATA_ROOT, "test")
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    best_val_rmse = float("inf")
    best_test_rmse = float("inf")
    best_eval_epoch = 0

    for ep in range(eval_epochs):
        logreg.train()
        for batch in train_loader:
            opt.zero_grad()
            with torch.no_grad():
                _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
            embed = torch.cat([spec_x.detach(), x_fp.detach()], dim=1)
            logits = logreg(embed)
            y = y.reshape(-1, 1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            val_preds = torch.Tensor().to(device)
            val_y = torch.Tensor().to(device)
            for batch in val_loader:
                _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
                embed = torch.cat([spec_x, x_fp], dim=1)
                preds = logreg(embed)
                y = y.reshape(-1, 1)
                val_preds = torch.cat((val_preds, preds), 0)
                val_y = torch.cat((val_y, y), 0)
            val_rmse = torch.sqrt(loss_fn(val_preds, val_y)).item()

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_eval_epoch = ep
                # Evaluate test
                test_preds = torch.Tensor().to(device)
                test_y = torch.Tensor().to(device)
                for batch in test_loader:
                    _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
                    embed = torch.cat([spec_x, x_fp], dim=1)
                    preds = logreg(embed)
                    y = y.reshape(-1, 1)
                    test_preds = torch.cat((test_preds, preds), 0)
                    test_y = torch.cat((test_y, y), 0)
                best_test_rmse = torch.sqrt(loss_fn(test_preds, test_y)).item()

    print(f"    test_rmse={best_test_rmse:.4f} (eval epoch {best_eval_epoch})")

    result = {
        "pretrain_seed": pretrain_seed,
        "pretrain_best_epoch": best_t,
        "pretrain_final_loss": round(best_loss, 6),
        "best_eval_epoch": best_eval_epoch,
        "best_val_rmse": round(best_val_rmse, 4),
        "best_test_rmse": round(best_test_rmse, 4),
    }

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results-path", type=str, default=RESULTS_PATH)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs("pkl", exist_ok=True)
    results_path = args.results_path
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else [9, 19, 29]

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "experiment": "FreeSolv Baseline (2D-only LH_Direct)",
            "data_root": DATA_ROOT,
            "metric": "RMSE (lower is better)",
            "results_per_seed": {},
        }

    for seed in seeds:
        key = f"seed_{seed}"
        if key in all_results.get("results_per_seed", {}):
            print(f"\nSkipping seed {seed} (already done)")
            continue
        result = run_one_seed(seed, device)
        all_results["results_per_seed"][key] = result
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    rmses = [v["best_test_rmse"] for v in all_results["results_per_seed"].values()]
    if rmses:
        all_results["summary"] = {
            "all_test_rmses": rmses,
            "mean_rmse": round(np.mean(rmses), 4),
            "std_rmse": round(np.std(rmses), 4),
        }
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBASELINE SUMMARY: RMSE = {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
