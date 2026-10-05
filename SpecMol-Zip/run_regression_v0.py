"""Generic V0 (2D-only) baseline runner for regression tasks (ESOL, Lipo, FreeSolv).

V0 uses LH_Direct (no pair_repr, no gate). Re-uses the same Uni-Mol-scaffold-split
.pt files as V2-T5 (down_task_{task}_unimol_v2/) so V0 and V2-T5 numbers are
matched cell-for-cell on the same molecule cohort. The V0 model simply ignores
the pair_repr_edge / pair_edge_index fields that LH_Direct_V2 consumes.

Usage:
    python run_regression_v0.py --task esol --seed 9
    python run_regression_v0.py --task lipo --seed 19
    python run_regression_v0.py --task freesolv         # all 3 seeds
"""

import argparse
import gc
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader

from model_gnn_pre import LH_Direct, LogReg
from train_utils import load_model_state_dict, load_pyg_inmemory_split, ours_loss, set_seed

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)


REGRESSION_TASKS = ("freesolv", "esol", "lipo")


def default_data_root(task):
    """Re-use V2-T5 .pt files: LH_Direct simply ignores pair_repr fields."""
    return f"down_task_{task}_unimol_v2"


def load_task_split(root, task, split):
    return load_pyg_inmemory_split(root, task, split)


def run_one_seed(task, pretrain_seed, device, data_root=None):
    set_seed(pretrain_seed)

    if data_root is None:
        data_root = default_data_root(task)

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
    print(f"V0 BASELINE PRETRAIN task={task} seed={pretrain_seed} data_root={data_root}")
    print(f"{'='*60}")

    data = load_task_split(data_root, task, "all")
    model = LH_Direct(
        in_dim=93, hid_dim=hid_dim, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type=fp_type,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    best_loss = float("inf")
    cnt_wait = 0
    best_t = 0
    tag = f"{task}_v0_{pretrain_seed}_{int(time.time())}"
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
    gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(load_model_state_dict(ckpt_path))
    model.eval()

    print(f"  EVAL regression downstream")
    logreg = LogReg(hid_dim=hid_dim, n_classes=1).to(device)
    opt = torch.optim.Adam(logreg.parameters(), lr=0.001, weight_decay=0.0)
    loss_fn = nn.MSELoss(reduction="mean")

    train_data = load_task_split(data_root, task, "train")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = load_task_split(data_root, task, "valid")
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_data = load_task_split(data_root, task, "test")
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    best_val_rmse = float("inf")
    best_test_rmse = float("inf")
    best_eval_epoch = 0

    EVAL_MIN_EPOCH = 100
    EVAL_IMPROVE_TOL = 1e-4

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

        logreg.eval()
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

            if ep >= EVAL_MIN_EPOCH and val_rmse < best_val_rmse - EVAL_IMPROVE_TOL:
                best_val_rmse = val_rmse
                best_eval_epoch = ep
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
    parser.add_argument("--task", required=True, choices=REGRESSION_TASKS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results-path", type=str, default=None,
                        help="Output JSON path. Defaults to {task}_v0_results.json")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root dir containing processed/{task}_{train,valid,test,all}.pt. "
                             "Defaults to down_task_{task}_unimol_v2 (re-uses V2-T5 .pt; "
                             "V0 model ignores pair_repr fields).")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs("pkl", exist_ok=True)

    task = args.task
    data_root = args.data_root or default_data_root(task)

    results_path = args.results_path or f"{task}_v0_results.json"
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else [9, 19, 29]

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {"results_per_seed": {}}

    all_results["experiment"] = f"{task.upper()} V0 (LH_Direct 2D baseline)"
    all_results["task"] = task
    all_results["data_root"] = data_root
    all_results["metric"] = "RMSE (lower is better)"
    all_results.setdefault("results_per_seed", {})

    rmses = []
    for s in seeds:
        result = run_one_seed(task=task, pretrain_seed=s, device=device, data_root=data_root)
        all_results["results_per_seed"][f"seed_{s}"] = result
        rmses.append(result["best_test_rmse"])
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    if rmses:
        mean = sum(rmses) / len(rmses)
        std = (sum((x - mean) ** 2 for x in rmses) / max(1, len(rmses) - 1)) ** 0.5
        all_results["summary"] = {
            "all_test_rmses": [round(x, 4) for x in rmses],
            "mean_rmse": round(mean, 4),
            "std_rmse": round(std, 4),
        }
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nV0 SUMMARY ({task}): RMSE = {mean:.4f} +/- {std:.4f}")
