"""Run V2-T5-nullify ablation: 3 pretrain seeds x 3 eval splits.

Identical to V2-T5 except PairToEdgeWeight output is forced to all-ones.
MLP parameters still exist, receive gradients, and are optimized (bias=+5 init preserved).
The weighted Laplacian path is still exercised (with weight=1).
"""

import argparse
import gc
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
from train_utils import (
    calculate_auc,
    load_model_state_dict,
    ours_loss,
    set_seed,
    train_classification_probe_epoch,
)
from utils_fp_downstream import TestbedDataset


def run_one_seed(pretrain_seed, eval_splits, device, path="down_task_v2"):
    set_seed(pretrain_seed)

    # Hyperparams (match V2-T5 exactly)
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
    print(f"PRETRAIN seed={pretrain_seed} [NULLIFY]")
    print(f"{'='*60}")

    data = TestbedDataset(root=path, dataset="all", task=task, type=fp_type)
    model = LH_Direct_V2(
        in_dim=93, hid_dim=hid_dim, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type=fp_type,
        nullify_pair=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    bias_init = model.pair_to_edge_weight.mlp[2].bias.item()
    nullify_flag = model.pair_to_edge_weight.nullify
    print(f"PairToEdgeWeight bias init: {bias_init:.4f}, nullify={nullify_flag}")

    best_loss = float("inf")
    cnt_wait = 0
    best_t = 0
    tag = f"v2null_{pretrain_seed}_{int(time.time())}"
    ckpt_path = f"pkl/best_spec_model_bace_{tag}.pkl"
    loss_history = []

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
        del loader
        gc.collect()
        torch.cuda.empty_cache()
        avg_loss = total_loss / total_num
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), ckpt_path)
            if epoch % 50 == 0 or epoch < 5:
                bias_now = model.pair_to_edge_weight.mlp[2].bias.item()
                print(f"  Epoch {epoch}: loss={avg_loss:.6f} (new best), bias={bias_now:.4f}")
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    bias_final = model.pair_to_edge_weight.mlp[2].bias.item()
    print(f"  Best epoch: {best_t}, best loss: {best_loss:.6f}")
    print(f"  PairToEdgeWeight bias final: {bias_final:.4f}")

    # Free pretrain resources before eval (4GB GPU headroom)
    del optimizer, data
    gc.collect()
    torch.cuda.empty_cache()

    # Load best checkpoint
    model.load_state_dict(load_model_state_dict(ckpt_path))
    model.eval()

    # Eval
    result = {
        "pretrain_best_epoch": best_t,
        "pretrain_final_loss": round(best_loss, 4),
        "early_stopped": cnt_wait == patience,
        "bias_init": round(bias_init, 4),
        "bias_final": round(bias_final, 4),
        "nullify": True,
        "loss_history_sample": [round(loss_history[i], 6) for i in
                                 [0, 1, 2, 5, 10, 50, 100, 200, 500]
                                 if i < len(loss_history)],
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
            train_classification_probe_epoch(
                train_loader, model, logreg, opt, loss_fn, device, n_task=1
            )

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
        del logreg, opt, train_data, train_loader, val_data, val_loader, test_data, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    split_aucs = [v["test_auc"] for v in result["eval_splits"].values()]
    result["mean_test_auc"] = round(np.mean(split_aucs), 4)
    print(f"  Mean AUC for seed {pretrain_seed}: {result['mean_test_auc']:.4f}")

    # Cleanup checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--only-seed", type=int, default=None,
                         help="Run only this pretrain seed (for subprocess isolation)")
    _args = _parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs("pkl", exist_ok=True)

    pretrain_seeds = [9, 19, 29]
    eval_splits = [9, 19, 29]

    if _args.only_seed is not None:
        pretrain_seeds = [_args.only_seed]

    results_path = "v2_t5_nullify_bace_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "experiment": "V2-T5-nullify ablation on BACE (Uni-Mol fold split)",
            "model": "LH_Direct_V2 (nullify_pair=True: edge_weight forced to 1)",
            "data_root": "down_task_v2",
            "results_per_seed": {},
        }

    for seed in pretrain_seeds:
        if f"seed_{seed}" in all_results.get("results_per_seed", {}):
            print(f"\nSkipping seed {seed} (already done)")
            continue
        result = run_one_seed(seed, eval_splits, device)
        all_results["results_per_seed"][f"seed_{seed}"] = result
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"seed_{seed} saved to {results_path}")

    # Write summary if all 3 seeds are present
    if all(f"seed_{s}" in all_results.get("results_per_seed", {}) for s in [9, 19, 29]):
        all_aucs = []
        for s in [9, 19, 29]:
            all_aucs.extend([v["test_auc"] for v in all_results["results_per_seed"][f"seed_{s}"]["eval_splits"].values()])
        all_results["summary"] = {
            "all_9_test_aucs": all_aucs,
            "grand_mean": round(np.mean(all_aucs), 4),
            "grand_std": round(np.std(all_aucs), 4),
            "per_seed_means": [
                all_results["results_per_seed"][f"seed_{s}"]["mean_test_auc"]
                for s in [9, 19, 29]
            ],
        }
        print(f"\n{'='*60}")
        print(f"V2-T5-NULLIFY: grand mean = {all_results['summary']['grand_mean']:.4f} "
              f"+/- {all_results['summary']['grand_std']:.4f}")
        print(f"V2-T5:         grand mean = 0.837 +/- 0.009")
        print(f"Baseline V0:   grand mean = 0.797 +/- 0.085")
        print(f"FP-only:       grand mean = 0.846 +/- 0.010")
        print(f"{'='*60}")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
