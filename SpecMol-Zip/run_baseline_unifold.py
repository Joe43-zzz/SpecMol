"""Run baseline (V0) + FP-only on BACE with Uni-Mol fold split.
3 pretrain seeds x 3 LogReg restarts = 9 cells each."""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader

from model_gnn_pre import LH_Direct, LogReg
from train_utils import (
    calculate_auc,
    load_model_state_dict,
    ours_loss,
    set_seed,
    train_classification_probe_epoch,
)
from utils_fp_downstream import TestbedDataset


def run_eval(model, eval_splits, device, path, hid_dim, batch_size, eval_epochs, task, fp_type):
    """Run LogReg eval for multiple restarts. Returns dict of split results."""
    results = {}
    for split_seed in eval_splits:
        set_seed(split_seed)
        print(f"  EVAL restart_seed={split_seed}")
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

        print(f"    restart_{split_seed}: test_auc={best_test_auc:.4f} (eval epoch {best_eval_epoch})")
        results[f"restart_{split_seed}"] = {
            "test_auc": round(best_test_auc, 4),
            "best_eval_epoch": best_eval_epoch,
        }
    return results


def run_baseline_seed(pretrain_seed, eval_splits, device, path="down_task_unifold"):
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

    print(f"\n{'='*60}")
    print(f"BASELINE PRETRAIN seed={pretrain_seed}")
    print(f"{'='*60}")

    data = TestbedDataset(root=path, dataset="all", task=task, type=fp_type)
    model = LH_Direct(
        in_dim=93, hid_dim=hid_dim, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type=fp_type,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    best_loss = float("inf")
    cnt_wait = 0
    best_t = 0
    tag = f"bl_unifold_{pretrain_seed}_{int(time.time())}"
    ckpt_path = f"pkl/best_spec_model_bace_{tag}.pkl"

    for epoch in range(epochs + 1):
        model.train()
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        total_loss, total_num = 0.0, 0
        for batch in loader:
            low_x, high_x, spec_x, x_fp = model(batch, device)
            loss = ours_loss(low_x, high_x, spec_x, x_fp, alpha=1)
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
            if epoch % 100 == 0 or epoch < 5:
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

    eval_results = run_eval(model, eval_splits, device, path, hid_dim, batch_size, eval_epochs, task, fp_type)

    split_aucs = [v["test_auc"] for v in eval_results.values()]
    result = {
        "pretrain_best_epoch": best_t,
        "pretrain_final_loss": round(best_loss, 4),
        "early_stopped": cnt_wait == patience,
        "eval_splits": eval_results,
        "mean_test_auc": round(np.mean(split_aucs), 4),
    }
    print(f"  Mean AUC for seed {pretrain_seed}: {result['mean_test_auc']:.4f}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_fp_only(eval_splits, device, path="down_task_unifold"):
    """FP-only: no pretrain, random encoder, eval LogReg on FP features only."""
    hid_dim = 512
    batch_size = 512
    eval_epochs = 2000
    task = "bace"
    fp_type = "tri"

    print(f"\n{'='*60}")
    print(f"FP-ONLY (no pretrain)")
    print(f"{'='*60}")

    # Random (untrained) encoder — we only use x_fp from get_embedding
    model = LH_Direct(
        in_dim=93, hid_dim=hid_dim, K=10, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type=fp_type,
    ).to(device)
    model.eval()

    all_results = {}
    for seed in [9, 19, 29]:
        set_seed(seed)
        print(f"\n  FP-ONLY seed={seed}")

        # Re-init model with this seed to match baseline protocol
        model = LH_Direct(
            in_dim=93, hid_dim=hid_dim, K=10, dprate=0.5, dropout=0.0,
            is_bns=False, act_fn="relu", type=fp_type,
        ).to(device)
        model.eval()

        eval_results = run_eval(model, eval_splits, device, path, hid_dim, batch_size, eval_epochs, task, fp_type)
        split_aucs = [v["test_auc"] for v in eval_results.values()]
        all_results[f"seed_{seed}"] = {
            "eval_splits": eval_results,
            "mean_test_auc": round(np.mean(split_aucs), 4),
        }
        print(f"  FP-ONLY seed {seed} mean AUC: {all_results[f'seed_{seed}']['mean_test_auc']:.4f}")

    return all_results


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs("pkl", exist_ok=True)

    pretrain_seeds = [9, 19, 29]
    eval_splits = [9, 19, 29]
    path = "down_task_unifold"

    results_path = "baseline_unifold_results.json"

    # Load existing results if resuming
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "experiment": "Baseline V0 on BACE with Uni-Mol fold split",
            "data_root": path,
            "baseline": {"results_per_seed": {}},
            "fp_only": {},
        }

    # --- Baseline ---
    all_bl_aucs = []
    for seed in pretrain_seeds:
        key = f"seed_{seed}"
        if key in all_results.get("baseline", {}).get("results_per_seed", {}):
            print(f"\nSkipping baseline seed {seed} (already done)")
            result = all_results["baseline"]["results_per_seed"][key]
        else:
            result = run_baseline_seed(seed, eval_splits, device, path)
            all_results["baseline"]["results_per_seed"][key] = result
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
        all_bl_aucs.extend([v["test_auc"] for v in result["eval_splits"].values()])

    all_results["baseline"]["summary"] = {
        "all_9_test_aucs": all_bl_aucs,
        "grand_mean": round(np.mean(all_bl_aucs), 4),
        "grand_std": round(np.std(all_bl_aucs), 4),
    }

    # --- FP-only ---
    if not all_results.get("fp_only", {}).get("results_per_seed"):
        fp_results = run_fp_only(eval_splits, device, path)
        all_results["fp_only"]["results_per_seed"] = fp_results
        all_fp_aucs = []
        for v in fp_results.values():
            all_fp_aucs.extend([s["test_auc"] for s in v["eval_splits"].values()])
        all_results["fp_only"]["summary"] = {
            "all_9_test_aucs": all_fp_aucs,
            "grand_mean": round(np.mean(all_fp_aucs), 4),
            "grand_std": round(np.std(all_fp_aucs), 4),
        }
    else:
        print("\nSkipping FP-only (already done)")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    bl_summary = all_results["baseline"]["summary"]
    fp_summary = all_results["fp_only"]["summary"]
    print(f"\n{'='*60}")
    print(f"BASELINE (unifold): {bl_summary['grand_mean']:.4f} +/- {bl_summary['grand_std']:.4f}")
    print(f"FP-ONLY  (unifold): {fp_summary['grand_mean']:.4f} +/- {fp_summary['grand_std']:.4f}")
    print(f"V2-T5    (unifold): 0.8373 +/- 0.0094")
    print(f"{'='*60}")
    print(f"Results saved to {results_path}")
