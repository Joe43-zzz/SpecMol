"""Run V2-T7 full BACE pretrain + downstream eval.

T7 = sparse bidirectional pair-biased attention (Uni-Mol-inspired).
After E1 fix (LayerNorm + small-init bias_proj + 1/sqrt(2K) scale on
out_proj/delta_proj), the architecture is sound (no dying attention,
no NaN, all 6 modules learn).

Defaults: 1 pretrain seed (9), 3 eval splits (9/19/29) for fast first
read on whether T7 differentiates from T5 (0.837 ± 0.009 baseline).
Pass --pretrain_seeds 9,19,29 for full 3-seed benchmark.
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
from train_utils import (
    calculate_auc,
    load_model_state_dict,
    ours_loss,
    set_seed,
    train_classification_probe_epoch,
)
from utils_fp_downstream import TestbedDataset


def run_one_seed(pretrain_seed, eval_splits, device, path="down_task_v2",
                 epochs=1000, eval_epochs=500, patience=100,
                 results_path=None, all_results=None,
                 min_eval_epoch=100, eval_improve_tol=1e-4):
    set_seed(pretrain_seed)

    batch_size = 512
    lr = 0.0001
    hid_dim = 512
    K = 10
    task = "bace"
    fp_type = "tri"
    alpha = 1

    print(f"\n{'='*60}")
    print(f"PRETRAIN seed={pretrain_seed} (T7)")
    print(f"{'='*60}")

    data = TestbedDataset(root=path, dataset="all", task=task, type=fp_type)
    model = LH_Direct_V2(
        in_dim=93, hid_dim=hid_dim, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type=fp_type,
        pair_dim=64, t7=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    bias_init = model.pair_to_edge_weight.mlp[2].bias.item()
    pair_attn = model.encoder.prop1.pair_attn
    q_init = pair_attn.q.weight.norm().item()
    op_init = pair_attn.out_proj.weight.norm().item()
    print(f"PairToEdgeWeight bias init: {bias_init:.4f}")
    print(f"T7 q init: {q_init:.4f}, out_proj init: {op_init:.4f}")

    best_loss = float("inf")
    cnt_wait = 0
    best_t = 0
    tag = f"v2_t7_{pretrain_seed}_{int(time.time())}"
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
        avg_loss = total_loss / total_num
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), ckpt_path)
            if epoch % 50 == 0 or epoch < 5:
                op_now = pair_attn.out_proj.weight.norm().item()
                print(f"  Epoch {epoch}: loss={avg_loss:.6f} (new best), out_proj={op_now:.4f}")
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    bias_final = model.pair_to_edge_weight.mlp[2].bias.item()
    op_final = pair_attn.out_proj.weight.norm().item()
    print(f"  Best epoch: {best_t}, best loss: {best_loss:.6f}")
    print(f"  PairToEdgeWeight bias final: {bias_final:.4f}")
    print(f"  T7 out_proj final: {op_final:.4f}")

    del optimizer, data, loader
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.load_state_dict(load_model_state_dict(ckpt_path))
    model.eval()

    result = {
        "pretrain_best_epoch": best_t,
        "pretrain_final_loss": round(best_loss, 4),
        "early_stopped": cnt_wait == patience,
        "bias_init": round(bias_init, 4),
        "bias_final": round(bias_final, 4),
        "t7_out_proj_init": round(op_init, 4),
        "t7_out_proj_final": round(op_final, 4),
        "loss_history_sample": [round(loss_history[i], 6) for i in
                                 [0, 1, 2, 5, 10, 50, 100, 200, 500]
                                 if i < len(loss_history)],
        "eval_splits": {},
    }

    for split_seed in eval_splits:
        print(f"  EVAL split_seed={split_seed}")
        set_seed(split_seed)
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

                # Downstream model-selection guards (mirror main_pretrain.py 2026-05-14 fix):
                # (i) require ep >= min_eval_epoch so tiny BACE val=152 early lucky
                #     peaks cannot lock model selection; (ii) require strict improvement
                #     tol so identical val AUCs don't ping-pong test selection.
                if ep >= min_eval_epoch and val_auc > val_auc_best + eval_improve_tol:
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
        # Incremental save after each split (so a wall-clock timeout still
        # leaves partial results for inspection).
        if results_path is not None and all_results is not None:
            all_results.setdefault("results_per_seed", {})[f"seed_{pretrain_seed}"] = result
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"    [incremental save] {results_path} updated")

    split_aucs = [v["test_auc"] for v in result["eval_splits"].values()]
    result["mean_test_auc"] = round(np.mean(split_aucs), 4)
    print(f"  Mean AUC for seed {pretrain_seed}: {result['mean_test_auc']:.4f}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_seeds", type=str, default="9",
                        help="Comma-separated pretrain seeds (default: 9 for fast first read)")
    parser.add_argument("--eval_splits", type=str, default="9,19,29")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval_epochs", type=int, default=500)
    parser.add_argument("--min_eval_epoch", type=int, default=100,
                        help="Refuse val-best update before this eval epoch (BACE val=152 noise fix)")
    parser.add_argument("--eval_improve_tol", type=float, default=1e-4,
                        help="Strict-improvement tolerance for val_auc selection")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--path", type=str, default="down_task_v2")
    parser.add_argument("--results_path", type=str, default="v2_t7_bace_results.json")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs("pkl", exist_ok=True)

    pretrain_seeds = [int(s.strip()) for s in args.pretrain_seeds.split(",") if s.strip()]
    eval_splits = [int(s.strip()) for s in args.eval_splits.split(",") if s.strip()]

    print(f"Device: {device}")
    print(f"Pretrain seeds: {pretrain_seeds}")
    print(f"Eval splits: {eval_splits}")
    print(f"Pretrain epochs: {args.epochs}, eval epochs: {args.eval_epochs}")

    results_path = args.results_path
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "experiment": "V2-T7 sparse bidirectional pair-biased attention on BACE",
            "model": "LH_Direct_V2(t7=True) — Uni-Mol-style pair-bias attention",
            "data_root": args.path,
            "results_per_seed": {},
        }

    all_aucs = []
    for seed in pretrain_seeds:
        if f"seed_{seed}" in all_results.get("results_per_seed", {}):
            print(f"\nSkipping seed {seed} (already done)")
            result = all_results["results_per_seed"][f"seed_{seed}"]
        else:
            result = run_one_seed(seed, eval_splits, device, path=args.path,
                                   epochs=args.epochs, eval_epochs=args.eval_epochs,
                                   patience=args.patience,
                                   results_path=results_path, all_results=all_results,
                                   min_eval_epoch=args.min_eval_epoch,
                                   eval_improve_tol=args.eval_improve_tol)
            all_results["results_per_seed"][f"seed_{seed}"] = result
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
        all_aucs.extend([v["test_auc"] for v in result["eval_splits"].values()])

    all_results["summary"] = {
        "all_test_aucs": all_aucs,
        "grand_mean": round(np.mean(all_aucs), 4),
        "grand_std": round(np.std(all_aucs), 4),
        "per_seed_means": [
            all_results["results_per_seed"][f"seed_{s}"]["mean_test_auc"]
            for s in pretrain_seeds
        ],
    }

    print(f"\n{'='*60}")
    print(f"FINAL T7: grand mean = {all_results['summary']['grand_mean']:.4f} "
          f"+/- {all_results['summary']['grand_std']:.4f}")
    print(f"V2-T5 baseline: 0.837 +/- 0.009 (pre-B4 data, same data_root)")
    print(f"{'='*60}")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")
