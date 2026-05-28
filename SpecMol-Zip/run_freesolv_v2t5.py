"""Run FreeSolv V2-T5/T7 on FreeSolv with pair_repr regression downstream.

Usage:
    python run_freesolv_v2t5.py --seed 9
    python run_freesolv_v2t5.py  # runs all 3 seeds
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
from train_utils import load_model_state_dict, load_pyg_inmemory_split, ours_loss, set_seed

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)


TASK = "freesolv"
DATA_ROOT_DEFAULT = "down_task_freesolv_v2"
PAIR_DIM_DEFAULT = 64
RESULTS_PATH = "freesolv_v2t5_results.json"


def describe_pair_source(data_root, pair_dim):
    """Infer pair_repr provenance from the dataset root, not just feature width."""
    normalized = os.path.normpath(data_root)
    root_name = os.path.basename(normalized)

    if root_name == "down_task_freesolv_v2":
        return {
            "pair_source": "RDKit 3D + GBF expansion (64-dim, max_dist=10A, sigma=0.5)",
            "exp_suffix": "GBF pair_repr",
        }
    if root_name == "down_task_freesolv_unimol_v2":
        return {
            "pair_source": f"Uni-Mol encoder_pair_rep ({pair_dim}-dim)",
            "exp_suffix": "Uni-Mol pair_repr",
        }
    return {
        "pair_source": f"Unknown pair source (data_root={data_root}, pair_dim={pair_dim})",
        "exp_suffix": f"pair_repr from {root_name}",
    }


def load_freesolv_split(root, split, seed=9):
    """Load a FreeSolv .pt split directly."""
    return load_pyg_inmemory_split(root, "freesolv", split)


def run_one_seed(pretrain_seed, device, t7=False, data_root=DATA_ROOT_DEFAULT,
                 pair_dim=PAIR_DIM_DEFAULT):
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

    variant_tag = "T7" if t7 else "V2-T5"
    print(f"\n{'='*60}")
    print(f"{variant_tag} PRETRAIN seed={pretrain_seed} data_root={data_root} pair_dim={pair_dim}")
    print(f"{'='*60}")

    data = load_freesolv_split(data_root, "all")
    model = LH_Direct_V2(
        in_dim=93, hid_dim=hid_dim, K=K, dprate=0.5, dropout=0.0,
        is_bns=False, act_fn="relu", type=fp_type, pair_dim=pair_dim,
        t7=t7, t7_num_heads=4, t7_head_dim=32, t7_dropout=0.0, t7_init_std=0.02,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    bias_init = model.pair_to_edge_weight.mlp[2].bias.item()
    print(f"PairToEdgeWeight bias init: {bias_init:.4f}")

    best_loss = float("inf")
    cnt_wait = 0
    best_t = 0
    tag = f"freesolv_v2_{pretrain_seed}_{int(time.time())}"
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

    t7_gate_final = None
    if t7:
        try:
            # LH_Direct_V2 nests prop1 inside encoder; pair_attn holds the per-head gate.
            root = getattr(model, "encoder", model)
            pair_attn = getattr(getattr(root, "prop1", None), "pair_attn", None) if hasattr(root, "prop1") else None
            if pair_attn is not None and hasattr(pair_attn, "attn_gate"):
                t7_gate_final = torch.sigmoid(pair_attn.attn_gate.detach()).tolist()
                print(f"  [t7-gate] final per-head sigmoid(gate): {[round(g, 6) for g in t7_gate_final]}")
            else:
                print("  [t7-gate] pair_attn or attn_gate not found")
        except Exception as e:
            print(f"  [t7-gate] failed to read attn_gate: {e}")

    del optimizer, data, loader
    import gc; gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(load_model_state_dict(ckpt_path))
    model.eval()

    # Regression downstream eval
    print(f"  EVAL regression downstream")
    logreg = LogReg(hid_dim=hid_dim, n_classes=1).to(device)
    opt = torch.optim.Adam(logreg.parameters(), lr=0.001, weight_decay=0.0)
    loss_fn = nn.MSELoss(reduction="mean")

    train_data = load_freesolv_split(data_root, "train")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = load_freesolv_split(data_root, "valid")
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_data = load_freesolv_split(data_root, "test")
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    best_val_rmse = float("inf")
    best_test_rmse = float("inf")
    best_eval_epoch = 0

    # Same guards as main_pretrain.py: avoid early-epoch noise locking selection
    # on tiny FreeSolv val set (64 mol). See feedback_no_motivated_reasoning.md.
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

        # Disable Dropout for deterministic val/test inference.
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
        "bias_init": round(bias_init, 4),
        "bias_final": round(bias_final, 4),
        "best_eval_epoch": best_eval_epoch,
        "best_val_rmse": round(best_val_rmse, 4),
        "best_test_rmse": round(best_test_rmse, 4),
    }
    if t7 and t7_gate_final is not None:
        result["t7_gate_final"] = [round(g, 6) for g in t7_gate_final]

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results-path", type=str, default=None,
                        help="Output JSON path. Defaults to freesolv_{variant}_results.json")
    parser.add_argument("--t7", action="store_true",
                        help="Use T7 variant (V2-T5 + pair-biased attention with per-head gates).")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT_DEFAULT,
                        help="Root dir containing processed/freesolv_{train,valid,test,all}.pt")
    parser.add_argument("--pair-dim", type=int, default=PAIR_DIM_DEFAULT,
                        help="pair_repr_edge feature dim recorded in the processed dataset.")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs("pkl", exist_ok=True)

    variant_label = "T7" if args.t7 else "V2-T5"
    default_results_path = (
        "freesolv_t7_bare_results.json" if args.t7 else RESULTS_PATH
    )
    results_path = args.results_path or default_results_path
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else [9, 19, 29]

    pair_meta = describe_pair_source(args.data_root, args.pair_dim)
    pair_source = pair_meta["pair_source"]
    exp_suffix = pair_meta["exp_suffix"]

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "results_per_seed": {},
        }

    # Refresh top-level metadata even when reusing an existing per-seed JSON.
    all_results["experiment"] = f"FreeSolv {variant_label} (LH_Direct_V2 + {exp_suffix})"
    all_results["data_root"] = args.data_root
    all_results["metric"] = "RMSE (lower is better)"
    all_results["pair_repr_source"] = pair_source
    all_results["pair_dim"] = args.pair_dim
    all_results.setdefault("results_per_seed", {})

    for seed in seeds:
        key = f"seed_{seed}"
        if key in all_results.get("results_per_seed", {}):
            print(f"\nSkipping seed {seed} (already done)")
            continue
        result = run_one_seed(
            seed, device, t7=args.t7,
            data_root=args.data_root, pair_dim=args.pair_dim,
        )
        all_results["results_per_seed"][key] = result
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    rmses = [v["best_test_rmse"] for v in all_results["results_per_seed"].values()]
    if rmses:
        all_results["summary"] = {
            "all_test_rmses": rmses,
            "mean_rmse": round(np.mean(rmses), 4),
            "std_rmse": round(np.std(rmses), 4),
        }
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{variant_label} SUMMARY: RMSE = {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
