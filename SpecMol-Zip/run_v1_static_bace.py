"""Run V1 static Uni-Mol edge-weight experiment on BACE.

V1 definition for the fair BACE table:
  - dataset: BACE only
  - split: Uni-Mol scaffold fold CSV already materialized in down_task_v2
  - topology: original chemical bond edge_index only
  - 3D signal: static edge_weight = softplus(mean(pair_repr[i,j,:]))
  - no learned pair-to-edge mapper, no normalization, no bond-type scaling
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool as gmp

from model_gnn_pre import ChebNetII, LogReg
from train_utils import (
    calculate_auc,
    load_model_state_dict,
    ours_loss,
    set_seed,
    train_classification_probe_epoch,
)
from utils_fp_downstream import TestbedDataset


PRETRAIN_SEEDS = [9, 19, 29]
EVAL_SPLITS = [9, 19, 29]


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class LHDirectV1Static(nn.Module):
    """Baseline LH_Direct with fixed Uni-Mol scalar weights on chem bond edges."""

    def __init__(self, in_dim, hid_dim, k, dprate, dropout, is_bns, act_fn, fp_type="tri"):
        super().__init__()
        self.encoder = ChebNetII(
            num_features=in_dim,
            hidden=hid_dim,
            K=k,
            dprate=dprate,
            dropout=dropout,
            is_bns=is_bns,
            act_fn=act_fn,
        )
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        if fp_type == "tri":
            self.mlp_input = 1489
        elif fp_type == "pub":
            self.mlp_input = 881
        elif fp_type == "maccs":
            self.mlp_input = 167
        elif fp_type == "erg":
            self.mlp_input = 441
        else:
            raise ValueError(f"unsupported fp_type={fp_type}")
        self.mlp = MLP(input_size=self.mlp_input, hidden_size=1024, output_size=hid_dim, dropout=0.2)

    @staticmethod
    def compute_static_edge_weight(data, device):
        pair_repr_edge = data.pair_repr_edge.to(device)
        pair_edge_index = data.pair_edge_index.to(device)
        chem_edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        n_total = batch.size(0)

        all_enc = pair_edge_index[0] * n_total + pair_edge_index[1]
        all_enc_sorted, sort_perm = all_enc.sort()
        chem_enc = chem_edge_index[0] * n_total + chem_edge_index[1]
        pos = torch.searchsorted(all_enc_sorted, chem_enc).clamp(max=all_enc_sorted.size(0) - 1)
        if not (all_enc_sorted[pos] == chem_enc).all():
            raise ValueError("Some chem bond edges were not found in pair_edge_index")

        pair_scalar = F.softplus(pair_repr_edge.mean(dim=-1))
        return pair_scalar[sort_perm[pos]].to(torch.float32)

    def _forward_impl(self, data, device):
        feat = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        fp = data.fps.to(device)
        edge_weight = self.compute_static_edge_weight(data, device)

        h1 = self.encoder(x=feat, edge_index=edge_index, edge_weight=edge_weight, highpass=True)
        high_x_mean = gmp(h1, batch)
        h2 = self.encoder(x=feat, edge_index=edge_index, edge_weight=edge_weight, highpass=False)
        low_x_mean = gmp(h2, batch)
        h = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        spec_x_mean = gmp(h, batch)

        fp = fp.reshape(len(fp) // self.mlp_input, self.mlp_input)
        x_fp = self.mlp(fp)
        return low_x_mean, high_x_mean, spec_x_mean, x_fp

    def get_embedding(self, data, device):
        low_x_mean, high_x_mean, spec_x_mean, x_fp = self._forward_impl(data, device)
        return low_x_mean, high_x_mean, spec_x_mean, x_fp, data.y.to(device)

    def forward(self, data, device):
        return self._forward_impl(data, device)


def edge_weight_stats(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    weights = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            weights.append(model.compute_static_edge_weight(batch, device).cpu())
    weight = torch.cat(weights)
    return {
        "mean": round(float(weight.mean()), 6),
        "std": round(float(weight.std()), 6),
        "min": round(float(weight.min()), 6),
        "max": round(float(weight.max()), 6),
    }


def run_one_seed(pretrain_seed, eval_splits, device, args):
    set_seed(pretrain_seed)

    print(f"\n{'=' * 60}")
    print(f"V1 STATIC PRETRAIN seed={pretrain_seed}")
    print(f"{'=' * 60}")

    data = TestbedDataset(root=args.data_root, dataset="all", task="bace", type=args.fp_type)
    model = LHDirectV1Static(
        in_dim=93,
        hid_dim=args.hid_dim,
        k=args.k,
        dprate=args.dprate,
        dropout=args.dropout,
        is_bns=False,
        act_fn="relu",
        fp_type=args.fp_type,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-7)

    initial_stats = edge_weight_stats(model, data, device, args.batch_size)
    print(f"  Static edge_weight stats: {initial_stats}")

    best_loss = float("inf")
    cnt_wait = 0
    best_t = 0
    loss_history = []
    tag = f"v1static_{pretrain_seed}_{int(time.time())}"
    ckpt_path = os.path.join("pkl", f"best_spec_model_bace_{tag}.pkl")

    for epoch in range(args.epochs + 1):
        model.train()
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
        total_loss = 0.0
        total_num = 0
        for batch in loader:
            low_x, high_x, spec_x, x_fp = model(batch, device)
            loss = ours_loss(low_x, high_x, spec_x, x_fp, alpha=args.alpha)
            total_num += len(batch)
            total_loss += float(loss.item()) * len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / max(total_num, 1)
        loss_history.append(avg_loss)
        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), ckpt_path)
            if epoch % 50 == 0 or epoch < 5:
                print(f"  Epoch {epoch}: loss={avg_loss:.6f} (new best)")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    del optimizer, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.load_state_dict(load_model_state_dict(ckpt_path))
    model.eval()

    result = {
        "pretrain_best_epoch": best_t,
        "pretrain_final_loss": round(best_loss, 4),
        "early_stopped": cnt_wait == args.patience,
        "edge_weight_formula": "softplus(mean(pair_repr_edge, dim=-1)) on chemical bond edge_index",
        "normalization": "none",
        "bond_type_scaling": "none",
        "missing_3d_fallback": "none; down_task_v2 contains pair_repr_edge for all retained BACE molecules",
        "initial_edge_weight_stats": initial_stats,
        "loss_history_sample": [
            round(loss_history[i], 6)
            for i in [0, 1, 2, 5, 10, 50, 100, 200, 500]
            if i < len(loss_history)
        ],
        "eval_splits": {},
    }

    for split_seed in eval_splits:
        print(f"  EVAL split_seed={split_seed}")
        logreg = LogReg(hid_dim=args.hid_dim, n_classes=1).to(device)
        opt = optim.Adam(logreg.parameters(), lr=args.logreg_lr, weight_decay=0.0)
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        train_data = TestbedDataset(root=args.data_root, dataset="train", task="bace", type=args.fp_type, seed=split_seed)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_data = TestbedDataset(root=args.data_root, dataset="valid", task="bace", type=args.fp_type, seed=split_seed)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_data = TestbedDataset(root=args.data_root, dataset="test", task="bace", type=args.fp_type, seed=split_seed)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        val_auc_best = 0.0
        best_test_auc = 0.0
        best_eval_epoch = 0

        for ep in range(args.eval_epochs):
            train_classification_probe_epoch(
                train_loader, model, logreg, opt, loss_fn, device, n_task=1
            )

            with torch.no_grad():
                logreg.eval()
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
                    val_auc_best = val_auc
                    best_test_auc = calculate_auc(test_y.cpu().numpy(), test_logits.cpu().numpy(), 1)
                    best_eval_epoch = ep

        print(f"    split_{split_seed}: test_auc={best_test_auc:.4f} (eval epoch {best_eval_epoch})")
        result["eval_splits"][f"split_{split_seed}"] = {
            "test_auc": round(best_test_auc, 4),
            "best_eval_epoch": best_eval_epoch,
        }
        del logreg, opt, train_data, train_loader, val_data, val_loader, test_data, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    split_aucs = [v["test_auc"] for v in result["eval_splits"].values()]
    result["mean_test_auc"] = round(float(np.mean(split_aucs)), 4)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--data-root", default="down_task_v2")
    parser.add_argument("--results-path", default="v1_static_bace_results.json")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--logreg-lr", type=float, default=1e-3)
    parser.add_argument("--hid-dim", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dprate", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--fp-type", default="tri")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if args.gpu != -1 and torch.cuda.is_available() else "cpu"
    os.makedirs("pkl", exist_ok=True)

    pretrain_seeds = [args.only_seed] if args.only_seed is not None else PRETRAIN_SEEDS
    eval_splits = EVAL_SPLITS

    if os.path.exists(args.results_path):
        with open(args.results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "experiment": "V1 static Uni-Mol scalar edge weights on BACE",
            "model": "LHDirectV1Static (ChebNetII + fixed Uni-Mol edge_weight + FP MLP)",
            "data_root": args.data_root,
            "split_source": "Uni-Mol scaffold k10 seed42; fold 0=test, fold 1=valid",
            "graph_topology": "original chemical bond edge_index from down_task_v2",
            "pair_repr_edge": True,
            "edge_weight_3d": "computed on the fly, not stored",
            "seeds": {"pretrain": PRETRAIN_SEEDS, "eval_restarts": EVAL_SPLITS},
            "results_per_seed": {},
        }

    for seed in pretrain_seeds:
        key = f"seed_{seed}"
        if key in all_results.get("results_per_seed", {}):
            print(f"\nSkipping seed {seed} (already done)")
            continue
        all_results["results_per_seed"][key] = run_one_seed(seed, eval_splits, device, args)
        with open(args.results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"seed_{seed} saved to {args.results_path}")

    if all(f"seed_{s}" in all_results.get("results_per_seed", {}) for s in PRETRAIN_SEEDS):
        all_aucs = []
        for seed in PRETRAIN_SEEDS:
            all_aucs.extend([
                v["test_auc"]
                for v in all_results["results_per_seed"][f"seed_{seed}"]["eval_splits"].values()
            ])
        all_results["summary"] = {
            "all_9_test_aucs": all_aucs,
            "grand_mean": round(float(np.mean(all_aucs)), 4),
            "grand_std": round(float(np.std(all_aucs)), 4),
            "per_seed_means": [
                all_results["results_per_seed"][f"seed_{seed}"]["mean_test_auc"]
                for seed in PRETRAIN_SEEDS
            ],
        }
        print(f"\n{'=' * 60}")
        print(
            f"V1 STATIC: grand mean = {all_results['summary']['grand_mean']:.4f} "
            f"+/- {all_results['summary']['grand_std']:.4f}"
        )
        print(f"{'=' * 60}")
        with open(args.results_path, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
