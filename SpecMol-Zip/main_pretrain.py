import argparse
import json
import os
import time
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from model_gnn_pre import LH_Direct, LogReg
from utils_fp_downstream import TestbedDataset
import torch.nn as nn
from train_utils import (
    calculate_auc,
    calculate_mae,
    calculate_rmse,
    classification_probe_batch,
    load_model_state_dict,
    ours_loss,
    regression_probe_batch,
    set_seed,
    train_classification_probe_epoch,
    train_regression_probe_epoch,
)

REGRESSION_TASKS = ('freesolv', 'esol', 'lipo')

from rdkit import RDLogger

# 禁用RDKit的日志输出
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# Downstream model-selection guards (fix for BACE val/test outlier bug, 2026-05-14;
# tightened 2026-05-15 after seeing seed-29 outliers slip past at epoch 102-118).
# Empirical pattern: tiny val sets (BACE val=152) produce noisy val AUC; early-epoch
# lucky peaks were locking model selection while LogReg dropout was still active during
# eval. We require (i) min epoch before selection, (ii) strict-improvement tolerance.
# Boundary observation (2026-05-15): even MIN=100 leaks — B4-reverted seed 29 and
# V2-T5 / T6 on certain BACE seeds peak val between epoch 102-123 then plateau.
# MIN=200 closes that window; observed healthy best_epochs cluster ≥200 across BBBP V0
# and post-fix BACE V0, so this does not exclude legitimate convergence.
EVAL_MIN_EPOCH = 200
EVAL_IMPROVE_TOL = 1e-4


def train(spect_net, data_loader, train_optimizer, device, alpha):
    spect_net.train()
    total_loss, total_num, train_bar = 0.0, 0, data_loader

    for tem in train_bar:
        low_x, high_x, spec_x, x_fp = spect_net(tem, device)
        
        loss = ours_loss(low_x, high_x, spec_x, x_fp, alpha)
        #print("loss", loss)

        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        #train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))

        train_optimizer.zero_grad()
        loss.backward()
        if hasattr(spect_net, "capture_gradient_diagnostics"):
            spect_net.capture_gradient_diagnostics()
        train_optimizer.step()

    return total_loss / total_num


def should_log_diagnostics(epoch, best_t, interval):
    fixed_epochs = {0, 1, 5, 10, 50, 100, 200, 500}
    if epoch in fixed_epochs or epoch == best_t:
        return True
    return interval > 0 and epoch % interval == 0


def print_model_diagnostics(model, epoch):
    def small(value):
        return f"{value:.6e}"

    stats = getattr(model, "latest_t6_safe_stats", None)
    grad_norms = getattr(model, "latest_grad_norms", None)
    if stats:
        ew = stats["edge_weight"]
        print(
            "[diagnostics] "
            f"epoch={epoch} mode={stats['mode']} gate={small(stats['gate'])} "
            f"delta_pair_ratio={small(stats['delta_pair_ratio'])} "
            f"pair_cosine_to_original={stats['pair_cosine_to_original']:.6f} "
            f"edge_weight_mean={ew['mean']:.6f} edge_weight_std={ew['std']:.6f} "
            f"edge_weight_min={ew['min']:.6f} edge_weight_max={ew['max']:.6f} "
            f"edge_weight_shape={ew['shape']} edge_count={stats['edge_count']} "
            f"edge_weight_finite={ew['finite']}"
        )
    if grad_norms:
        grad_text = " ".join(
            f"{name}_grad={small(value)}" for name, value in sorted(grad_norms.items())
        )
        print(f"[diagnostics] epoch={epoch} {grad_text}")
    current_t6_stats = getattr(model, "latest_current_t6_stats", None)
    if current_t6_stats:
        for branch_stats in current_t6_stats:
            updates = branch_stats.get("updates", [])
            edge_weights = branch_stats.get("edge_weights", [])
            if not updates or not edge_weights:
                continue
            last_update = updates[-1]
            last_edge = edge_weights[-1]
            branch = "high" if branch_stats.get("highpass") else "low"
            print(
                "[diagnostics] "
                f"epoch={epoch} mode=current_t6 branch={branch} "
                f"last_update_step={last_update['step']} "
                f"delta_pair_ratio={small(last_update['delta_pair_ratio'])} "
                f"pair_cosine_to_original={last_update['pair_cosine_to_original']:.6f} "
                f"last_edge_step={last_edge['step']} "
                f"edge_weight_mean={last_edge['mean']:.6f} "
                f"edge_weight_std={last_edge['std']:.6f} "
                f"edge_weight_min={last_edge['min']:.6f} "
                f"edge_weight_max={last_edge['max']:.6f} "
                f"edge_weight_shape={last_edge['shape']} "
                f"edge_weight_finite={last_edge['finite']}"
            )
    current_t7_stats = getattr(model, "latest_current_t7_stats", None)
    if current_t7_stats:
        for branch_stats in current_t7_stats:
            updates = branch_stats.get("updates", [])
            edge_weights = branch_stats.get("edge_weights", [])
            if not updates or not edge_weights:
                continue
            last_update = updates[-1]
            last_edge = edge_weights[-1]
            branch = "high" if branch_stats.get("highpass") else "low"
            print(
                "[diagnostics] "
                f"epoch={epoch} mode=current_t7 branch={branch} "
                f"last_update_step={last_update['step']} "
                f"delta_pair_ratio={small(last_update['delta_pair_ratio'])} "
                f"pair_cosine_to_original={last_update['pair_cosine_to_original']:.6f} "
                f"out_attn_norm={small(last_update['out_attn_norm'])} "
                f"last_edge_step={last_edge['step']} "
                f"edge_weight_mean={last_edge['mean']:.6f} "
                f"edge_weight_std={last_edge['std']:.6f} "
                f"edge_weight_min={last_edge['min']:.6f} "
                f"edge_weight_max={last_edge['max']:.6f} "
                f"edge_weight_shape={last_edge['shape']} "
                f"edge_weight_finite={last_edge['finite']}"
            )


def _tensor_summary(values, n_bins=20):
    """Mean / std / min / max / quantiles / histogram for a 1-D float tensor.

    Output is JSON-friendly Python primitives only. Quantile keys are the
    rounded integer percentile (e.g. {"1": ...}) so they remain stable across
    refactors.
    """
    if values.numel() == 0:
        return {"n": 0}
    values = values.detach().float().flatten().cpu()
    finite = torch.isfinite(values)
    if not bool(finite.all().item()):
        values = values[finite]
        if values.numel() == 0:
            return {"n": 0, "all_non_finite": True}
    quantile_probs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    quantiles = torch.quantile(values, quantile_probs).tolist()
    lo = float(values.min().item())
    hi = float(values.max().item())
    if hi <= lo:
        hist = {"edges": [lo, hi], "counts": [int(values.numel())]}
    else:
        counts = torch.histc(values, bins=n_bins, min=lo, max=hi).tolist()
        edges = torch.linspace(lo, hi, n_bins + 1).tolist()
        hist = {"edges": edges, "counts": [int(c) for c in counts]}
    return {
        "n": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=True).item()) if values.numel() > 1 else 0.0,
        "min": lo,
        "max": hi,
        "quantiles": {
            "1": quantiles[0], "5": quantiles[1], "25": quantiles[2],
            "50": quantiles[3], "75": quantiles[4], "95": quantiles[5],
            "99": quantiles[6],
        },
        "histogram": hist,
    }


def dump_mlp_phi_stats(spec_model, pretrain_data, batch_size, device,
                        task, random_seed, tag, args):
    """Hook the PairToEdgeWeight MLP, run one inference pass over the pretrain
    'all' split, and write distribution statistics to mlp_phi_stats/<tag>.json.

    Two streams are captured:
      raw_pre_sigmoid:                 the [E,1] output of pair_to_edge_weight.mlp,
                                       i.e. the per-direction raw scalar BEFORE
                                       the (raw_ij + raw_ji)/2 symmetrization and
                                       sigmoid. Directly comparable to the
                                       learned bias term b (initialized to +5).
      edge_weight_post_sigmoid_nonloop:
                                       the [E_chembond] edge_weight tensor that
                                       the symmetric-Laplacian construction
                                       actually consumes, with self-loop entries
                                       (hard-coded to 1.0) stripped out.
    """
    out_dir = "mlp_phi_stats"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}.json")

    pte = spec_model.pair_to_edge_weight
    raw_buf = []
    weight_buf = []

    def _mlp_hook(_module, _inp, output):
        raw_buf.append(output.detach())

    def _gate_hook(_module, _inp, output):
        weight_buf.append(output.detach())

    h1 = pte.mlp.register_forward_hook(_mlp_hook)
    h2 = pte.register_forward_hook(_gate_hook)

    try:
        loader = DataLoader(pretrain_data, batch_size=batch_size, shuffle=False)
        n_edges_seen = 0
        max_edges = 200_000
        spec_model.eval()
        with torch.no_grad():
            for batch in loader:
                _ = spec_model(batch, device)
                if weight_buf:
                    n_edges_seen += int(weight_buf[-1].numel())
                if n_edges_seen >= max_edges:
                    break
    finally:
        h1.remove()
        h2.remove()

    raw = torch.cat([t.flatten() for t in raw_buf], dim=0) if raw_buf else torch.empty(0)
    weights_with_loops = (
        torch.cat([t.flatten() for t in weight_buf], dim=0) if weight_buf else torch.empty(0)
    )
    non_loop_mask = weights_with_loops != 1.0
    weights_nonloop = weights_with_loops[non_loop_mask]
    self_loop_fraction = (
        (1.0 - non_loop_mask.float().mean().item()) if weights_with_loops.numel() else 0.0
    )

    bias_param = pte.mlp[2].bias
    bias_value = (
        float(bias_param.detach().item())
        if bias_param is not None and bias_param.numel() == 1
        else None
    )

    variant = "v2_t5"
    if getattr(args, "t6", False):
        variant = "t6"
    elif getattr(args, "t6_safe", False):
        variant = "t6_safe"
    elif getattr(args, "t7", False):
        variant = "t7"

    payload = {
        "task": task,
        "variant": variant,
        "pretrain_seed": random_seed,
        "tag": tag,
        "checkpoint_loaded_from_best_epoch": True,
        "global_bias_b": bias_value,
        "self_loop_fraction": self_loop_fraction,
        "raw_pre_sigmoid": _tensor_summary(raw),
        "edge_weight_post_sigmoid_nonloop": _tensor_summary(weights_nonloop),
        "edge_weight_post_sigmoid_with_loops": _tensor_summary(weights_with_loops),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    raw_s = payload["raw_pre_sigmoid"]
    gate_s = payload["edge_weight_post_sigmoid_nonloop"]
    if raw_s.get("n"):
        print(
            f"[gate-stats] bias={bias_value:.4f}  raw(pre-sig): "
            f"mean={raw_s['mean']:.4f} std={raw_s['std']:.4f} "
            f"min={raw_s['min']:.4f} max={raw_s['max']:.4f}  "
            f"gate(post-sig, non-loop): mean={gate_s['mean']:.4f} "
            f"std={gate_s['std']:.4f} min={gate_s['min']:.4f} "
            f"max={gate_s['max']:.4f}  -> {out_path}"
        )
    else:
        print(f"[gate-stats] WARNING no raw samples captured; wrote stub to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGC Train')
    parser.add_argument('--datafile', default='in-vitro')
    parser.add_argument('--path', default='down_task')
    parser.add_argument('--task', default='bace')
    # NOTE (H3, 2026-05-13 audit): this flag is currently a no-op. The contrastive
    # temperature is hardcoded to tau=0.5 inside train_utils.ours_loss(...). All historical
    # V0/V2-T5/T6 results were produced at tau=0.5; wiring this flag through without
    # changing its default would silently shift effective temperature to 0.1. Kept for
    # future explicit ablations; leave as no-op for now.
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--is_bns', type=bool, default=False)
    parser.add_argument('--act_fn', default='relu', help='activation function')
    parser.add_argument('--K', default=10, type=int)
    parser.add_argument('--random_seed', default=9, type=int)
    parser.add_argument('--hid_dim', default=512, type=int)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--type', default='tri', type=str) #tri 1489 pub 881 maccs 167 erg 441
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--split_seed', default=9, type=int)
    parser.add_argument('--eval_epochs', default=2000, type=int)
    parser.add_argument('--eval_seeds', default='9,19,29,39,49,59,69,79,89,99')
    parser.add_argument('--use_v2', action='store_true',
                        help='Use V2 model (static-pair weighted Laplacian)')
    parser.add_argument('--t6', action='store_true',
                        help='Enable T6 bidirectional pair-node update (requires --use_v2)')
    parser.add_argument('--t6_safe', action='store_true',
                        help='Enable T6-safe pre-Cheb gated pair update (requires --use_v2)')
    parser.add_argument('--t6_safe_frozen_zero', action='store_true',
                        help='Force T6-safe gate to exactly zero for T5-equivalence sanity checks')
    parser.add_argument('--t6_safe_delta_init_std', default=1e-3, type=float,
                        help='Std for T6-safe delta projection init; frozen-zero ignores this')
    parser.add_argument('--t7', action='store_true',
                        help='Enable T7 sparse pair-biased attention with bidirectional pair update '
                             '(requires --use_v2; mutually exclusive with --t6 and --t6_safe)')
    parser.add_argument('--t7_num_heads', default=4, type=int,
                        help='Number of attention heads for T7 (default 4)')
    parser.add_argument('--t7_head_dim', default=32, type=int,
                        help='Per-head dim for T7 (default 32, total attn_dim=heads*head_dim)')
    parser.add_argument('--t7_dropout', default=0.0, type=float,
                        help='Attention dropout for T7 (default 0.0)')
    parser.add_argument('--t7_init_std', default=0.02, type=float,
                        help='Init std for T7 out_proj/delta_proj (default 0.02)')
    parser.add_argument('--t6_warmup_epochs', default=0, type=int,
                        help='Linear LR warmup over the first N epochs (T6 stability). 0 disables (default).')
    parser.add_argument('--bias_init', default=5.0, type=float,
                        help='PairToEdgeWeight MLP output bias initialization. Default 5.0 gives '
                             'sigma(5)=0.993 near-identity gate (preserves all historical V2-T5/T6 '
                             'results). Lower values (e.g. 1.0 -> sigma=0.73, sigma~=0.20) escape '
                             'the saturation basin so the gate can actually learn. Requires --use_v2.')
    parser.add_argument('--randomize_pair', action='store_true',
                        help='Ablation: replace data.pair_repr_edge with same-shape Gaussian noise '
                             'before pair_to_edge_weight. Tests whether the V2-T5 gating gain is '
                             'geometry-driven (Uni-Mol pair) or just per-bond MLP capacity '
                             'exploiting a learnable scalar. Requires --use_v2.')
    parser.add_argument('--fp_only', action='store_true',
                        help='Downstream probe ablation: zero out the spec_x (GNN encoder) half of '
                             'the embed before LogReg, so the linear probe only sees the fingerprint '
                             'MLP output. Used to test whether a benchmark is fingerprint-saturated. '
                             'Pretraining is unchanged; only the linear-probe stage differs.')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Print T6-safe edge-weight, pair-drift, and gradient diagnostics')
    parser.add_argument('--diagnostic_interval', default=0, type=int,
                        help='Optional extra diagnostic logging interval in epochs')
    parser.add_argument('--dump_gate_stats', action='store_true',
                        help='After loading the best checkpoint, run one inference pass to dump '
                             'the V2-T5 / T6 PairToEdgeWeight raw MLP outputs and final edge gates '
                             'to mlp_phi_stats/<tag>.json (used to evidence that the gate actually '
                             'moves away from the b=+5 identity bias). Requires --use_v2.')
    # args parse
    args = parser.parse_args()
    print(args)
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    
    batch_size, epochs = args.batch_size, args.epochs
    task, random_seed = args.task, args.random_seed
    
    set_seed(random_seed)
    
    clr_tasks = {'bbbp': 1, 'hiv': 1, 'bace': 1, 'tox21': 12, 'clintox': 2, 'sider': 27, 'MUV': 17, 'toxcast':617, 'PCBA':128, 'ecoli':1,
                 'freesolv': 1, 'esol': 1, 'lipo': 1}
    task_num = clr_tasks[task]
    task_type = 'regression' if task in REGRESSION_TASKS else 'classification'
    datafile = 'now'
    save_name_pre = '{}_{}_{}'.format(batch_size, epochs, datafile)
    os.makedirs('results/'+save_name_pre, exist_ok=True)

    data = TestbedDataset(root=args.path, dataset='all', task=task, type=args.type)

    if args.use_v2:
        from model_gnn_pre_v2 import LH_Direct_V2
        spec_model = LH_Direct_V2(
            in_dim=93,
            hid_dim=args.hid_dim,
            K=args.K,
            dprate=args.dprate,
            dropout=args.dropout,
            is_bns=args.is_bns,
            act_fn=args.act_fn,
            type=args.type,
            t6=args.t6,
            t6_safe=args.t6_safe,
            t6_safe_frozen_zero=args.t6_safe_frozen_zero,
            t6_safe_delta_init_std=args.t6_safe_delta_init_std,
            t7=args.t7,
            t7_num_heads=args.t7_num_heads,
            t7_head_dim=args.t7_head_dim,
            t7_dropout=args.t7_dropout,
            t7_init_std=args.t7_init_std,
            bias_init=args.bias_init,
            randomize_pair=args.randomize_pair,
        )
    else:
        spec_model = LH_Direct(
            in_dim=93,
            hid_dim=args.hid_dim,
            K=args.K,
            dprate=args.dprate,
            dropout=args.dropout,
            is_bns=args.is_bns,
            act_fn=args.act_fn,
            type=args.type,
        )
    spec_model = spec_model.to(args.device)
    optimizer = optim.Adam(spec_model.parameters(), lr=args.lr, weight_decay=1e-7) #TODO: spec

    best_loss = float("inf")
    cnt_wait = 0
    os.makedirs("pkl", exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    tag = f"{args.task}_seed{random_seed}_job{job_id}_{int(time.time())}"
    ckpt_path = os.path.join("pkl", f"best_spec_model_{tag}.pkl")
    best_t = 0
    for epoch in range(0, epochs + 1):
        # P0.4: optional linear LR warmup for T6 stability. No-op when --t6_warmup_epochs=0 (default).
        if args.t6_warmup_epochs > 0 and epoch < args.t6_warmup_epochs:
            warm_lr = args.lr * (epoch + 1) / args.t6_warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = warm_lr
        elif args.t6_warmup_epochs > 0 and epoch == args.t6_warmup_epochs:
            for g in optimizer.param_groups:
                g['lr'] = args.lr
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        train_loss = train(spec_model, data_loader, optimizer, args.device, args.alpha)

        if train_loss < best_loss:
            best_loss = train_loss
            best_t = epoch
            print(f'In Epoch {epoch}th, the train loss is {best_loss}.')
            torch.save(spec_model.state_dict(), ckpt_path) #TODO: mlp save
            #print(ckpt_path)
            cnt_wait = 0
        else:
            cnt_wait +=1
        
        if cnt_wait ==args.patience:
            print("Early stopping!")
            break 
        if args.diagnostics and should_log_diagnostics(epoch, best_t, args.diagnostic_interval):
            print_model_diagnostics(spec_model, epoch)
        
    print('Loading {}th eppoch'.format( best_t + 1 ))
    
    spec_model.load_state_dict(load_model_state_dict(ckpt_path))
    spec_model.eval()

    # ------------------------------------------------------------------
    # Optional: dump V2-T5 / T6 gate statistics for the paper (M5 hook).
    # ------------------------------------------------------------------
    if args.dump_gate_stats and args.use_v2 and hasattr(spec_model, "pair_to_edge_weight"):
        dump_mlp_phi_stats(
            spec_model=spec_model,
            pretrain_data=data,
            batch_size=batch_size,
            device=args.device,
            task=task,
            random_seed=random_seed,
            tag=tag,
            args=args,
        )
    elif args.dump_gate_stats:
        print("[gate-stats] skipped: requires --use_v2 with a PairToEdgeWeight module")

    # ------------------------------------------------------------------
    # T7: log the final attention gate (VeriMAP S1/S2 diagnostic).
    # The gate is a per-head [num_heads] vector on pair_attn after S2; before
    # S2 it is a scalar on prop1. sigmoid(...).flatten() handles both shapes,
    # so this one block serves the S1 (scalar) and S2 (per-head) runs alike.
    # ------------------------------------------------------------------
    if args.t7:
        prop1 = spec_model.encoder.prop1
        gate = getattr(getattr(prop1, "pair_attn", None), "attn_gate", None)
        if gate is None:
            gate = getattr(prop1, "attn_gate", None)
        if gate is not None:
            sig = [round(float(s), 6)
                   for s in torch.sigmoid(gate.detach()).flatten().tolist()]
            print(f"[t7-gate] final per-head sigmoid(gate): {sig}")
        else:
            print("[t7-gate] WARNING: --t7 set but no attn_gate found on prop1")

    if task_type == 'regression':
        loss_fn = nn.MSELoss(reduction='mean')
    else:
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    
    split_seeds = [int(seed.strip()) for seed in str(args.eval_seeds).split(',') if seed.strip()]
    for split_seed in split_seeds:
        set_seed(split_seed)
        logreg = LogReg(hid_dim=args.hid_dim, n_classes=task_num).to(args.device)
        opt = torch.optim.Adam(logreg.parameters(), lr=0.001, weight_decay=0.0) #original:0.01
        # Deterministic shuffle generator: same split_seed gives same batch order
        # across runs, eliminating one source of restart-to-restart variance.
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(split_seed)
        train_data = TestbedDataset(root=args.path, dataset='train', task=task, type=args.type,seed=split_seed)
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=shuffle_gen)
        val_data = TestbedDataset(root=args.path, dataset='valid', task=task, type=args.type, seed=split_seed)
        val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_data = TestbedDataset(root=args.path, dataset='test', task=task, type=args.type,seed=split_seed)
        test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        if task_type == 'regression':
            best_val_rmse = float('inf')
            best_epoch = 0
            test_rmse = float('inf')
            test_mae = float('inf')
            for epoch in range(args.eval_epochs):
                train_regression_probe_epoch(
                    train_data_loader, spec_model, logreg, opt, loss_fn, args.device
                )
                logreg.eval()
                with torch.no_grad():
                    val_preds = torch.Tensor().to(args.device)
                    val_y = torch.Tensor().to(args.device)
                    for tem in val_data_loader:
                        preds, y = regression_probe_batch(tem, spec_model, logreg, args.device)
                        val_preds = torch.cat((val_preds, preds), 0)
                        val_y = torch.cat((val_y, y), 0)
                    val_rmse = calculate_rmse(val_y, val_preds)
                    if epoch >= EVAL_MIN_EPOCH and val_rmse < best_val_rmse - EVAL_IMPROVE_TOL:
                        print('Val rmse in Epoch {} is {}'.format(epoch, val_rmse))
                        best_val_rmse = val_rmse
                        test_preds = torch.Tensor().to(args.device)
                        test_y = torch.Tensor().to(args.device)
                        for tem in test_data_loader:
                            preds, y = regression_probe_batch(tem, spec_model, logreg, args.device)
                            test_preds = torch.cat((test_preds, preds), 0)
                            test_y = torch.cat((test_y, y), 0)
                        test_rmse = calculate_rmse(test_y, test_preds)
                        test_mae = calculate_mae(test_y, test_preds)
                        print('Test rmse in Epoch {} is {} (mae {})'.format(epoch, test_rmse, test_mae))
                        best_epoch = epoch
            print('Best Test RMSE for {} in Epoch {} is {} (MAE {})'.format(args.task, best_epoch, test_rmse, test_mae))
        else:
            val_auc_best = 0
            best_epoch = 0
            test_auc = 0.0
            for epoch in range(args.eval_epochs):
                train_classification_probe_epoch(
                    train_data_loader, spec_model, logreg, opt, loss_fn, args.device, n_task=task_num,
                    fp_only=args.fp_only,
                )

                # Disable Dropout during val/test inference (LogReg has p=0.2 dropout that
                # otherwise stays active under no_grad and amplifies AUC noise on small val sets).
                logreg.eval()
                with torch.no_grad():
                    val_logits = torch.Tensor().to(args.device)
                    val_y = torch.Tensor().to(args.device)
                    for tem in val_data_loader:
                        logits, y = classification_probe_batch(tem, spec_model, logreg, args.device, task_num, fp_only=args.fp_only)
                        val_logits = torch.cat((val_logits, logits),0)
                        val_y = torch.cat((val_y, y),0)
                    val_auc = calculate_auc(val_y.cpu().numpy(), val_logits.cpu().numpy(), task_num)
                    # Two-part guard against early-epoch noise locking model selection:
                    # (1) require minimum epochs before any selection (avoid lucky-spike trap);
                    # (2) require improvement >= EVAL_IMPROVE_TOL (avoid sub-noise updates).
                    if epoch >= EVAL_MIN_EPOCH and val_auc > val_auc_best + EVAL_IMPROVE_TOL:
                        print('Val auc in Epoch {} is {}'.format(epoch, val_auc))
                        val_auc_best = val_auc
                        test_logits = torch.Tensor().to(args.device)
                        test_y = torch.Tensor().to(args.device)
                        for tem in test_data_loader:
                            logits, y = classification_probe_batch(tem, spec_model, logreg, args.device, task_num, fp_only=args.fp_only)
                            test_logits = torch.cat((test_logits, logits),0)
                            test_y = torch.cat((test_y, y),0)
                        test_auc = calculate_auc(test_y.cpu().numpy(), test_logits.cpu().numpy(), task_num)
                        print('Test auc in Epoch {} is {}'.format(epoch, test_auc))
                        best_epoch = epoch
            print('Best Test Auc for {} in Epoch {} is {}'.format(args.task, best_epoch, test_auc))
            
        
