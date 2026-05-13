"""Nullify ablation: 5 cells x 3 seeds reality check on BACE.

Cells:
  C0: V0 baseline (FP+GNN, no pair_repr path)
  C1: V2-T5 (static pair_repr init, no T6 update)
  C2: T6 code path but out_proj zeroed each step (should equal C1)
  C3: T6-full (out_proj + edge_weight MLP both learnable)
  C4: T6 only-update (out_proj learnable, edge_weight MLP frozen bias=5)
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from utils_fp_downstream import TestbedDataset
from model_gnn_pre import LH_Direct, LogReg
from model_gnn_pre_v2 import LH_Direct_V2
from train_utils import calculate_auc, ours_loss, set_seed, train_classification_probe_epoch

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)


# ── Contrastive loss (identical to main_pretrain.py) ──────────────���───────────
def gpu_snapshot():
    """Return 'util%, used MiB / total MiB' or 'N/A'."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'], timeout=5,
        ).decode().strip()
        parts = [x.strip() for x in out.split(',')]
        return f'{parts[0]}%, {parts[1]}/{parts[2]} MiB'
    except Exception:
        return 'N/A'


# ── Cell definitions ──────────────────────────────────────────────────────────
CELLS = {
    'C0': dict(desc='V0 baseline (FP+GNN, no pair)',
               model='v0', data='down_task_2d',
               t6=False, zero_out_proj=False, freeze_ew_mlp=False),
    'C1': dict(desc='V2-T5 (static pair_repr)',
               model='v2', data='down_task_v2',
               t6=False, zero_out_proj=False, freeze_ew_mlp=False),
    'C2': dict(desc='T6 path, out_proj zeroed (should eq C1)',
               model='v2', data='down_task_v2',
               t6=True, zero_out_proj=True, freeze_ew_mlp=False),
    'C3': dict(desc='T6-full',
               model='v2', data='down_task_v2',
               t6=True, zero_out_proj=False, freeze_ew_mlp=False),
    'C4': dict(desc='T6 only-update (ew MLP frozen)',
               model='v2', data='down_task_v2',
               t6=True, zero_out_proj=False, freeze_ew_mlp=True),
}


def build_model(cell_cfg, device, hid_dim=512, K=10):
    if cell_cfg['model'] == 'v0':
        model = LH_Direct(
            in_dim=93, hid_dim=hid_dim, K=K,
            dprate=0.5, dropout=0.0, is_bns=False, act_fn='relu', type='tri',
        )
    else:
        model = LH_Direct_V2(
            in_dim=93, hid_dim=hid_dim, K=K,
            dprate=0.5, dropout=0.0, is_bns=False, act_fn='relu', type='tri',
            pair_dim=64, t6=cell_cfg['t6'],
        )
    model = model.to(device)

    # C4: freeze pair_to_edge_weight MLP (keeps bias=5 init)
    if cell_cfg['freeze_ew_mlp'] and hasattr(model, 'pair_to_edge_weight'):
        model.pair_to_edge_weight.requires_grad_(False)

    return model


# ── Pretraining ───────────────────────────────────────────────────────────────
def pretrain(model, data_root, cell_cfg, device,
             batch_size=256, epochs=1000, patience=100, lr=1e-4):
    data = TestbedDataset(root=data_root, dataset='all', task='bace', type='tri')
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-7,
    )

    best_loss = float('inf')
    cnt_wait = 0
    best_state = None
    best_epoch = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.time()

    for epoch in range(epochs + 1):
        model.train()
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        total_loss, total_num = 0.0, 0

        for batch_data in loader:
            low_x, high_x, spec_x, x_fp = model(batch_data, device)
            loss = ours_loss(low_x, high_x, spec_x, x_fp)
            total_num += batch_data.num_graphs
            total_loss += loss.item() * batch_data.num_graphs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # C2: zero out_proj after every optimizer step
            if cell_cfg['zero_out_proj']:
                prop = getattr(model.encoder, 'prop1', None)
                if prop and hasattr(prop, 'node_to_pair'):
                    prop.node_to_pair.out_proj.weight.data.zero_()

        avg_loss = total_loss / total_num

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            cnt_wait = 0
        else:
            cnt_wait += 1

        # Log sparingly: epoch 0 (VRAM check), then every 100, and final
        if epoch == 0 or epoch % 100 == 0:
            vram_peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            print(f'    [E{epoch:>4}] loss={avg_loss:.6f} best={best_loss:.6f}'
                  f'  GPU: {gpu_snapshot()}  VRAM_peak={vram_peak:.0f}MB')

        if cnt_wait >= patience:
            print(f'    Early stop at epoch {epoch} (best @ {best_epoch})')
            break

    elapsed = time.time() - t_start
    model.load_state_dict(best_state)
    model.eval()
    print(f'    Pretrain done: best_loss={best_loss:.6f} @ epoch {best_epoch}'
          f'  ({elapsed:.0f}s)')
    return best_loss, best_epoch


# ── Downstream LogReg eval ────────────────────────────────────────────────────
def eval_logreg(model, data_root, device, hid_dim=512,
                batch_size=256, eval_epochs=2000, split_seeds=None):
    all_split_seeds = [9, 19, 29, 39, 49]
    if split_seeds is None:
        split_seeds = all_split_seeds
    task_num = 1  # BACE = single-task binary

    results = []
    for ss in split_seeds:
        train_loader = DataLoader(
            TestbedDataset(root=data_root, dataset='train', task='bace', type='tri', seed=ss),
            batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            TestbedDataset(root=data_root, dataset='valid', task='bace', type='tri', seed=ss),
            batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            TestbedDataset(root=data_root, dataset='test', task='bace', type='tri', seed=ss),
            batch_size=batch_size, shuffle=False)

        logreg = LogReg(hid_dim=hid_dim, n_classes=task_num).to(device)
        opt = torch.optim.Adam(logreg.parameters(), lr=0.001, weight_decay=0.0)
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        val_auc_best = 0.0
        test_auc_at_best = 0.0

        for ep in range(eval_epochs):
            train_classification_probe_epoch(
                train_loader, model, logreg, opt, loss_fn, device, n_task=task_num
            )

            with torch.no_grad():
                val_logits_all, val_y_all = [], []
                for batch_data in val_loader:
                    low_x, high_x, spec_x, x_fp, y = model.get_embedding(batch_data, device)
                    embed = torch.cat([spec_x, x_fp], dim=1)
                    val_logits_all.append(logreg(embed))
                    val_y_all.append(y.reshape(-1, task_num))
                vl = torch.cat(val_logits_all, 0)
                vy = torch.cat(val_y_all, 0)
                val_auc = calculate_auc(vy.cpu().numpy(), vl.cpu().numpy(), task_num)

                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    tl_all, ty_all = [], []
                    for batch_data in test_loader:
                        low_x, high_x, spec_x, x_fp, y = model.get_embedding(batch_data, device)
                        embed = torch.cat([spec_x, x_fp], dim=1)
                        tl_all.append(logreg(embed))
                        ty_all.append(y.reshape(-1, task_num))
                    tl = torch.cat(tl_all, 0)
                    ty = torch.cat(ty_all, 0)
                    test_auc_at_best = calculate_auc(
                        ty.cpu().numpy(), tl.cpu().numpy(), task_num)

        print(f'      split={ss}: test_auc={test_auc_at_best:.4f}')
        results.append(test_auc_at_best)

    return float(np.mean(results)), float(np.std(results)), results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--eval_epochs', type=int, default=2000)
    parser.add_argument('--seeds', default='9,19,29')
    parser.add_argument('--cells', default='C0,C1,C2,C3,C4',
                        help='Comma-separated cell names to run')
    parser.add_argument('--eval_splits', type=int, default=5,
                        help='Number of scaffold split seeds for LogReg eval')
    parser.add_argument('--output', default='nullify_ablation_results.json',
                        help='Output JSON filename')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    cell_names = [c.strip() for c in args.cells.split(',')]
    device = args.device

    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Batch size: {args.batch_size}')
    print(f'Pretrain epochs: {args.epochs}, patience: {args.patience}')
    print(f'LogReg eval epochs: {args.eval_epochs}')
    print(f'Seeds: {seeds}')
    print(f'Cells: {cell_names}')

    os.makedirs('pkl', exist_ok=True)

    all_results = {}  # {cell: [mean_auc_per_seed, ...]}

    for cell_name in cell_names:
        cell_cfg = CELLS[cell_name]
        print(f'\n{"="*70}')
        print(f'{cell_name}: {cell_cfg["desc"]}')
        print(f'  model={cell_cfg["model"]} data={cell_cfg["data"]} t6={cell_cfg["t6"]}'
              f' zero_out={cell_cfg["zero_out_proj"]} freeze_ew={cell_cfg["freeze_ew_mlp"]}')
        print(f'{"="*70}')

        cell_aucs = []
        cell_t0 = time.time()
        for seed in seeds:
            print(f'\n  --- {cell_name} / seed={seed} ---')
            set_seed(seed)

            model = build_model(cell_cfg, device)
            pretrain(model, cell_cfg['data'], cell_cfg, device,
                     batch_size=args.batch_size, epochs=args.epochs,
                     patience=args.patience)

            eval_split_seeds = [9, 19, 29, 39, 49][:args.eval_splits]
            print(f'    LogReg eval starting ({args.eval_epochs} ep x {args.eval_splits} splits)...')
            mean_auc, std_auc, split_aucs = eval_logreg(
                model, cell_cfg['data'], device,
                batch_size=args.batch_size, eval_epochs=args.eval_epochs,
                split_seeds=eval_split_seeds,
            )
            print(f'    => AUC across splits: {mean_auc:.4f} +/- {std_auc:.4f}')
            cell_aucs.append(mean_auc)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cell_elapsed = time.time() - cell_t0
        print(f'\n  {cell_name} DONE in {cell_elapsed:.0f}s'
              f'  AUCs={[f"{v:.4f}" for v in cell_aucs]}')
        all_results[cell_name] = cell_aucs

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f'\n\n{"="*70}')
    print('NULLIFY ABLATION SUMMARY  (BACE, batch=256, 3 pretrain seeds)')
    print(f'{"="*70}')

    c0_mean = np.mean(all_results.get('C0', [0]))
    c1_mean = np.mean(all_results.get('C1', [0]))

    print(f'\n{"Cell":<6} | {"mean AUC":>10} | {"std":>8} |'
          f' {"vs C0":>8} | {"vs C1":>8} | Description')
    print('-' * 78)
    for cn in cell_names:
        vals = all_results[cn]
        m, s = np.mean(vals), np.std(vals)
        d0 = f'{(m-c0_mean)*100:+.2f}pp' if 'C0' in all_results else '---'
        d1 = f'{(m-c1_mean)*100:+.2f}pp' if 'C1' in all_results else '---'
        print(f'{cn:<6} | {m:>10.4f} | {s:>8.4f} | {d0:>8} | {d1:>8} | {CELLS[cn]["desc"]}')

    # Save
    out = {'cells': {}, 'config': vars(args)}
    for cn in cell_names:
        out['cells'][cn] = {
            'desc': CELLS[cn]['desc'],
            'values': all_results[cn],
            'mean': float(np.mean(all_results[cn])),
            'std': float(np.std(all_results[cn])),
        }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
