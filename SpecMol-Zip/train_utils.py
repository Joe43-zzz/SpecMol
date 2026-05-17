"""Shared training helpers for BACE/FreeSolv experiment scripts."""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ours_loss(f_l, f_h, f_a, f_fp, alpha=1, tau=0.5, eps=1e-8):
    """Contrastive pretraining loss with finite denominators for singleton batches."""
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    s_la = torch.exp(cos(f_l, f_a) / tau)
    s_lh = torch.exp(cos(f_l, f_h) / tau)
    s_ha = torch.exp(cos(f_h, f_a) / tau)
    s_fp_a = torch.exp(cos(f_fp, f_a) / tau)
    n = f_l.size(0)

    l_denom = (torch.sum(s_la) - s_la + s_lh).clamp_min(eps)
    h_denom = (torch.sum(s_ha) - s_ha + s_lh).clamp_min(eps)
    fp_a_denom = (torch.sum(s_fp_a) - s_fp_a).clamp_min(eps)

    loss_l_1 = torch.log(s_la / l_denom)
    loss_h_1 = torch.log(s_ha / h_denom)
    loss_fpa = torch.log(s_fp_a / fp_a_denom)
    loss = torch.sum(loss_l_1) + torch.sum(loss_h_1) + alpha * torch.sum(loss_fpa)
    return -loss / n


def calculate_auc(y_true, y_score, n_task):
    auc_list = []
    for i in range(n_task):
        sub_y = y_true[i::n_task]
        sub_score = y_score[i::n_task]
        mask = sub_y != 999
        sub_y = sub_y[mask]
        sub_score = sub_score[mask]
        if len(sub_y) > 0 and np.unique(sub_y).size > 1:
            auc_list.append(roc_auc_score(sub_y, sub_score))
    return float(np.mean(auc_list)) if auc_list else 0.0


def load_model_state_dict(path, map_location=None):
    """Load a model state dict with weights_only when the installed torch supports it."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_pyg_inmemory_split(root, task, split):
    """Load a saved PyG InMemoryDataset split without creating temp directories."""
    from torch_geometric.data import InMemoryDataset

    pt_path = os.path.join(root, "processed", f"{task}_{split}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing: {pt_path}")

    class _PtDataset(InMemoryDataset):
        def __init__(self, dataset_root, processed_name):
            self._processed_name = processed_name
            super().__init__(root=dataset_root)
            path = os.path.join(self.processed_dir, self._processed_name)
            try:
                self.data, self.slices = torch.load(path, weights_only=False)
            except TypeError:
                self.data, self.slices = torch.load(path)

        @property
        def processed_file_names(self):
            return [self._processed_name]

        def process(self):
            raise FileNotFoundError(f"Missing pre-built PyG split: {pt_path}")

    return _PtDataset(root, os.path.basename(pt_path))


def classification_probe_batch(batch, model, logreg, device, n_task, return_shaped=False):
    """Return logits/labels for a frozen encoder classification probe.

    return_shaped=False (default): legacy API, returns 1-D (logits[mask], y[mask])
    over valid (non-999) entries. Preserves the existing eval call sites.

    return_shaped=True: returns (logits, y, mask) all shaped [B, n_task]; used
    by the per-task-weighted training path so it can balance tasks.
    """
    with torch.no_grad():
        _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
    embed = torch.cat([spec_x.detach(), x_fp.detach()], dim=1)
    logits = logreg(embed)
    y = y.reshape(-1, n_task)
    mask = y != 999
    if return_shaped:
        return logits, y, mask
    return logits[mask], y[mask]


def train_classification_probe_epoch(loader, model, logreg, optimizer, loss_fn, device, n_task):
    """One mini-batch SGD epoch for frozen classification linear eval.

    Uses per-task weighting: for each task t, average BCE over its valid entries,
    then average across tasks. For n_task=1 this is mathematically identical to
    the legacy `BCEWithLogitsLoss(reduction='mean')` over all valid entries, so
    BACE/BBBP behavior is unchanged. For multi-label tasks (Tox21 n_task=12 with
    ~30% missing labels per task; ClinTox n_task=2) it gives each task equal
    gradient weight regardless of label-validity count, which the legacy uniform
    mean did not do.

    The `loss_fn` argument is kept in the signature for API compatibility but is
    ignored — BCEWithLogitsLoss(reduction='none') is built internally so we can
    apply the per-task aggregation directly. (Removing the arg would force every
    caller in main_pretrain.py to update.)
    """
    bce_none = torch.nn.BCEWithLogitsLoss(reduction='none')
    logreg.train()
    model.eval()
    for batch in loader:
        optimizer.zero_grad()
        logits, y, mask = classification_probe_batch(
            batch, model, logreg, device, n_task, return_shaped=True
        )
        if mask.sum() == 0:
            continue
        # Per-element loss on float labels; mask out missing (999) entries.
        per_elem = bce_none(logits, y.float()) * mask.float()
        # Per-task sum then divide by per-task valid count; tasks with no valid
        # samples in this batch contribute 0 (denominator clamped to 1 to avoid
        # NaN, then the corresponding mask sum gates them out of the mean).
        per_task_valid_count = mask.sum(dim=0).clamp(min=1).float()
        per_task_loss = per_elem.sum(dim=0) / per_task_valid_count
        # Mean across tasks that actually had at least one valid sample.
        task_has_signal = mask.sum(dim=0) > 0
        if task_has_signal.sum() == 0:
            continue
        loss = per_task_loss[task_has_signal].mean()
        loss.backward()
        optimizer.step()


def regression_probe_batch(batch, model, logreg, device):
    """Return predictions/targets for a frozen-encoder regression probe.

    Mirrors classification_probe_batch but without n_task masking (regression
    targets are dense, no 999 missing-label sentinel).
    """
    with torch.no_grad():
        _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
    embed = torch.cat([spec_x.detach(), x_fp.detach()], dim=1)
    preds = logreg(embed)
    y = y.reshape(-1, 1).float()
    return preds, y


def train_regression_probe_epoch(loader, model, logreg, optimizer, loss_fn, device):
    """One mini-batch SGD epoch for frozen regression linear eval (MSE)."""
    logreg.train()
    model.eval()
    for batch in loader:
        optimizer.zero_grad()
        preds, y = regression_probe_batch(batch, model, logreg, device)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()


def calculate_rmse(y_true, y_pred):
    """Root mean squared error. Inputs accept numpy arrays or torch tensors."""
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    return float(np.sqrt(np.mean((y_true.reshape(-1) - y_pred.reshape(-1)) ** 2)))


def calculate_mae(y_true, y_pred):
    """Mean absolute error. Inputs accept numpy arrays or torch tensors."""
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    return float(np.mean(np.abs(y_true.reshape(-1) - y_pred.reshape(-1))))
