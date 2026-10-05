"""Shared training helpers for BACE/FreeSolv experiment scripts."""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


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


def calculate_pr_auc(y_true, y_score, n_task):
    """Macro PR-AUC (average precision), mirroring calculate_auc's masking.

    Row-major n_task stride, drop 999-missing entries, skip single-class tasks.
    average_precision_score is rank-based so raw logits are valid inputs (same as
    roc_auc_score). Reported ALONGSIDE ROC-AUC for a fair head-to-head against the
    RF baseline (which already emits average_precision_score) on imbalanced sets
    (BACE positive-rate, ClinTox CT_TOX rare task) where ROC-AUC is optimistic.
    """
    ap_list = []
    for i in range(n_task):
        sub_y = y_true[i::n_task]
        sub_score = y_score[i::n_task]
        mask = sub_y != 999
        sub_y = sub_y[mask]
        sub_score = sub_score[mask]
        if len(sub_y) > 0 and np.unique(sub_y).size > 1:
            ap_list.append(average_precision_score(sub_y, sub_score))
    return float(np.mean(ap_list)) if ap_list else 0.0


def compute_pos_weight(loader, n_task, clamp_min=0.1, clamp_max=10.0):
    """Per-task pos_weight = n_neg / n_pos over the training labels (999 ignored),
    matching the RF baseline's class_weight='balanced'. Returns a [n_task] tensor,
    clamped to [clamp_min, clamp_max] so extreme imbalance (e.g. ClinTox CT_TOX)
    does not explode the gradient. Tasks with no positives default to weight 1.0.
    Counting only -- safe to run on CPU before the epoch loop.
    """
    pos = torch.zeros(n_task)
    neg = torch.zeros(n_task)
    for batch in loader:
        y = batch.y.reshape(-1, n_task).float()
        valid = y != 999
        pos += ((y == 1) & valid).sum(dim=0).float()
        neg += ((y == 0) & valid).sum(dim=0).float()
    pw = torch.ones(n_task)
    has_pos = pos > 0
    pw[has_pos] = (neg[has_pos] / pos[has_pos]).clamp(min=clamp_min, max=clamp_max)
    return pw


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


def classification_probe_batch(batch, model, logreg, device, n_task, return_shaped=False, fp_only=False, spec_only=False):
    """Return logits/labels for a frozen encoder classification probe.

    return_shaped=False (default): legacy API, returns 1-D (logits[mask], y[mask])
    over valid (non-999) entries. Preserves the existing eval call sites.

    return_shaped=True: returns (logits, y, mask) all shaped [B, n_task]; used
    by the per-task-weighted training path so it can balance tasks.

    fp_only=True: zero out the spec_x (GNN encoder) half of the embed before
    feeding LogReg. Keeps the LogReg input dim at hid_dim*2 so checkpoint
    shapes are unchanged; the LogReg's left-half weights see zero input and
    contribute zero to the logit. Used for the BBBP / BACE fingerprint-only
    control (audit reviewer attack #1).
    """
    with torch.no_grad():
        _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
    spec_part = torch.zeros_like(spec_x) if fp_only else spec_x.detach()
    fp_part = torch.zeros_like(x_fp) if spec_only else x_fp.detach()
    embed = torch.cat([spec_part, fp_part], dim=1)
    logits = logreg(embed)
    y = y.reshape(-1, n_task)
    mask = y != 999
    if return_shaped:
        return logits, y, mask
    return logits[mask], y[mask]


def train_classification_probe_epoch(loader, model, logreg, optimizer, loss_fn, device, n_task, fp_only=False, spec_only=False, pos_weight=None):
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

    pos_weight (optional [n_task] tensor): per-task positive-class weight
    (n_neg/n_pos) for class-imbalanced sets, matching RF's class_weight='balanced'
    (baselines_ml.py). DEFAULT None => behavior is byte-identical to the legacy
    path; only the --class_weight flag activates it. Threaded into
    BCEWithLogitsLoss so it scales the positive term per task before the existing
    per-task mean aggregation.
    """
    pw = pos_weight.to(device) if pos_weight is not None else None
    bce_none = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pw)
    logreg.train()
    model.eval()
    for batch in loader:
        optimizer.zero_grad()
        logits, y, mask = classification_probe_batch(
            batch, model, logreg, device, n_task, return_shaped=True, fp_only=fp_only, spec_only=spec_only
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


def regression_probe_batch(batch, model, logreg, device, n_task=1):
    """Return predictions/targets for a frozen-encoder regression probe.

    Mirrors classification_probe_batch but without n_task masking (regression
    targets are dense, no 999 missing-label sentinel). n_task>1 supports
    multi-target regression (QM8/QM9: 12 properties per molecule).
    """
    with torch.no_grad():
        _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
    embed = torch.cat([spec_x.detach(), x_fp.detach()], dim=1)
    preds = logreg(embed)
    y = y.reshape(-1, n_task).float()
    return preds, y


def train_regression_probe_epoch(loader, model, logreg, optimizer, loss_fn, device, n_task=1):
    """One mini-batch SGD epoch for frozen regression linear eval (MSE)."""
    logreg.train()
    model.eval()
    for batch in loader:
        optimizer.zero_grad()
        preds, y = regression_probe_batch(batch, model, logreg, device, n_task=n_task)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()


# ----------------------------------------------------------------------------
# Lever A: end-to-end fine-tuning helpers.
#
# These are exact twins of the frozen-probe functions above, with the two
# gradient barriers removed: (1) no `with torch.no_grad()` around the encoder
# forward, (2) no `.detach()` on the embeddings, and (3) `model.train()` instead
# of `model.eval()` so dropout/BN are active. The frozen-probe functions above
# are left byte-identical, so behavior with `--finetune` absent is unchanged.
# get_embedding (model_gnn_pre_v2.py) has no internal no_grad/detach, so gradient
# flows into the encoder (and the T7 attention gate) through these.
# ----------------------------------------------------------------------------

def finetune_classification_batch(batch, model, logreg, device, n_task, return_shaped=False, fp_only=False, spec_only=False):
    """Classification batch with gradient flowing into the encoder (fine-tune)."""
    _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
    spec_part = torch.zeros_like(spec_x) if fp_only else spec_x
    fp_part = torch.zeros_like(x_fp) if spec_only else x_fp
    embed = torch.cat([spec_part, fp_part], dim=1)
    logits = logreg(embed)
    y = y.reshape(-1, n_task)
    mask = y != 999
    if return_shaped:
        return logits, y, mask
    return logits[mask], y[mask]


def train_finetune_classification_epoch(loader, model, logreg, optimizer, loss_fn, device, n_task, fp_only=False, spec_only=False, pos_weight=None):
    """One joint encoder+head fine-tune epoch (classification).

    Same per-task weighted BCE as train_classification_probe_epoch; the only
    differences are model.train() and gradient reaching the encoder. pos_weight
    (optional [n_task] tensor) is the same opt-in class-balancing knob; None keeps
    the legacy behavior byte-identical.
    """
    pw = pos_weight.to(device) if pos_weight is not None else None
    bce_none = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pw)
    logreg.train()
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        logits, y, mask = finetune_classification_batch(
            batch, model, logreg, device, n_task, return_shaped=True, fp_only=fp_only, spec_only=spec_only
        )
        if mask.sum() == 0:
            continue
        per_elem = bce_none(logits, y.float()) * mask.float()
        per_task_valid_count = mask.sum(dim=0).clamp(min=1).float()
        per_task_loss = per_elem.sum(dim=0) / per_task_valid_count
        task_has_signal = mask.sum(dim=0) > 0
        if task_has_signal.sum() == 0:
            continue
        loss = per_task_loss[task_has_signal].mean()
        loss.backward()
        optimizer.step()


def finetune_regression_batch(batch, model, logreg, device, n_task=1):
    """Regression batch with gradient flowing into the encoder (fine-tune)."""
    _, _, spec_x, x_fp, y = model.get_embedding(batch, device)
    embed = torch.cat([spec_x, x_fp], dim=1)
    preds = logreg(embed)
    y = y.reshape(-1, n_task).float()
    return preds, y


def train_finetune_regression_epoch(loader, model, logreg, optimizer, loss_fn, device, n_task=1):
    """One joint encoder+head fine-tune epoch (regression, MSE)."""
    logreg.train()
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        preds, y = finetune_regression_batch(batch, model, logreg, device, n_task=n_task)
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
