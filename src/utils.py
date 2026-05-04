"""
Shared utilities for reproducibility, data splitting, and evaluation.

This module keeps generic helpers out of training and graph code so the
pipeline remains easy to describe in a thesis (clear separation of concerns).
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


def set_seed(seed: int) -> None:
    """
    Fix random seeds for repeatable experiments.

    Why this is used:
    Master's-level work should report reproducible results; seeding controls
    stochasticity from weight initialization, dropout, and sampling.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_val_test_split_masks(
    labels: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create boolean masks for nodes by stratified random split.

    Why this is used:
    GCN training in this project uses semi-supervised node classification masks
    (train/val/test on nodes). Stratification keeps class balance similar across
    splits when reporting accuracy.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    num_nodes = labels.numel()
    indices = torch.randperm(num_nodes, generator=generator)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    classes = labels.unique()
    for c in classes.tolist():
        class_idx = (labels == c).nonzero(as_tuple=False).view(-1)
        class_idx = class_idx[torch.randperm(class_idx.numel(), generator=generator)]
        n = class_idx.numel()
        n_train = max(1, int(train_ratio * n)) if n > 0 else 0
        n_val = max(0, int(val_ratio * n)) if n > 0 else 0
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train_idx = class_idx[:n_train]
        val_idx = class_idx[n_train : n_train + n_val]
        test_idx = class_idx[n_train + n_val :]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    # Ensure every node is assigned somewhere (rounding can leave gaps).
    unassigned = ~(train_mask | val_mask | test_mask)
    if unassigned.any():
        test_mask[unassigned] = True

    return train_mask, val_mask, test_mask


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Compute classification accuracy over masked nodes.

    Why this is used:
    Accuracy is a standard, interpretable metric for binary/multiclass attack
    detection in academic reporting.
    """

    if mask.sum() == 0:
        return 0.0
    preds = logits[mask].argmax(dim=1)
    correct = (preds == labels[mask]).sum().item()
    return correct / int(mask.sum().item())


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """
    Compute accuracy, precision, recall, and macro F1-score over masked nodes.
    
    Why this is used:
    Provides a comprehensive evaluation of model performance, especially important
    for imbalanced cyber security datasets where accuracy alone is misleading.
    """
    if mask.sum() == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    trues = labels[mask].cpu().numpy()
    
    correct = (preds == trues).sum()
    acc = correct / len(trues)
    
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="macro", zero_division=0)
    
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1)
    }

