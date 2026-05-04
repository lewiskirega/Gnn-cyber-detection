"""
Training and evaluation loop for the GCN node classifier.

The loop follows common semi-supervised node classification practice: optimize
loss on training nodes, tune using validation nodes, and report test accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data import Data

from src.model import GCNClassifier
from src.utils import accuracy, compute_metrics


@dataclass
class TrainConfig:
    """Hyperparameters for reproducible experiments."""

    epochs: int = 100
    learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    hidden_channels: int = 16
    dropout: float = 0.5


def train_model(
    model: GCNClassifier,
    data: Data,
    device: torch.device,
    config: TrainConfig,
    epoch_callback: Optional[Callable[[int, float, float, float], None]] = None,
) -> GCNClassifier:
    """
    Train the GCN with Adam and cross-entropy on labeled training nodes.

    Why this is used:
    This is the standard optimization setup for multiclass node classification
    in PyG-style pipelines and is straightforward to justify in a thesis.
    """

    model = model.to(device)
    data = data.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                logits_eval = model(data.x, data.edge_index)
                train_acc = accuracy(logits_eval, data.y, data.train_mask)
                val_acc = accuracy(logits_eval, data.y, data.val_mask)
            loss_v = float(loss.item())
            print(
                f"Epoch {epoch:04d} | loss={loss_v:.4f} "
                f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
            )
            if epoch_callback is not None:
                epoch_callback(epoch, loss_v, train_acc, val_acc)

    return model


@torch.no_grad()
def evaluate(
    model: GCNClassifier,
    data: Data,
    device: torch.device,
) -> Tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Compute test accuracy and return predictions/probabilities for inspection.

    Why this is used:
    Showing predictions supports qualitative analysis (e.g., which endpoints
    are flagged) alongside quantitative accuracy in the report.
    """

    model.eval()
    data = data.to(device)
    logits = model(data.x, data.edge_index)
    test_metrics = compute_metrics(logits, data.y, data.test_mask)
    probs = F.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    return test_metrics, preds, probs


def build_model_for_data(data: Data, config: TrainConfig) -> GCNClassifier:
    """
    Instantiate a GCN with input dimension inferred from node features.

    Why this is used:
    Avoids magic numbers in `main.py` and prevents shape mismatches when
    feature engineering changes.
    """

    in_channels = int(data.x.size(1))
    num_classes = int(data.y.max().item()) + 1
    if num_classes < 2:
        raise ValueError(
            "Node labels have only one class after graph construction. "
            "Add both benign and attack flows in the CSV (or check labels)."
        )
    return GCNClassifier(
        in_channels=in_channels,
        hidden_channels=config.hidden_channels,
        num_classes=num_classes,
        dropout=config.dropout,
    )


def run_training_pipeline(
    data: Data,
    config: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
    epoch_callback: Optional[Callable[[int, float, float, float], None]] = None,
) -> Tuple[GCNClassifier, dict[str, float], torch.Tensor, torch.Tensor]:
    """
    High-level API: train, then evaluate on the test mask.

    Why this is used:
    Keeps `main.py` minimal while preserving modular training code for reuse.
    """

    cfg = config or TrainConfig()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_data(data, cfg)
    model = train_model(model, data, dev, cfg, epoch_callback=epoch_callback)
    test_metrics, preds, probs = evaluate(model, data, dev)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
    return model, test_metrics, preds, probs
