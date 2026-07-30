"""Evaluation helpers for GNN classification reports and dissertation-quality figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _build_class_names(label_mapping: Dict[str, int], num_classes: int) -> List[str]:
    reverse_map = {value: key for key, value in label_mapping.items()}
    return [str(reverse_map.get(idx, idx)) for idx in range(num_classes)]


def _prepare_arrays(
    y_true: torch.Tensor,
    preds: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    y_true_np = y_true.detach().cpu().numpy().astype(int)
    preds_np = preds.detach().cpu().numpy().astype(int)
    probs_np = None
    if probs is not None:
        probs_np = probs.detach().cpu().numpy()
    return y_true_np, preds_np, probs_np


def _select_binary_scores(y_true_np: np.ndarray, probs_np: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if probs_np is None:
        return y_true_np, None
    if probs_np.ndim == 2 and probs_np.shape[1] >= 2:
        positive_id = 1 if 1 in np.unique(y_true_np) else int(np.unique(y_true_np)[1])
        if positive_id < probs_np.shape[1]:
            return (y_true_np == positive_id).astype(int), probs_np[:, positive_id]
        return (y_true_np == 1).astype(int), probs_np[:, 1]
    if probs_np.ndim == 1:
        return (y_true_np == 1).astype(int), probs_np
    return (y_true_np == 1).astype(int), probs_np[:, 0]


def _plot_confusion_matrix(
    y_true_np: np.ndarray,
    preds_np: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true_np, preds_np, labels=list(range(len(class_names))))
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title("Confusion Matrix", pad=10, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    _save_figure(fig, output_path)


def _plot_roc_curve(
    y_true_np: np.ndarray,
    probs_np: Optional[np.ndarray],
    output_path: Path,
) -> float:
    if probs_np is None:
        raise ValueError("Probability scores are required for ROC analysis.")
    binary_true, scores = _select_binary_scores(y_true_np, probs_np)
    if binary_true.size == 0 or scores is None:
        raise ValueError("Unable to compute ROC curve from the provided labels and scores.")
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
    fpr, tpr, _ = roc_curve(binary_true, scores)
    roc_auc = roc_auc_score(binary_true, scores)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", lw=1.1, label="Random Guess")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    _save_figure(fig, output_path)
    return float(roc_auc)


def _plot_precision_recall_curve(
    y_true_np: np.ndarray,
    probs_np: Optional[np.ndarray],
    output_path: Path,
) -> float:
    if probs_np is None:
        raise ValueError("Probability scores are required for precision-recall analysis.")
    binary_true, scores = _select_binary_scores(y_true_np, probs_np)
    if binary_true.size == 0 or scores is None:
        raise ValueError("Unable to compute precision-recall curve from the provided labels and scores.")
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
    precision, recall, _ = precision_recall_curve(binary_true, scores)
    average_precision = average_precision_score(binary_true, scores)
    ax.plot(recall, precision, color="#d62728", lw=2.2, label=f"Precision-Recall (AP = {average_precision:.3f})")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")
    _save_figure(fig, output_path)
    return float(average_precision)


def _plot_training_accuracy_curve(
    history: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    if not history:
        return
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    epochs = [row.get("epoch", idx) for idx, row in enumerate(history)]
    train_acc = [float(row.get("train_acc", 0.0)) for row in history]
    val_acc = [float(row.get("val_acc", np.nan)) for row in history]
    ax.plot(epochs, train_acc, color="#1f77b4", marker="o", linewidth=1.8, label="Training Accuracy")
    ax.plot(epochs, val_acc, color="#2ca02c", marker="s", linewidth=1.8, label="Validation Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    _save_figure(fig, output_path)


def _plot_training_loss_curve(
    history: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    if not history:
        return
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    epochs = [row.get("epoch", idx) for idx, row in enumerate(history)]
    train_loss = [float(row.get("train_loss", row.get("loss", 0.0))) for row in history]
    val_loss = [float(row.get("val_loss", np.nan)) for row in history]
    ax.plot(epochs, train_loss, color="#1f77b4", marker="o", linewidth=1.8, label="Training Loss")
    ax.plot(epochs, val_loss, color="#d62728", marker="s", linewidth=1.8, label="Validation Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    _save_figure(fig, output_path)


def generate_evaluation_artifacts(
    y_true: torch.Tensor,
    preds: torch.Tensor,
    probs: Optional[torch.Tensor],
    label_mapping: Dict[str, int],
    output_dir: Path,
    num_nodes: int,
    num_edges: int,
    num_features: int,
    training_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate figures, report text, and metrics for the trained classifier."""

    output_dir.mkdir(exist_ok=True, parents=True)
    y_true_np, preds_np, probs_np = _prepare_arrays(y_true, preds, probs)
    num_classes = int(len(label_mapping) if label_mapping else max(int(y_true_np.max()), 1) + 1)
    class_names = _build_class_names(label_mapping, num_classes)

    confusion_path = output_dir / "confusion_matrix.png"
    roc_path = output_dir / "roc_curve.png"
    pr_path = output_dir / "precision_recall_curve.png"
    accuracy_path = output_dir / "accuracy_curve.png"
    loss_path = output_dir / "loss_curve.png"
    report_path = output_dir / "classification_report.txt"
    metrics_path = output_dir / "metrics.json"

    _plot_confusion_matrix(y_true_np, preds_np, class_names, confusion_path)
    roc_auc = _plot_roc_curve(y_true_np, probs_np, roc_path)
    average_precision = _plot_precision_recall_curve(y_true_np, probs_np, pr_path)
    if training_history:
        _plot_training_accuracy_curve(training_history, accuracy_path)
        _plot_training_loss_curve(training_history, loss_path)

    report_text = classification_report(
        y_true_np,
        preds_np,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_path.write_text(report_text, encoding="utf-8")

    metrics = {
        "accuracy": float(np.mean(preds_np == y_true_np)),
        "precision": float(precision_score(y_true_np, preds_np, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true_np, preds_np, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true_np, preds_np, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "num_features": int(num_features),
        "num_classes": int(num_classes),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "classification_report": report_text,
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(confusion_path),
        "roc_curve_path": str(roc_path),
        "precision_recall_curve_path": str(pr_path),
        "accuracy_curve_path": str(accuracy_path),
        "loss_curve_path": str(loss_path),
    }
