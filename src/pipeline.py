"""
End-to-end experiment orchestration (data → graph → GCN → metrics).

Why this is used:
CLI (`main.py`) and the graphical UI share the same code path so results are
comparable and the thesis can describe one pipeline with two interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd
import torch
from torch_geometric.data import Data

from src.data_loader import DataLoaderConfig, TrafficDataLoader
from src.graph_builder import FlowKnnGraphBuilder, TrafficGraphBuilder
from src.train import TrainConfig, build_model_for_data, evaluate, train_model
from src.utils import set_seed


@dataclass
class ExperimentConfig:
    """User-facing knobs for one training run (CLI and GUI)."""

    data_path: Path
    graph_mode: Literal["auto", "hosts", "flows"] = "auto"
    label_column: Optional[str] = None
    max_samples: int = 25_000
    knn_k: int = 15
    epochs: int = 100
    learning_rate: float = 1e-2
    hidden_channels: int = 16
    dropout: float = 0.5
    seed: int = 42
    positive_class: Optional[str] = None
    noise_level_features: float = 0.0
    noise_level_edges: float = 0.0


@dataclass
class ExperimentResult:
    """Structured output for logging, UI tables, and thesis figures."""

    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    label_mapping: Dict[str, int]
    graph_mode_used: str
    num_nodes: int
    num_edges: int
    num_features: int
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    sample_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    device: str = "cpu"
    info_message: str = ""


def project_root() -> Path:
    """Directory containing `main.py` (parent of `src/`)."""

    return Path(__file__).resolve().parent.parent


def default_data_path() -> Path:
    """Same resolution rules as the CLI default."""

    data_dir = project_root() / "data"
    primary = data_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    if primary.is_file():
        return primary
    if data_dir.is_dir():
        csv_files = sorted(
            p for p in data_dir.glob("*.csv") if p.name.lower() != "traffic.csv"
        )
        if csv_files:
            return csv_files[0]
    return primary


def peek_has_host_ips(data_path: Path, src_name: str = "src_ip", dst_name: str = "dst_ip") -> bool:
    """Return True if the CSV header includes host endpoint columns."""

    header = pd.read_csv(data_path, nrows=0)
    header.columns = header.columns.str.strip()
    return src_name in header.columns and dst_name in header.columns


def run_experiment(
    config: ExperimentConfig,
    epoch_callback: Optional[Callable[[int, float, float, float], None]] = None,
) -> ExperimentResult:
    """
    Load data, build graph, train GCN, evaluate, and package results.

    Why this is used:
    Single entry point for reproducible experiments from code, CLI, or GUI.
    """

    set_seed(config.seed)
    data_path = Path(config.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    default_cfg = DataLoaderConfig()
    use_hosts = config.graph_mode == "hosts" or (
        config.graph_mode == "auto"
        and peek_has_host_ips(data_path, default_cfg.src_column, default_cfg.dst_column)
    )

    if config.graph_mode == "hosts" and not peek_has_host_ips(
        data_path, default_cfg.src_column, default_cfg.dst_column
    ):
        raise ValueError(
            "graph_mode='hosts' requires src_ip and dst_ip columns in the CSV."
        )

    positive_class = config.positive_class or ("attack" if use_hosts else "DDoS")
    history: List[Dict[str, Any]] = []

    def _on_epoch(epoch: int, loss: float, train_acc: float, val_acc: float) -> None:
        row = {
            "epoch": epoch,
            "loss": loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(epoch, loss, train_acc, val_acc)

    if use_hosts:
        loader = TrafficDataLoader()
        dataframe, feature_columns, encoded_labels, label_mapping = loader.prepare_dataset(
            data_path,
            positive_class_name=positive_class,
        )
        builder = TrafficGraphBuilder(loader.config)
        graph_bundle = builder.build_pyg_data(
            dataframe=dataframe,
            feature_columns=feature_columns,
            edge_labels=encoded_labels,
        )
        pyg_data = graph_bundle.data
        graph_mode_used = "hosts"
        info = f"Host graph: {int(pyg_data.num_nodes)} nodes, {pyg_data.edge_index.size(1)} directed edge entries."
    else:
        label_name = config.label_column or "Label"
        loader = TrafficDataLoader(DataLoaderConfig(label_column=label_name))
        x_matrix, y_vector, feature_columns, label_mapping = loader.prepare_flow_dataset(
            data_path,
            positive_class_name=positive_class,
            max_samples=config.max_samples,
            random_state=config.seed,
        )
        flow_builder = FlowKnnGraphBuilder(k_neighbors=config.knn_k, random_state=config.seed)
        pyg_data = flow_builder.build_pyg_data(
            x_matrix,
            y_vector,
            split_seed=config.seed,
        )
        graph_mode_used = "flows"
        info = (
            f"Flow kNN graph: {int(pyg_data.num_nodes)} nodes, "
            f"{len(feature_columns)} input features, k={config.knn_k}."
        )

    train_cfg = TrainConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
    )

    if config.noise_level_features > 0:
        noise = torch.randn_like(pyg_data.x) * config.noise_level_features
        pyg_data.x = pyg_data.x + noise
        
    if config.noise_level_edges > 0:
        from torch_geometric.utils import dropout_edge
        # PyG dropout_edge requires training=True to actually drop
        edge_index, _ = dropout_edge(pyg_data.edge_index, p=config.noise_level_edges, training=True)
        pyg_data.edge_index = edge_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_data(pyg_data, train_cfg)
    model = train_model(model, pyg_data, device, train_cfg, epoch_callback=_on_epoch)
    test_metrics, preds, probs = evaluate(model, pyg_data, device)

    # Save the trained model weights for real-world deployment simulation
    out_dir = project_root() / "outputs"
    out_dir.mkdir(exist_ok=True)
    model_save_path = out_dir / f"trained_gnn_{graph_mode_used}.pth"
    torch.save(model.state_dict(), model_save_path)
    
    info += f"\nModel weights saved to: {model_save_path.relative_to(project_root())}"

    test_idx = pyg_data.test_mask.nonzero(as_tuple=False).view(-1)[:20]
    rows = []
    rev = {v: k for k, v in label_mapping.items()}
    for i in test_idx.tolist():
        yi = int(pyg_data.y[i].item())
        pi = int(preds[i].item())
        rows.append(
            {
                "node_index": i,
                "true_class_id": yi,
                "true_label": rev.get(yi, str(yi)),
                "pred_class_id": pi,
                "pred_label": rev.get(pi, str(pi)),
                "confidence": float(probs[i, pi].item()),
            }
        )
    sample_df = pd.DataFrame(rows)

    return ExperimentResult(
        test_accuracy=test_metrics["accuracy"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        test_f1=test_metrics["f1"],
        label_mapping=label_mapping,
        graph_mode_used=graph_mode_used,
        num_nodes=int(pyg_data.num_nodes),
        num_edges=int(pyg_data.edge_index.size(1)),
        num_features=int(pyg_data.x.size(1)),
        training_history=history,
        sample_predictions=sample_df,
        device=str(device),
        info_message=info,
    )
