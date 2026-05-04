"""
Entry point: load dataset CSV, build graph, train GCN, report results.

This file orchestrates modules only; core logic lives under `src/` for clarity
and testability (appropriate for a Master's implementation chapter).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import ExperimentConfig, default_data_path, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GCN-based coordinated attack detection on a cloud traffic graph."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data_path()),
        help="Path to CSV (host graph: src_ip, dst_ip, label; flow: numeric features + Label).",
    )
    parser.add_argument(
        "--graph-mode",
        choices=["auto", "hosts", "flows"],
        default="auto",
        help="auto: use host graph if src_ip/dst_ip exist, else flow kNN graph.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column name after stripping spaces (e.g. Label for CIC-IDS).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25_000,
        help="Max flow rows when using flow graph (stratified subsample).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="Number of neighbors per node in flow kNN graph.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positive-class",
        type=str,
        default=None,
        help="Positive class name for binary labels (default: attack for hosts, DDoS for flows).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        print(
            f"Dataset not found: {data_path.resolve()}\n"
            "Add a CSV under data/, or pass an explicit path, e.g.\n"
            "  python main.py --data data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        )
        raise SystemExit(1)

    cfg = ExperimentConfig(
        data_path=data_path,
        graph_mode=args.graph_mode,
        label_column=args.label_col,
        max_samples=args.max_samples,
        knn_k=args.knn_k,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_channels=args.hidden,
        dropout=args.dropout,
        seed=args.seed,
        positive_class=args.positive_class,
    )

    result = run_experiment(cfg)

    print(result.info_message)
    print("Device:", result.device)
    print("Graph mode:", result.graph_mode_used)
    print("Nodes:", result.num_nodes, "| Edge index size:", result.num_edges, "| Features:", result.num_features)
    print("Label mapping:", result.label_mapping)
    print(f"Test accuracy: {result.test_accuracy:.4f}")
    print(f"Test precision: {result.test_precision:.4f}")
    print(f"Test recall: {result.test_recall:.4f}")
    print(f"Test F1-score: {result.test_f1:.4f}")
    print("\nSample test predictions:")
    print(result.sample_predictions.to_string(index=False))


if __name__ == "__main__":
    main()
