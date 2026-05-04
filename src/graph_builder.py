"""
Graph construction from tabular network traffic.

Each CSV row represents a directed communication edge (src_ip -> dst_ip).
NetworkX is used as an interpretable intermediate representation; the graph is
then converted to a PyTorch Geometric `Data` object for GCN training.

Node supervision is derived from edge-level labels: a node is labeled as
attack if it participates in at least one malicious flow. This matches the
intuition that coordinated activity compromises or involves multiple endpoints
in the infrastructure graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data_loader import DataLoaderConfig
from src.utils import train_val_test_split_masks


@dataclass(frozen=True)
class GraphBuildResult:
    """Container for the PyG graph and bookkeeping maps for analysis/debugging."""

    data: Data
    node_to_id: Dict[str, int]
    id_to_node: Dict[int, str]
    nx_graph: nx.DiGraph


class TrafficGraphBuilder:
    """
    Builds a directed traffic graph and exports PyTorch Geometric tensors.

    Why this is used:
    Separating graph construction from model code lets you experiment with
    alternative graph definitions (weighted edges, temporal windows) without
    changing the GCN implementation.
    """

    def __init__(self, config: Optional[DataLoaderConfig] = None) -> None:
        self.config = config or DataLoaderConfig()

    def build_networkx(
        self,
        dataframe: pd.DataFrame,
        feature_columns: List[str],
        edge_labels: pd.Series,
    ) -> nx.DiGraph:
        """
        Construct a directed graph where aggregated edges carry mean features
        and a maliciousness flag (max over merged rows).

        Why this is used:
        Duplicate (src, dst) pairs are common in flow logs; aggregation reduces
        noise and avoids multigraph complexity while preserving strong malicious
        signals via the max label rule.
        """

        if len(dataframe) != len(edge_labels):
            raise ValueError("edge_labels must align row-wise with dataframe.")

        src = self.config.src_column
        dst = self.config.dst_column
        work = dataframe[[src, dst]].copy()
        work["_edge_label"] = edge_labels.astype(int).values
        for col in feature_columns:
            work[col] = dataframe[col]

        agg: Dict[str, str] = {c: "mean" for c in feature_columns}
        agg["_edge_label"] = "max"
        grouped = work.groupby([src, dst], as_index=False).agg(agg)

        graph = nx.DiGraph()
        for row in grouped.itertuples(index=False):
            u = str(getattr(row, src))
            v = str(getattr(row, dst))
            label = int(getattr(row, "_edge_label"))
            edge_attrs = {"label": label}
            for col in feature_columns:
                edge_attrs[col] = float(getattr(row, col))
            if graph.has_edge(u, v):
                # Extremely unlikely after groupby; kept for safety.
                prev = graph.edges[u, v]
                prev["label"] = max(int(prev["label"]), label)
                for col in feature_columns:
                    prev[col] = float(prev[col])
            else:
                graph.add_edge(u, v, **edge_attrs)

        return graph

    def _node_label_from_edges(self, graph: nx.DiGraph) -> Dict[str, int]:
        """
        Assign each node label = 1 if any incident edge is malicious.

        Why this is used:
        Flow-level labels become node-level supervision so a standard GCN can
        classify endpoints involved in attack traffic—useful for lateral
        movement and bot-like coordination patterns in a cloud graph.
        """

        labels: Dict[str, int] = {node: 0 for node in graph.nodes}
        for u, v, attrs in graph.edges(data=True):
            malicious = int(attrs.get("label", 0))
            if malicious:
                labels[u] = 1
                labels[v] = 1
        return labels

    def _node_feature_matrix(
        self,
        graph: nx.DiGraph,
        feature_columns: List[str],
        node_ids: List[str],
    ) -> np.ndarray:
        """
        Build node features: degrees plus optional traffic statistics.

        Why this is used:
        Degree captures structural connectivity (fan-in/fan-out), while means of
        edge attributes summarize typical traffic behavior incident to a host.
        """

        feats: List[List[float]] = []
        for node in node_ids:
            indeg = float(graph.in_degree(node))
            outdeg = float(graph.out_degree(node))
            if not feature_columns:
                feats.append([indeg, outdeg])
                continue

            in_vals = [[] for _ in feature_columns]
            out_vals = [[] for _ in feature_columns]
            for _, _, attrs in graph.in_edges(node, data=True):
                for i, col in enumerate(feature_columns):
                    in_vals[i].append(float(attrs.get(col, 0.0)))
            for _, _, attrs in graph.out_edges(node, data=True):
                for i, col in enumerate(feature_columns):
                    out_vals[i].append(float(attrs.get(col, 0.0)))

            row: List[float] = [indeg, outdeg]
            for i in range(len(feature_columns)):
                in_mean = float(np.mean(in_vals[i])) if in_vals[i] else 0.0
                out_mean = float(np.mean(out_vals[i])) if out_vals[i] else 0.0
                row.extend([in_mean, out_mean])
            feats.append(row)

        return np.asarray(feats, dtype=np.float32)

    def to_pytorch_geometric(
        self,
        graph: nx.DiGraph,
        feature_columns: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ) -> GraphBuildResult:
        """
        Convert a NetworkX graph into a PyG `Data` object for GCN training.

        Why this is used:
        PyTorch Geometric expects sparse edge indices and contiguous node ids;
        this function performs that mapping and attaches train/val/test masks.
        """

        nodes = sorted(graph.nodes)
        node_to_id = {n: i for i, n in enumerate(nodes)}
        id_to_node = {i: n for n, i in node_to_id.items()}

        edge_pairs: List[Tuple[int, int]] = []
        for u, v in graph.edges:
            i, j = node_to_id[str(u)], node_to_id[str(v)]
            edge_pairs.append((i, j))
            edge_pairs.append((j, i))

        if not edge_pairs:
            raise ValueError("Graph has no edges; cannot train a GCN.")

        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        x_np = self._node_feature_matrix(graph, feature_columns, [str(n) for n in nodes])
        x = torch.tensor(x_np, dtype=torch.float32)

        node_labels_dict = self._node_label_from_edges(graph)
        y = torch.tensor([node_labels_dict[str(n)] for n in nodes], dtype=torch.long)

        train_mask, val_mask, test_mask = train_val_test_split_masks(
            y, train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed
        )

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return GraphBuildResult(
            data=data,
            node_to_id=node_to_id,
            id_to_node=id_to_node,
            nx_graph=graph,
        )

    def build_pyg_data(
        self,
        dataframe: pd.DataFrame,
        feature_columns: List[str],
        edge_labels: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ) -> GraphBuildResult:
        """
        End-to-end helper used by `main.py` to go from CSV rows to PyG `Data`.

        Why this is used:
        A single call keeps the orchestration code short and thesis-friendly.
        """

        nx_graph = self.build_networkx(dataframe, feature_columns, edge_labels)
        return self.to_pytorch_geometric(
            nx_graph,
            feature_columns=feature_columns,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )


class FlowKnnGraphBuilder:
    """
    Build a graph where each row (flow) is a node and edges link k nearest
    neighbors in standardized feature space.

    Why this is used:
    Flow exports without IP endpoints cannot form a host communication graph.
    A kNN graph is a standard way to obtain a relational structure from tabular
    data so GCN layers can combine each flow with locally similar flows.
    """

    def __init__(self, k_neighbors: int = 15, random_state: int = 42) -> None:
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def build_pyg_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ) -> Data:
        """
        Construct a PyG `Data` object from a feature matrix and flow labels.

        Why this is used:
        This path reuses the same GCN training code as the host graph pipeline
        while matching datasets that only provide per-flow statistics.
        """

        try:
            from sklearn.neighbors import kneighbors_graph
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - env-specific
            raise ImportError(
                "Flow graph mode requires scikit-learn. Install with: pip install scikit-learn"
            ) from exc

        if features.ndim != 2:
            raise ValueError("features must be a 2D array (n_samples, n_features).")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must be 1D with length equal to number of rows.")

        n = features.shape[0]
        k = min(self.k_neighbors, n - 1)
        if k < 1:
            raise ValueError("Need at least 2 samples to build a kNN graph.")

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(features)

        adj = kneighbors_graph(
            x_scaled,
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
        )
        adj = adj.maximum(adj.T)
        coo = adj.tocoo()
        edge_index = torch.stack(
            [
                torch.from_numpy(coo.row.astype(np.int64)),
                torch.from_numpy(coo.col.astype(np.int64)),
            ],
            dim=0,
        )

        x = torch.tensor(x_scaled, dtype=torch.float32)
        y = torch.tensor(labels.astype(np.int64), dtype=torch.long)

        train_mask, val_mask, test_mask = train_val_test_split_masks(
            y, train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed
        )

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data
