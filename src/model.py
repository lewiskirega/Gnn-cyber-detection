"""
Graph Convolutional Network (GCN) for node-level attack classification.

A GCN layers relational structure: each node's representation aggregates
information from its neighbors, which is appropriate for detecting patterns
that only emerge when viewing coordinated interactions across a cloud graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCNClassifier(nn.Module):
    """
    Two-layer GCN classifier mapping node features to class logits.

    Why this is used:
    This is a standard, well-cited baseline for graph learning; it balances
    technical correctness with simplicity expected in an academic project.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute per-node logits for downstream softmax/cross-entropy training.

        Why this is used:
        Returning logits (not probabilities) matches `CrossEntropyLoss`, which
        applies log-softmax internally in a numerically stable way.
        """

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)
