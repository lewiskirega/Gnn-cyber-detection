"""
Advanced GNN Architecture Module for Coordinated Cloud Attack Detection.

Implements upgraded Graph Neural Networks:
1. Multi-head `GATv2Conv` (Dynamic Attention Graph Attention Networks v2).
2. Residual `GraphSAGE` with Jumping Knowledge (`JK="cat"`).
3. Layer Normalization (`nn.LayerNorm`), residual skip connections, and tuned dropout (0.15–0.25)
   to prevent oversmoothing while maximizing feature discrimination.
"""

from __future__ import annotations

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, JumpingKnowledge


class AdvancedGNNClassifier(nn.Module):
    """
    Optimized multi-head GATv2 / Residual JK-GraphSAGE Classifier for node-level attack detection.
    
    Architectural highlights:
    - Multi-head GATv2Conv layer with dynamic attention.
    - Jumping Knowledge (JK='cat') aggregating representations across multi-layer hops.
    - Layer Normalization on each GNN block.
    - Residual skip connections.
    - Tuned dropout rate (0.15 - 0.25).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.20,
        arch_type: str = "GATv2",  # 'GATv2' or 'GraphSAGE'
        use_jk: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.arch_type = arch_type
        self.use_jk = use_jk

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dropout),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        current_dim = hidden_channels

        for i in range(num_layers):
            if arch_type == "GATv2":
                # Multi-head GATv2 with dynamic attention
                conv = GATv2Conv(
                    in_channels=current_dim,
                    out_channels=hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=1,  # Edge weights support
                )
                next_dim = hidden_channels * heads
            elif arch_type == "GraphSAGE":
                conv = SAGEConv(
                    in_channels=current_dim,
                    out_channels=hidden_channels,
                    aggr="mean",
                )
                next_dim = hidden_channels
            else:
                raise ValueError(f"Unsupported architecture type: {arch_type}")

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(next_dim))
            self.residuals.append(
                nn.Linear(current_dim, next_dim) if current_dim != next_dim else nn.Identity()
            )
            current_dim = next_dim

        if use_jk:
            # Jumping Knowledge concatenation across layers
            self.jk = JumpingKnowledge(mode="cat")
            jk_dim = hidden_channels * num_layers if arch_type == "GraphSAGE" else (hidden_channels * heads) * num_layers
        else:
            self.jk = None
            jk_dim = current_dim

        # Final prediction head with LayerNorm & tuned dropout
        self.classifier = nn.Sequential(
            nn.Linear(jk_dim, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels * 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute node classification logits.
        """
        h = self.input_proj(x)
        layer_representations: List[torch.Tensor] = []

        if edge_weight is not None and edge_weight.dim() == 1:
            edge_attr = edge_weight.unsqueeze(-1)
        else:
            edge_attr = edge_weight

        for conv, norm, res in zip(self.convs, self.norms, self.residuals):
            h_res = res(h)
            if self.arch_type == "GATv2":
                h_conv = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h_conv = conv(h, edge_index)

            # Residual skip connection + LayerNorm + Activation + Dropout
            h = norm(h_conv + h_res)
            h = F.leaky_relu(h, negative_slope=0.2)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_representations.append(h)

        if self.use_jk and self.jk is not None:
            h_final = self.jk(layer_representations)
        else:
            h_final = layer_representations[-1]

        logits = self.classifier(h_final)
        return logits


class GATv2Classifier(AdvancedGNNClassifier):
    """Convenience subclass for dynamic multi-head GATv2 Conv model."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.20,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=3,
            heads=heads,
            dropout=dropout,
            arch_type="GATv2",
            use_jk=True,
        )


class SAGEJKClassifier(AdvancedGNNClassifier):
    """Convenience subclass for Residual GraphSAGE with Jumping Knowledge model."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=3,
            heads=1,
            dropout=dropout,
            arch_type="GraphSAGE",
            use_jk=True,
        )


if __name__ == "__main__":
    print("Testing Advanced GNN Architectures...")
    x = torch.randn(100, 21)
    edge_index = torch.randint(0, 100, (2, 400))
    edge_weight = torch.rand(400)

    gat_model = GATv2Classifier(in_channels=21, hidden_channels=32, num_classes=2)
    gat_logits = gat_model(x, edge_index, edge_weight)
    print(f"GATv2Classifier output shape: {gat_logits.shape}")

    sage_model = SAGEJKClassifier(in_channels=21, hidden_channels=32, num_classes=2)
    sage_logits = sage_model(x, edge_index, edge_weight)
    print(f"SAGEJKClassifier output shape: {sage_logits.shape}")
