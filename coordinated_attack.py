"""
Coordinated Cloud Attack Simulation & Topological Feature Engineering Module.

This module provides:
1. Coordinated multi-source cloud attack traffic generator & graph injection routines.
2. High-precision topological feature extraction:
   - Bidirectional flow statistics (in-degree, out-degree, flow volume ratios).
   - Graph centrality metrics (degree centrality, clustering coefficient, PageRank).
   - Packet burstiness & temporal inter-arrival statistics.
   - Dynamic edge weighting to expose synchronized, coordinated attack clusters.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.utils import train_val_test_split_masks


class CoordinatedAttackSimulator:
    """
    Simulates coordinated multi-source cloud attack patterns (e.g., botnet DDoS,
    distributed microservice flooding, synchronized burst attacks) and injects them
    into graph topologies.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.rng = np.random.RandomState(random_state)

    def generate_coordinated_attack_graph(
        self,
        num_benign_nodes: int = 800,
        num_attacker_nodes: int = 150,
        num_victim_nodes: int = 50,
        base_feature_dim: int = 20,
    ) -> Tuple[nx.DiGraph, Dict[str, int]]:
        """
        Synthesize a realistic cloud infrastructure graph with injected coordinated attack clusters.
        
        Benign traffic exhibits localized, lower-burst patterns across microservices.
        Coordinated attack nodes show synchronized high-frequency bursts targeting victim endpoints.
        """
        graph: nx.DiGraph = nx.DiGraph()
        node_labels: Dict[str, int] = {}

        benign_ids = [f"node_benign_{i}" for i in range(num_benign_nodes)]
        attacker_ids = [f"node_attacker_{i}" for i in range(num_attacker_nodes)]
        victim_ids = [f"node_victim_{i}" for i in range(num_victim_nodes)]

        for n in benign_ids:
            graph.add_node(n)
            node_labels[n] = 0

        for n in attacker_ids:
            graph.add_node(n)
            node_labels[n] = 1

        for n in victim_ids:
            graph.add_node(n)
            node_labels[n] = 1  # Victim nodes involved in attack flows

        # 1. Connect benign background network (mesh / hub-and-spoke cloud services)
        for u in benign_ids:
            # Connect to 3-6 other benign nodes
            k = int(self.rng.randint(3, 7))
            targets = self.rng.choice(benign_ids, size=k, replace=False)
            for v in targets:
                v_str = str(v)
                if u != v_str:
                    pkt_count = int(self.rng.randint(5, 50))
                    pkt_rate = float(self.rng.uniform(10.0, 100.0))
                    duration = float(self.rng.uniform(0.1, 5.0))
                    bytes_transferred = float(pkt_count * self.rng.randint(64, 1500))
                    burstiness = float(self.rng.uniform(0.01, 0.25))
                    synced_burst_score = float(self.rng.uniform(0.0, 0.1))
                    
                    graph.add_edge(
                        u,
                        v_str,
                        label=0,
                        pkt_count=float(pkt_count),
                        pkt_rate=pkt_rate,
                        duration=duration,
                        bytes_transferred=bytes_transferred,
                        burstiness=burstiness,
                        synced_burst_score=synced_burst_score,
                    )

        # 2. Inject Coordinated Multi-Source Attack Traffic
        # Coordinated attackers synchronize packet bursts targeting victim cloud microservices
        burst_phase = float(self.rng.uniform(0.8, 1.0))  # High synchronization coefficient
        for attacker in attacker_ids:
            # Each attacker targets multiple victim nodes in synchronized bursts
            num_targets = int(self.rng.randint(2, max(3, num_victim_nodes // 2)))
            targeted_victims = self.rng.choice(victim_ids, size=num_targets, replace=False)
            
            for victim in targeted_victims:
                victim_str = str(victim)
                pkt_count = int(self.rng.randint(500, 5000))  # High volume attack
                pkt_rate = float(self.rng.uniform(1000.0, 10000.0))
                duration = float(self.rng.uniform(0.01, 0.5))  # Sudden intense burst
                bytes_transferred = float(pkt_count * self.rng.randint(1000, 1500))
                burstiness = float(self.rng.uniform(0.75, 1.0))  # High burstiness
                synced_score = float(burst_phase * self.rng.uniform(0.85, 1.0))

                graph.add_edge(
                    attacker,
                    victim_str,
                    label=1,
                    pkt_count=float(pkt_count),
                    pkt_rate=pkt_rate,
                    duration=duration,
                    bytes_transferred=bytes_transferred,
                    burstiness=burstiness,
                    synced_burst_score=synced_score,
                )

        return graph, node_labels


class TopologicalGraphEngineer:
    """
    High-precision Graph & Feature Engineering Pipeline.
    Enriches graph nodes with:
    - Bidirectional flow statistics (in-degree, out-degree, flow ratios).
    - Degree centrality, clustering coefficient, PageRank.
    - Temporal packet burstiness.
    - Dynamic edge weighting matrix.
    """

    def __init__(self, use_centrality: bool = True) -> None:
        self.use_centrality = use_centrality

    def compute_dynamic_edge_weights(self, graph: nx.DiGraph) -> Dict[Tuple[str, str], float]:
        """
        Compute dynamic edge weights based on packet rate, burstiness, and synchronized attack score.
        High edge weights expose coordinated cloud attack clusters to attention aggregation layers.
        """
        edge_weights: Dict[Tuple[str, str], float] = {}
        for u, v in graph.edges:
            u_str, v_str = str(u), str(v)
            data = graph.get_edge_data(u_str, v_str) or {}
            pkt_rate = float(data.get("pkt_rate", 1.0))
            burstiness = float(data.get("burstiness", 0.1))
            synced_score = float(data.get("synced_burst_score", 0.0))
            
            # Dynamic weighting formula highlighting synchronized burst communication
            w = 1.0 + float(np.log1p(pkt_rate)) * 0.1 + (burstiness * 2.0) + (synced_score * 3.0)
            edge_weights[(u_str, v_str)] = float(w)

        return edge_weights

    def extract_topological_node_features(
        self,
        graph: nx.DiGraph,
        feature_keys: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract rich node features incorporating topological metrics & bidirectional flow stats.
        """
        if feature_keys is None:
            feature_keys = ["pkt_count", "pkt_rate", "duration", "bytes_transferred", "burstiness"]

        nodes: List[str] = [str(n) for n in sorted(graph.nodes)]
        
        # Pre-compute graph-wide topological metrics
        deg_cent_raw = nx.degree_centrality(graph)
        degree_cent: Dict[str, float] = {str(k): float(v) for k, v in deg_cent_raw.items()}

        try:
            clustering_raw = nx.clustering(graph.to_undirected())
            if isinstance(clustering_raw, dict):
                clustering: Dict[str, float] = {str(k): float(v) for k, v in clustering_raw.items()}
            else:
                clustering = {n: 0.0 for n in nodes}
        except Exception:
            clustering = {n: 0.0 for n in nodes}
            
        try:
            pagerank_raw = nx.pagerank(graph, max_iter=200)
            pagerank: Dict[str, float] = {str(k): float(v) for k, v in pagerank_raw.items()}
        except Exception:
            pagerank = {n: 1.0 / max(len(nodes), 1) for n in nodes}

        feature_names: List[str] = [
            "in_degree",
            "out_degree",
            "total_degree",
            "degree_centrality",
            "clustering_coef",
            "pagerank",
        ]
        for key in feature_keys:
            feature_names.extend([f"in_mean_{key}", f"out_mean_{key}", f"bi_ratio_{key}"])

        feature_matrix: List[List[float]] = []

        in_degree_dict: Dict[str, int] = dict(graph.in_degree())  # type: ignore
        out_degree_dict: Dict[str, int] = dict(graph.out_degree())  # type: ignore

        for node in nodes:
            in_deg = float(in_degree_dict.get(node, 0))
            out_deg = float(out_degree_dict.get(node, 0))
            tot_deg = in_deg + out_deg
            deg_c = float(degree_cent.get(node, 0.0))
            clust_c = float(clustering.get(node, 0.0))
            pr_c = float(pagerank.get(node, 0.0))

            row: List[float] = [in_deg, out_deg, tot_deg, deg_c, clust_c, pr_c]

            preds = [str(u) for u in graph.predecessors(node)]
            succs = [str(v) for v in graph.successors(node)]

            for key in feature_keys:
                in_vals = [float(graph[u][node].get(key, 0.0)) for u in preds]
                out_vals = [float(graph[node][v].get(key, 0.0)) for v in succs]

                in_mean = float(np.mean(in_vals)) if in_vals else 0.0
                out_mean = float(np.mean(out_vals)) if out_vals else 0.0
                
                # Flow volume ratio (bidirectional flow statistic)
                bi_ratio = out_mean / (in_mean + 1e-5) if (in_mean + out_mean) > 0 else 0.0

                row.extend([in_mean, out_mean, bi_ratio])

            feature_matrix.append(row)

        feats_np = np.array(feature_matrix, dtype=np.float32)
        # Standardize features
        mean = np.mean(feats_np, axis=0, keepdims=True)
        std = np.std(feats_np, axis=0, keepdims=True) + 1e-6
        feats_normalized = (feats_np - mean) / std

        return feats_normalized, feature_names

    def build_enriched_pyg_data(
        self,
        graph: nx.DiGraph,
        node_labels: Dict[str, int],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Data:
        """
        Build PyTorch Geometric `Data` object containing dynamic edge weights and enriched node features.
        """
        nodes = [str(n) for n in sorted(graph.nodes)]
        node_to_id = {n: i for i, n in enumerate(nodes)}

        x_np, feature_names = self.extract_topological_node_features(graph)
        edge_weights_dict = self.compute_dynamic_edge_weights(graph)

        edge_pairs: List[Tuple[int, int]] = []
        edge_weights: List[float] = []

        for u, v in graph.edges:
            u_str, v_str = str(u), str(v)
            src_id, dst_id = node_to_id[u_str], node_to_id[v_str]
            w = edge_weights_dict.get((u_str, v_str), 1.0)
            
            # Forward directed edge
            edge_pairs.append((src_id, dst_id))
            edge_weights.append(w)
            
            # Reverse edge for message passing symmetry
            edge_pairs.append((dst_id, src_id))
            edge_weights.append(w * 0.8)

        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)

        x = torch.tensor(x_np, dtype=torch.float32)
        y = torch.tensor([node_labels[n] for n in nodes], dtype=torch.long)

        train_mask, val_mask, test_mask = train_val_test_split_masks(
            y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight_tensor,
            y=y,
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.feature_names = feature_names

        return data


def generate_and_inject_coordinated_attack_data(
    num_benign: int = 1000,
    num_attacker: int = 200,
    num_victim: int = 50,
    seed: int = 42,
) -> Data:
    """
    Helper entrypoint to generate synthetic cloud traffic graph with injected coordinated attack clusters.
    """
    simulator = CoordinatedAttackSimulator(random_state=seed)
    graph, labels = simulator.generate_coordinated_attack_graph(
        num_benign_nodes=num_benign,
        num_attacker_nodes=num_attacker,
        num_victim_nodes=num_victim,
    )
    engineer = TopologicalGraphEngineer(use_centrality=True)
    return engineer.build_enriched_pyg_data(graph, labels, seed=seed)


if __name__ == "__main__":
    print("Testing coordinated attack simulation and topological feature engineering...")
    pyg_data = generate_and_inject_coordinated_attack_data()
    assert pyg_data.edge_index is not None
    assert pyg_data.y is not None
    assert pyg_data.x is not None
    assert isinstance(pyg_data.y, torch.Tensor)
    y_tensor = pyg_data.y
    benign_count = int((y_tensor == 0).sum().item())
    attack_count = int((y_tensor == 1).sum().item())
    print("Generated PyG Graph Data successfully:")
    print(f"  Nodes: {pyg_data.num_nodes}")
    print(f"  Edges: {pyg_data.edge_index.size(1)}")
    print(f"  Node Feature Dim: {pyg_data.x.size(1)}")
    print(f"  Class Distribution - Benign (0): {benign_count}, Attack (1): {attack_count}")
