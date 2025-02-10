import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
import json
from dataclasses import dataclass


@dataclass
class TransactionNode:
    transaction_id: str
    customer_id: str
    merchant_id: str
    amount: float
    timestamp: datetime
    location: Tuple[float, float]
    risk_score: float


@dataclass
class GraphMetrics:
    density: float
    avg_clustering: float
    avg_degree: float
    components: int
    diameter: float
    avg_path_length: float
    centrality_scores: Dict[str, float]


class TransactionGraphBuilder:
    def __init__(self, config: Dict):
        self.config = config
        self.graph = nx.Graph()
        self.node_mapping = {}
        self.edge_weights = {}
        self.temporal_edges = {}
        self.risk_thresholds = {"high": 0.8, "medium": 0.5, "low": 0.2}

    def _create_node_id(self, transaction: Dict) -> str:
        return f"T_{transaction['transaction_id']}"

    def _calculate_edge_weight(self, t1: TransactionNode, t2: TransactionNode) -> float:
        time_diff = abs((t2.timestamp - t1.timestamp).total_seconds())
        max_time_diff = self.config.get("max_time_difference", 86400)
        time_factor = 1 - (time_diff / max_time_diff)

        amount_ratio = min(t1.amount, t2.amount) / max(t1.amount, t2.amount)

        loc1, loc2 = t1.location, t2.location
        distance = np.sqrt((loc2[0] - loc1[0]) ** 2 + (loc2[1] - loc1[1]) ** 2)
        max_distance = self.config.get("max_distance", 100)
        distance_factor = 1 - (distance / max_distance)

        risk_factor = (t1.risk_score + t2.risk_score) / 2

        weights = {"time": 0.3, "amount": 0.2, "distance": 0.2, "risk": 0.3}

        edge_weight = (
            weights["time"] * max(0, time_factor)
            + weights["amount"] * amount_ratio
            + weights["distance"] * max(0, distance_factor)
            + weights["risk"] * risk_factor
        )

        return max(0, min(1, edge_weight))

    def _add_temporal_edge(self, t1_id: str, t2_id: str, timestamp: datetime):
        if t1_id not in self.temporal_edges:
            self.temporal_edges[t1_id] = []
        self.temporal_edges[t1_id].append((t2_id, timestamp))

    def add_transaction(self, transaction: Dict):
        node = TransactionNode(
            transaction_id=transaction["transaction_id"],
            customer_id=transaction["customer_id"],
            merchant_id=transaction["merchant_id"],
            amount=float(transaction["amount"]),
            timestamp=datetime.fromisoformat(transaction["timestamp"]),
            location=(float(transaction["latitude"]), float(transaction["longitude"])),
            risk_score=float(transaction.get("risk_score", 0.0)),
        )

        node_id = self._create_node_id(transaction)
        self.node_mapping[node_id] = node

        self.graph.add_node(
            node_id,
            customer_id=node.customer_id,
            merchant_id=node.merchant_id,
            amount=node.amount,
            timestamp=node.timestamp.isoformat(),
            latitude=node.location[0],
            longitude=node.location[1],
            risk_score=node.risk_score,
        )

        self._add_edges_for_new_transaction(node_id, node)

    def _add_edges_for_new_transaction(
        self, new_node_id: str, new_node: TransactionNode
    ):
        time_window = timedelta(seconds=self.config.get("time_window", 86400))

        for existing_id, existing_node in self.node_mapping.items():
            if existing_id != new_node_id:
                time_diff = abs(new_node.timestamp - existing_node.timestamp)

                if time_diff <= time_window:
                    if (
                        new_node.customer_id == existing_node.customer_id
                        or new_node.merchant_id == existing_node.merchant_id
                    ):

                        weight = self._calculate_edge_weight(new_node, existing_node)
                        self.graph.add_edge(new_node_id, existing_id, weight=weight)
                        self.edge_weights[(new_node_id, existing_id)] = weight
                        self._add_temporal_edge(
                            existing_id, new_node_id, new_node.timestamp
                        )

    def build_customer_subgraph(self, customer_id: str) -> nx.Graph:
        customer_nodes = [
            node_id
            for node_id, node in self.node_mapping.items()
            if node.customer_id == customer_id
        ]
        return self.graph.subgraph(customer_nodes)

    def build_merchant_subgraph(self, merchant_id: str) -> nx.Graph:
        merchant_nodes = [
            node_id
            for node_id, node in self.node_mapping.items()
            if node.merchant_id == merchant_id
        ]
        return self.graph.subgraph(merchant_nodes)

    def build_risk_subgraph(self, risk_threshold: float) -> nx.Graph:
        high_risk_nodes = [
            node_id
            for node_id, node in self.node_mapping.items()
            if node.risk_score >= risk_threshold
        ]
        return self.graph.subgraph(high_risk_nodes)

    def find_dense_subgraphs(self, min_density: float = 0.7) -> List[nx.Graph]:
        dense_subgraphs = []

        components = list(nx.connected_components(self.graph))
        for component in components:
            subgraph = self.graph.subgraph(component)
            if nx.density(subgraph) >= min_density:
                dense_subgraphs.append(subgraph)

        return dense_subgraphs

    def find_suspicious_paths(
        self, source_id: str, target_id: str, max_length: int = 5
    ) -> List[List[str]]:
        try:
            paths = list(
                nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_length)
            )

            suspicious_paths = []
            for path in paths:
                path_risk = sum(
                    self.node_mapping[node_id].risk_score for node_id in path
                ) / len(path)
                if path_risk >= self.risk_thresholds["medium"]:
                    suspicious_paths.append(path)

            return suspicious_paths
        except nx.NetworkXNoPath:
            return []

    def find_temporal_cycles(self, max_cycle_length: int = 5) -> List[List[str]]:
        cycles = []
        visited = set()

        def dfs_cycles(node_id: str, path: List[str], start_time: datetime):
            if len(path) > max_cycle_length:
                return

            if node_id in self.temporal_edges:
                for next_node, timestamp in self.temporal_edges[node_id]:
                    if timestamp >= start_time:
                        if next_node == path[0] and len(path) >= 3:
                            cycles.append(path + [next_node])
                        elif next_node not in path:
                            dfs_cycles(next_node, path + [next_node], start_time)

        for node_id in self.graph.nodes():
            if node_id not in visited:
                visited.add(node_id)
                dfs_cycles(node_id, [node_id], self.node_mapping[node_id].timestamp)

        return cycles

    def calculate_graph_metrics(self) -> GraphMetrics:
        metrics = GraphMetrics(
            density=nx.density(self.graph),
            avg_clustering=nx.average_clustering(self.graph),
            avg_degree=sum(dict(self.graph.degree()).values())
            / self.graph.number_of_nodes(),
            components=nx.number_connected_components(self.graph),
            diameter=max(
                nx.diameter(g)
                for g in (
                    self.graph.subgraph(c) for c in nx.connected_components(self.graph)
                )
            ),
            avg_path_length=(
                nx.average_shortest_path_length(self.graph)
                if nx.is_connected(self.graph)
                else float("inf")
            ),
            centrality_scores=nx.eigenvector_centrality(self.graph, max_iter=1000),
        )
        return metrics

    def get_high_centrality_nodes(self, threshold: float = 0.8) -> Set[str]:
        centrality_scores = nx.eigenvector_centrality(self.graph, max_iter=1000)
        max_centrality = max(centrality_scores.values())

        return {
            node_id
            for node_id, score in centrality_scores.items()
            if score >= threshold * max_centrality
        }

    def export_graph(self, format: str = "networkx") -> Any:
        if format == "networkx":
            return self.graph
        elif format == "adjacency":
            return nx.adjacency_matrix(self.graph)
        elif format == "json":
            return nx.node_link_data(self.graph)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def clear(self):
        self.graph.clear()
        self.node_mapping.clear()
        self.edge_weights.clear()
        self.temporal_edges.clear()
