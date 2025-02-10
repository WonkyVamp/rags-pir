import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import community
from dataclasses import dataclass
import pandas as pd
from scipy.stats import entropy
from datetime import datetime, timedelta


@dataclass
class CommunityMetrics:
    modularity: float
    num_communities: int
    sizes: Dict[int, int]
    internal_densities: Dict[int, float]
    external_densities: Dict[int, float]
    risk_scores: Dict[int, float]
    temporal_patterns: Dict[int, List[Tuple[datetime, float]]]


class CommunityDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.communities = None
        self.metrics = None
        self.resolution = config.get("resolution", 1.0)
        self.min_community_size = config.get("min_community_size", 3)
        self.risk_threshold = config.get("risk_threshold", 0.7)
        self.temporal_window = config.get("temporal_window", 3600)

    def _detect_louvain_communities(self, graph: nx.Graph) -> Dict[str, int]:
        return community.best_partition(
            graph, resolution=self.resolution, random_state=42
        )

    def _detect_leiden_communities(self, graph: nx.Graph) -> Dict[str, int]:
        import leidenalg as la
        import igraph as ig

        g_ig = ig.Graph.from_networkx(graph)
        partition = la.find_partition(
            g_ig,
            la.ModularityVertexPartition,
            resolution_parameter=self.resolution,
            seed=42,
        )

        return {str(v.index): p_id for p_id, part in enumerate(partition) for v in part}

    def _calculate_community_metrics(
        self, graph: nx.Graph, communities: Dict[str, int]
    ) -> CommunityMetrics:
        inv_communities = defaultdict(list)
        for node, comm in communities.items():
            inv_communities[comm].append(node)

        modularity = community.modularity(communities, graph)
        num_communities = len(inv_communities)

        sizes = {comm: len(nodes) for comm, nodes in inv_communities.items()}

        internal_densities = {}
        external_densities = {}
        risk_scores = {}
        temporal_patterns = {}

        for comm_id, nodes in inv_communities.items():
            subgraph = graph.subgraph(nodes)
            internal_densities[comm_id] = nx.density(subgraph)

            external_edges = sum(
                1
                for n in nodes
                for neighbor in graph.neighbors(n)
                if communities[neighbor] != comm_id
            )
            possible_external = len(nodes) * (graph.number_of_nodes() - len(nodes))
            external_densities[comm_id] = (
                external_edges / possible_external if possible_external > 0 else 0
            )

            risk_scores[comm_id] = np.mean(
                [graph.nodes[n].get("risk_score", 0) for n in nodes]
            )

            temporal_patterns[comm_id] = self._analyze_temporal_patterns(graph, nodes)

        return CommunityMetrics(
            modularity=modularity,
            num_communities=num_communities,
            sizes=sizes,
            internal_densities=internal_densities,
            external_densities=external_densities,
            risk_scores=risk_scores,
            temporal_patterns=temporal_patterns,
        )

    def _analyze_temporal_patterns(
        self, graph: nx.Graph, nodes: List[str]
    ) -> List[Tuple[datetime, float]]:
        timestamps = [
            datetime.fromisoformat(graph.nodes[n]["timestamp"]) for n in nodes
        ]

        if not timestamps:
            return []

        start_time = min(timestamps)
        end_time = max(timestamps)
        current_time = start_time

        patterns = []
        while current_time <= end_time:
            window_end = current_time + timedelta(seconds=self.temporal_window)
            window_txns = sum(1 for t in timestamps if current_time <= t < window_end)
            activity_rate = window_txns / self.temporal_window
            patterns.append((current_time, activity_rate))
            current_time = window_end

        return patterns

    def _identify_bridge_communities(
        self, graph: nx.Graph, communities: Dict[str, int]
    ) -> Set[int]:
        edge_counts = defaultdict(int)

        for u, v in graph.edges():
            comm_u = communities[u]
            comm_v = communities[v]
            if comm_u != comm_v:
                edge_counts[(comm_u, comm_v)] += 1

        avg_edges = np.mean(list(edge_counts.values()))
        std_edges = np.std(list(edge_counts.values()))
        threshold = avg_edges + 2 * std_edges

        bridge_communities = set()
        for (comm1, comm2), count in edge_counts.items():
            if count > threshold:
                bridge_communities.add(comm1)
                bridge_communities.add(comm2)

        return bridge_communities

    def _identify_suspicious_communities(
        self, graph: nx.Graph, communities: Dict[str, int], metrics: CommunityMetrics
    ) -> Set[int]:
        suspicious = set()

        for comm_id in range(metrics.num_communities):
            if (
                metrics.risk_scores[comm_id] >= self.risk_threshold
                and metrics.internal_densities[comm_id] > 0.5
                and metrics.sizes[comm_id] >= self.min_community_size
            ):
                suspicious.add(comm_id)

        return suspicious

    def _analyze_community_roles(
        self, graph: nx.Graph, communities: Dict[str, int]
    ) -> Dict[str, Dict[str, float]]:
        roles = {}

        for node in graph.nodes():
            comm = communities[node]
            neighbors = list(graph.neighbors(node))

            if not neighbors:
                continue

            internal_edges = sum(1 for n in neighbors if communities[n] == comm)
            external_edges = len(neighbors) - internal_edges

            within_comm_ratio = internal_edges / len(neighbors)
            participation_coeff = 1 - within_comm_ratio**2

            z_score = (
                len(neighbors)
                - np.mean(
                    [
                        len(list(graph.neighbors(n)))
                        for n in graph.nodes()
                        if communities[n] == comm
                    ]
                )
            ) / max(
                np.std(
                    [
                        len(list(graph.neighbors(n)))
                        for n in graph.nodes()
                        if communities[n] == comm
                    ]
                ),
                1,
            )

            roles[node] = {
                "participation_coefficient": participation_coeff,
                "within_module_degree": z_score,
                "hub_score": graph.degree(node) * participation_coeff,
            }

        return roles

    def _calculate_community_similarity(
        self, communities1: Dict[str, int], communities2: Dict[str, int]
    ) -> float:
        from sklearn.metrics import adjusted_mutual_info_score

        nodes = set(communities1.keys()) & set(communities2.keys())
        if not nodes:
            return 0.0

        labels1 = [communities1[n] for n in nodes]
        labels2 = [communities2[n] for n in nodes]

        return adjusted_mutual_info_score(labels1, labels2)

    def detect_communities(
        self, graph: nx.Graph, method: str = "louvain"
    ) -> Tuple[Dict[str, int], CommunityMetrics]:
        if method == "louvain":
            self.communities = self._detect_louvain_communities(graph)
        elif method == "leiden":
            self.communities = self._detect_leiden_communities(graph)
        else:
            raise ValueError(f"Unsupported community detection method: {method}")

        self.metrics = self._calculate_community_metrics(graph, self.communities)

        return self.communities, self.metrics

    def analyze_community_evolution(
        self, graph: nx.Graph, time_windows: List[Tuple[datetime, datetime]]
    ) -> List[Tuple[Dict[str, int], float]]:
        evolution = []
        previous_communities = None

        for start_time, end_time in time_windows:
            subgraph = nx.Graph()

            for node, attrs in graph.nodes(data=True):
                timestamp = datetime.fromisoformat(attrs["timestamp"])
                if start_time <= timestamp < end_time:
                    subgraph.add_node(node, **attrs)

            for u, v, attrs in graph.edges(data=True):
                if u in subgraph and v in subgraph:
                    subgraph.add_edge(u, v, **attrs)

            if subgraph.number_of_nodes() > 0:
                communities = self._detect_louvain_communities(subgraph)

                if previous_communities:
                    similarity = self._calculate_community_similarity(
                        previous_communities, communities
                    )
                else:
                    similarity = 1.0

                evolution.append((communities, similarity))
                previous_communities = communities

        return evolution

    def export_community_data(self) -> Dict:
        if not self.communities or not self.metrics:
            raise ValueError("Must run detect_communities first")

        return {
            "communities": self.communities,
            "metrics": {
                "modularity": self.metrics.modularity,
                "num_communities": self.metrics.num_communities,
                "sizes": self.metrics.sizes,
                "internal_densities": self.metrics.internal_densities,
                "external_densities": self.metrics.external_densities,
                "risk_scores": self.metrics.risk_scores,
            },
        }

    def get_community_subgraph(self, graph: nx.Graph, community_id: int) -> nx.Graph:
        if not self.communities:
            raise ValueError("Must run detect_communities first")

        nodes = [
            node for node, comm in self.communities.items() if comm == community_id
        ]
        return graph.subgraph(nodes)
