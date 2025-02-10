from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import networkx as nx
import numpy as np
from collections import defaultdict
import json


@dataclass
class DependencyNode:
    node_id: str
    node_type: str
    evidence: Dict[str, Any]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class DependencyRelation:
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    evidence: Dict[str, Any]
    timestamp: datetime


class DependencyTracker:
    def __init__(self, config: Dict):
        self.config = config
        self.dependency_graph = nx.DiGraph()
        self.evidence_cache = {}
        self.relation_weights = {
            "causal": 1.0,
            "temporal": 0.8,
            "correlation": 0.6,
            "contextual": 0.4,
        }
        self.node_types = {
            "transaction": TransactionNode,
            "pattern": PatternNode,
            "risk": RiskNode,
            "evidence": EvidenceNode,
        }

    def add_dependency_node(self, node: DependencyNode) -> bool:
        try:
            self.dependency_graph.add_node(
                node.node_id,
                node_type=node.node_type,
                evidence=node.evidence,
                confidence=node.confidence,
                timestamp=node.timestamp,
                metadata=node.metadata,
            )
            return True
        except Exception as e:
            print(f"Error adding dependency node: {str(e)}")
            return False

    def add_dependency_relation(self, relation: DependencyRelation) -> bool:
        try:
            if not (
                self.dependency_graph.has_node(relation.source_id)
                and self.dependency_graph.has_node(relation.target_id)
            ):
                return False

            self.dependency_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relation_type=relation.relation_type,
                strength=relation.strength,
                evidence=relation.evidence,
                timestamp=relation.timestamp,
            )
            return True
        except Exception as e:
            print(f"Error adding dependency relation: {str(e)}")
            return False

    def get_node_dependencies(
        self, node_id: str, max_depth: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        if not self.dependency_graph.has_node(node_id):
            return {}

        dependencies = defaultdict(list)
        visited = set()

        def dfs_dependencies(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)

            for neighbor in self.dependency_graph.predecessors(current_id):
                edge_data = self.dependency_graph.edges[neighbor, current_id]
                relation_type = edge_data["relation_type"]
                strength = edge_data["strength"]

                dependencies[relation_type].append((neighbor, strength))
                dfs_dependencies(neighbor, depth + 1)

        dfs_dependencies(node_id, 0)
        return dict(dependencies)

    def analyze_dependency_chain(
        self, start_node: str, end_node: str
    ) -> Optional[Dict[str, Any]]:
        if not (
            self.dependency_graph.has_node(start_node)
            and self.dependency_graph.has_node(end_node)
        ):
            return None

        try:
            paths = list(
                nx.all_simple_paths(self.dependency_graph, start_node, end_node)
            )

            if not paths:
                return None

            chain_analysis = []
            for path in paths:
                path_strength = 1.0
                path_evidence = []

                for i in range(len(path) - 1):
                    edge_data = self.dependency_graph.edges[path[i], path[i + 1]]
                    path_strength *= edge_data["strength"]
                    path_evidence.append(edge_data["evidence"])

                chain_analysis.append(
                    {"path": path, "strength": path_strength, "evidence": path_evidence}
                )

            strongest_chain = max(chain_analysis, key=lambda x: x["strength"])
            return strongest_chain
        except Exception as e:
            print(f"Error analyzing dependency chain: {str(e)}")
            return None

    def find_critical_dependencies(
        self, confidence_threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        critical_deps = []

        for source, target, data in self.dependency_graph.edges(data=True):
            if data["strength"] >= confidence_threshold:
                source_node = self.dependency_graph.nodes[source]
                target_node = self.dependency_graph.nodes[target]

                if (
                    source_node["confidence"] >= confidence_threshold
                    and target_node["confidence"] >= confidence_threshold
                ):
                    critical_deps.append((source, target, data["strength"]))

        return sorted(critical_deps, key=lambda x: x[2], reverse=True)

    def aggregate_evidence(self, node_ids: List[str]) -> Dict[str, Any]:
        if not all(self.dependency_graph.has_node(nid) for nid in node_ids):
            return {}

        evidence_aggregate = defaultdict(list)
        confidence_scores = []
        timestamps = []

        for node_id in node_ids:
            node_data = self.dependency_graph.nodes[node_id]
            evidence = node_data["evidence"]

            for key, value in evidence.items():
                evidence_aggregate[key].append(value)

            confidence_scores.append(node_data["confidence"])
            timestamps.append(node_data["timestamp"])

        return {
            "evidence": dict(evidence_aggregate),
            "confidence": np.mean(confidence_scores),
            "time_range": {"start": min(timestamps), "end": max(timestamps)},
        }

    def detect_circular_dependencies(self) -> List[List[str]]:
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return [cycle for cycle in cycles if len(cycle) > 2]
        except Exception as e:
            print(f"Error detecting circular dependencies: {str(e)}")
            return []

    def validate_dependency_chain(self, chain: List[str]) -> Tuple[bool, List[str]]:
        if len(chain) < 2:
            return False, ["Chain too short"]

        errors = []

        for i in range(len(chain) - 1):
            source, target = chain[i], chain[i + 1]

            if not self.dependency_graph.has_edge(source, target):
                errors.append(f"Missing edge between {source} and {target}")
                continue

            edge_data = self.dependency_graph.edges[source, target]
            source_data = self.dependency_graph.nodes[source]
            target_data = self.dependency_graph.nodes[target]

            if edge_data["strength"] < 0.2:
                errors.append(f"Weak connection between {source} and {target}")

            if source_data["timestamp"] > target_data["timestamp"]:
                errors.append(f"Invalid temporal order between {source} and {target}")

        return len(errors) == 0, errors

    def prune_weak_dependencies(self, strength_threshold: float = 0.3) -> int:
        weak_edges = [
            (source, target)
            for source, target, data in self.dependency_graph.edges(data=True)
            if data["strength"] < strength_threshold
        ]

        self.dependency_graph.remove_edges_from(weak_edges)
        return len(weak_edges)

    def get_dependency_metrics(self) -> Dict[str, float]:
        try:
            return {
                "node_count": self.dependency_graph.number_of_nodes(),
                "edge_count": self.dependency_graph.number_of_edges(),
                "avg_degree": np.mean([d for _, d in self.dependency_graph.degree()]),
                "avg_strength": np.mean(
                    [
                        d["strength"]
                        for _, _, d in self.dependency_graph.edges(data=True)
                    ]
                ),
                "density": nx.density(self.dependency_graph),
                "transitivity": nx.transitivity(self.dependency_graph),
            }
        except Exception as e:
            print(f"Error calculating dependency metrics: {str(e)}")
            return {}


class TransactionNode(DependencyNode):
    def __init__(
        self, transaction_id: str, transaction_data: Dict[str, Any], confidence: float
    ):
        super().__init__(
            node_id=transaction_id,
            node_type="transaction",
            evidence=transaction_data,
            confidence=confidence,
            timestamp=datetime.fromisoformat(transaction_data["timestamp"]),
            metadata={
                "amount": transaction_data["amount"],
                "currency": transaction_data.get("currency", "USD"),
                "merchant_id": transaction_data.get("merchant_id"),
            },
        )


class PatternNode(DependencyNode):
    def __init__(
        self, pattern_id: str, pattern_data: Dict[str, Any], confidence: float
    ):
        super().__init__(
            node_id=pattern_id,
            node_type="pattern",
            evidence=pattern_data,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            metadata={
                "pattern_type": pattern_data["type"],
                "frequency": pattern_data.get("frequency", 0),
                "duration": pattern_data.get("duration", 0),
            },
        )


class RiskNode(DependencyNode):
    def __init__(self, risk_id: str, risk_data: Dict[str, Any], confidence: float):
        super().__init__(
            node_id=risk_id,
            node_type="risk",
            evidence=risk_data,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            metadata={
                "risk_level": risk_data["level"],
                "risk_factors": risk_data.get("factors", []),
                "priority": risk_data.get("priority", "medium"),
            },
        )


class EvidenceNode(DependencyNode):
    def __init__(
        self, evidence_id: str, evidence_data: Dict[str, Any], confidence: float
    ):
        super().__init__(
            node_id=evidence_id,
            node_type="evidence",
            evidence=evidence_data,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            metadata={
                "source": evidence_data.get("source", "unknown"),
                "category": evidence_data.get("category", "general"),
                "verification_status": evidence_data.get("verified", False),
            },
        )
