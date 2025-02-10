from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.stats import zscore
import asyncio
import json
from .base_agent import BaseAgent, AgentMessage


class PatternDetector(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(
            agent_id=agent_id, agent_type="pattern_detector", config=config
        )
        self.pattern_cache = {}
        self.transaction_buffer = defaultdict(list)
        self.graph = nx.Graph()
        self.clustering = DBSCAN(eps=0.3, min_samples=2)
        self.pattern_thresholds = {
            "velocity_threshold": 5.0,
            "amount_threshold": 1000.0,
            "time_window": 3600,
            "distance_threshold": 100.0,
        }

    async def initialize(self) -> bool:
        try:
            self.register_message_handler(
                "detect_patterns", self._handle_pattern_detection
            )
            self.register_message_handler(
                "update_thresholds", self._handle_threshold_update
            )
            self.register_message_handler("clear_patterns", self._handle_clear_patterns)
            return True
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            return False

    async def _handle_pattern_detection(self, message: AgentMessage) -> Dict[str, Any]:
        transactions = message.content.get("transactions", [])
        window_size = message.content.get("window_size", 3600)

        success, patterns = await self.process(
            {"transactions": transactions, "window_size": window_size}
        )

        if success:
            self._update_pattern_cache(patterns)
            return {"status": "success", "patterns": patterns}
        return {"status": "error", "message": "Pattern detection failed"}

    async def _handle_threshold_update(self, message: AgentMessage) -> Dict[str, Any]:
        new_thresholds = message.content.get("thresholds", {})
        self.pattern_thresholds.update(new_thresholds)
        return {"status": "success", "message": "Thresholds updated"}

    async def _handle_clear_patterns(self, message: AgentMessage) -> Dict[str, Any]:
        self.pattern_cache.clear()
        self.transaction_buffer.clear()
        self.graph.clear()
        return {"status": "success", "message": "Patterns cleared"}

    def _update_pattern_cache(self, patterns: Dict[str, Any]):
        for pattern_id, pattern in patterns.items():
            self.pattern_cache[pattern_id] = {
                "pattern": pattern,
                "timestamp": datetime.utcnow(),
                "status": "active",
            }

    def _detect_velocity_patterns(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        patterns = []
        sorted_txns = sorted(
            transactions, key=lambda x: datetime.fromisoformat(x["timestamp"])
        )

        for i in range(len(sorted_txns) - 1):
            current_txn = sorted_txns[i]
            next_txn = sorted_txns[i + 1]

            time_diff = (
                datetime.fromisoformat(next_txn["timestamp"])
                - datetime.fromisoformat(current_txn["timestamp"])
            ).total_seconds()

            if time_diff < self.pattern_thresholds["time_window"]:
                if (
                    float(next_txn["amount"]) + float(current_txn["amount"])
                    > self.pattern_thresholds["amount_threshold"]
                ):
                    patterns.append(
                        {
                            "type": "velocity",
                            "transactions": [
                                current_txn["transaction_id"],
                                next_txn["transaction_id"],
                            ],
                            "time_difference": time_diff,
                            "total_amount": float(next_txn["amount"])
                            + float(current_txn["amount"]),
                        }
                    )

        return patterns

    def _detect_location_patterns(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        patterns = []
        coordinates = []

        for txn in transactions:
            if "latitude" in txn and "longitude" in txn:
                coordinates.append([float(txn["latitude"]), float(txn["longitude"])])

        if len(coordinates) > 1:
            clusters = self.clustering.fit_predict(coordinates)

            cluster_groups = defaultdict(list)
            for idx, cluster_id in enumerate(clusters):
                if cluster_id != -1:
                    cluster_groups[cluster_id].append(transactions[idx])

            for cluster_id, cluster_txns in cluster_groups.items():
                if len(cluster_txns) >= 2:
                    patterns.append(
                        {
                            "type": "location_cluster",
                            "transactions": [t["transaction_id"] for t in cluster_txns],
                            "center": np.mean(coordinates, axis=0).tolist(),
                            "radius": np.std(coordinates, axis=0).mean(),
                        }
                    )

        return patterns

    def _detect_amount_patterns(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        patterns = []
        amounts = [float(t["amount"]) for t in transactions]

        if len(amounts) > 1:
            z_scores = zscore(amounts)
            outliers = [i for i, z in enumerate(z_scores) if abs(z) > 2]

            if outliers:
                patterns.append(
                    {
                        "type": "amount_anomaly",
                        "transactions": [
                            transactions[i]["transaction_id"] for i in outliers
                        ],
                        "z_scores": {
                            transactions[i]["transaction_id"]: z_scores[i]
                            for i in outliers
                        },
                        "mean_amount": np.mean(amounts),
                        "std_amount": np.std(amounts),
                    }
                )

        return patterns

    def _build_transaction_graph(self, transactions: List[Dict[str, Any]]):
        self.graph.clear()

        for txn in transactions:
            self.graph.add_node(txn["transaction_id"], **txn)

        for i, txn1 in enumerate(transactions):
            for txn2 in transactions[i + 1 :]:
                time_diff = abs(
                    (
                        datetime.fromisoformat(txn2["timestamp"])
                        - datetime.fromisoformat(txn1["timestamp"])
                    ).total_seconds()
                )

                if time_diff < self.pattern_thresholds["time_window"]:
                    self.graph.add_edge(
                        txn1["transaction_id"],
                        txn2["transaction_id"],
                        time_diff=time_diff,
                    )

    def _detect_graph_patterns(self) -> List[Dict[str, Any]]:
        patterns = []

        connected_components = list(nx.connected_components(self.graph))
        for component in connected_components:
            if len(component) >= 2:
                subgraph = self.graph.subgraph(component)
                patterns.append(
                    {
                        "type": "connected_transactions",
                        "transactions": list(component),
                        "density": nx.density(subgraph),
                        "diameter": (
                            nx.diameter(subgraph) if nx.is_connected(subgraph) else None
                        ),
                    }
                )

        return patterns

    async def process(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        try:
            transactions = data["transactions"]
            if not transactions:
                return True, {"patterns": []}

            self._build_transaction_graph(transactions)

            velocity_patterns = self._detect_velocity_patterns(transactions)
            location_patterns = self._detect_location_patterns(transactions)
            amount_patterns = self._detect_amount_patterns(transactions)
            graph_patterns = self._detect_graph_patterns()

            all_patterns = {
                "timestamp": datetime.utcnow().isoformat(),
                "patterns": {
                    "velocity": velocity_patterns,
                    "location": location_patterns,
                    "amount": amount_patterns,
                    "graph": graph_patterns,
                },
                "metadata": {
                    "transaction_count": len(transactions),
                    "time_window": data.get("window_size", 3600),
                    "thresholds": self.pattern_thresholds,
                },
            }

            return True, all_patterns

        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return False, {"error": str(e)}

    async def cleanup(self) -> bool:
        try:
            self.pattern_cache.clear()
            self.transaction_buffer.clear()
            self.graph.clear()
            return True
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
            return False
