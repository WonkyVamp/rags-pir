from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import entropy
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd


@dataclass
class SequencePattern:
    pattern_id: str
    transactions: List[str]
    time_deltas: List[float]
    amounts: List[float]
    risk_scores: List[float]
    confidence: float
    pattern_type: str


@dataclass
class SequenceMetrics:
    entropy: float
    periodicity: float
    complexity: float
    predictability: float
    anomaly_score: float


class SequenceAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.sequence_cache = {}
        self.pattern_registry = defaultdict(list)
        self.temporal_graph = nx.DiGraph()
        self.scaler = StandardScaler()
        self.clustering = DBSCAN(eps=0.3, min_samples=2)

        self.time_windows = {
            "short": 3600,  # 1 hour
            "medium": 86400,  # 1 day
            "long": 604800,  # 1 week
        }

    def _extract_sequence_features(
        self, transactions: List[Dict[str, Any]]
    ) -> np.ndarray:
        features = []
        sorted_txns = sorted(
            transactions, key=lambda x: datetime.fromisoformat(x["timestamp"])
        )

        for i in range(len(sorted_txns) - 1):
            current = sorted_txns[i]
            next_txn = sorted_txns[i + 1]

            time_delta = (
                datetime.fromisoformat(next_txn["timestamp"])
                - datetime.fromisoformat(current["timestamp"])
            ).total_seconds()

            amount_ratio = float(next_txn["amount"]) / float(current["amount"])
            risk_delta = float(next_txn.get("risk_score", 0)) - float(
                current.get("risk_score", 0)
            )

            features.append(
                [
                    time_delta,
                    amount_ratio,
                    risk_delta,
                    float(current.get("risk_score", 0)),
                    float(next_txn.get("risk_score", 0)),
                ]
            )

        return np.array(features)

    def _identify_temporal_patterns(
        self, features: np.ndarray, transactions: List[Dict[str, Any]]
    ) -> List[SequencePattern]:
        patterns = []

        if len(features) < 2:
            return patterns

        scaled_features = self.scaler.fit_transform(features)
        clusters = self.clustering.fit_predict(scaled_features)

        unique_clusters = set(clusters)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) < 2:
                continue

            cluster_features = features[cluster_indices]
            cluster_txns = [transactions[i : i + 2] for i in cluster_indices]

            time_deltas = cluster_features[:, 0]
            amounts = np.array([float(t[0]["amount"]) for t in cluster_txns])
            risk_scores = np.array(
                [float(t[0].get("risk_score", 0)) for t in cluster_txns]
            )

            pattern_type = self._determine_pattern_type(
                time_deltas, amounts, risk_scores
            )
            confidence = self._calculate_pattern_confidence(cluster_features)

            pattern = SequencePattern(
                pattern_id=f"pattern_{cluster_id}",
                transactions=[t[0]["transaction_id"] for t in cluster_txns],
                time_deltas=time_deltas.tolist(),
                amounts=amounts.tolist(),
                risk_scores=risk_scores.tolist(),
                confidence=confidence,
                pattern_type=pattern_type,
            )

            patterns.append(pattern)

        return patterns

    def _determine_pattern_type(
        self, time_deltas: np.ndarray, amounts: np.ndarray, risk_scores: np.ndarray
    ) -> str:
        time_std = np.std(time_deltas)
        amount_ratio = np.max(amounts) / np.min(amounts)
        risk_mean = np.mean(risk_scores)

        if time_std < 60 and amount_ratio > 10:
            return "rapid_high_value"
        elif time_std < 300 and risk_mean > 0.7:
            return "rapid_high_risk"
        elif np.all(np.abs(np.diff(amounts)) < 0.1 * amounts[:-1]):
            return "consistent_amount"
        elif np.all(np.abs(np.diff(time_deltas)) < 0.1 * time_deltas[:-1]):
            return "periodic"
        else:
            return "irregular"

    def _calculate_pattern_confidence(self, features: np.ndarray) -> float:
        time_consistency = 1 / (1 + np.std(features[:, 0]))
        amount_consistency = 1 / (1 + np.std(features[:, 1]))
        risk_consistency = 1 / (1 + np.std(features[:, 2:]))

        weights = [0.4, 0.3, 0.3]
        confidence = (
            weights[0] * time_consistency
            + weights[1] * amount_consistency
            + weights[2] * risk_consistency
        )

        return min(1.0, confidence)

    def _build_temporal_graph(
        self, transactions: List[Dict[str, Any]], patterns: List[SequencePattern]
    ):
        self.temporal_graph.clear()

        for txn in transactions:
            self.temporal_graph.add_node(txn["transaction_id"], **txn)

        for pattern in patterns:
            for i in range(len(pattern.transactions) - 1):
                self.temporal_graph.add_edge(
                    pattern.transactions[i],
                    pattern.transactions[i + 1],
                    pattern_id=pattern.pattern_id,
                    time_delta=pattern.time_deltas[i],
                    confidence=pattern.confidence,
                )

    def _calculate_sequence_metrics(
        self, features: np.ndarray, patterns: List[SequencePattern]
    ) -> SequenceMetrics:
        if len(features) < 2:
            return SequenceMetrics(0, 0, 0, 0, 0)

        time_deltas = features[:, 0]
        amounts = features[:, 1]

        # Calculate entropy
        hist, _ = np.histogram(time_deltas, bins="auto", density=True)
        sequence_entropy = entropy(hist + 1e-10)

        # Calculate periodicity
        fft = np.abs(np.fft.fft(time_deltas))
        periodicity = np.max(fft[1:]) / np.sum(fft[1:])

        # Calculate complexity
        complexity = len(patterns) / len(features)

        # Calculate predictability
        predictability = np.mean([p.confidence for p in patterns]) if patterns else 0

        # Calculate anomaly score
        scaled_features = self.scaler.fit_transform(features)
        anomaly_score = np.mean(np.abs(scaled_features))

        return SequenceMetrics(
            entropy=float(sequence_entropy),
            periodicity=float(periodicity),
            complexity=float(complexity),
            predictability=float(predictability),
            anomaly_score=float(anomaly_score),
        )

    def analyze_sequence(
        self, transactions: List[Dict[str, Any]]
    ) -> Tuple[List[SequencePattern], SequenceMetrics]:
        if not transactions:
            return [], SequenceMetrics(0, 0, 0, 0, 0)

        sequence_key = tuple(t["transaction_id"] for t in transactions)

        if sequence_key in self.sequence_cache:
            return self.sequence_cache[sequence_key]

        features = self._extract_sequence_features(transactions)
        patterns = self._identify_temporal_patterns(features, transactions)
        metrics = self._calculate_sequence_metrics(features, patterns)

        self._build_temporal_graph(transactions, patterns)

        for pattern in patterns:
            self.pattern_registry[pattern.pattern_type].append(pattern)

        self.sequence_cache[sequence_key] = (patterns, metrics)
        return patterns, metrics

    def find_similar_sequences(
        self, target_sequence: List[Dict[str, Any]], threshold: float = 0.8
    ) -> List[Tuple[List[Dict[str, Any]], float]]:
        target_patterns, _ = self.analyze_sequence(target_sequence)

        if not target_patterns:
            return []

        similar_sequences = []

        for cached_key, (cached_patterns, _) in self.sequence_cache.items():
            if not cached_patterns:
                continue

            similarity = self._calculate_sequence_similarity(
                target_patterns, cached_patterns
            )

            if similarity >= threshold:
                similar_sequences.append(
                    (
                        [t for t in self.temporal_graph.nodes() if t in cached_key],
                        similarity,
                    )
                )

        return sorted(similar_sequences, key=lambda x: x[1], reverse=True)

    def _calculate_sequence_similarity(
        self, patterns1: List[SequencePattern], patterns2: List[SequencePattern]
    ) -> float:
        if not patterns1 or not patterns2:
            return 0.0

        similarities = []

        for p1 in patterns1:
            for p2 in patterns2:
                if p1.pattern_type == p2.pattern_type:
                    time_sim = 1 / (
                        1
                        + np.mean(
                            np.abs(np.array(p1.time_deltas) - np.array(p2.time_deltas))
                        )
                    )
                    amount_sim = 1 / (
                        1 + np.mean(np.abs(np.array(p1.amounts) - np.array(p2.amounts)))
                    )
                    risk_sim = 1 / (
                        1
                        + np.mean(
                            np.abs(np.array(p1.risk_scores) - np.array(p2.risk_scores))
                        )
                    )

                    similarity = (time_sim + amount_sim + risk_sim) / 3
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def find_anomalous_subsequences(
        self, transactions: List[Dict[str, Any]], window_size: int = 5
    ) -> List[List[Dict[str, Any]]]:
        anomalous_sequences = []

        for i in range(len(transactions) - window_size + 1):
            subsequence = transactions[i : i + window_size]
            _, metrics = self.analyze_sequence(subsequence)

            if metrics.anomaly_score > self.config.get("anomaly_threshold", 0.8):
                anomalous_sequences.append(subsequence)

        return anomalous_sequences

    def get_pattern_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}

        for pattern_type, patterns in self.pattern_registry.items():
            if not patterns:
                continue

            confidences = [p.confidence for p in patterns]
            time_deltas = [td for p in patterns for td in p.time_deltas]
            risk_scores = [rs for p in patterns for rs in p.risk_scores]

            stats[pattern_type] = {
                "count": len(patterns),
                "avg_confidence": np.mean(confidences),
                "avg_time_delta": np.mean(time_deltas),
                "avg_risk_score": np.mean(risk_scores),
                "pattern_frequency": len(patterns) / len(self.sequence_cache),
            }

        return stats

    def predict_next_transaction(
        self, sequence: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if len(sequence) < 2:
            return None

        patterns, metrics = self.analyze_sequence(sequence)

        if not patterns or metrics.predictability < 0.5:
            return None

        most_recent = sequence[-1]
        relevant_patterns = [
            p for p in patterns if most_recent["transaction_id"] in p.transactions
        ]

        if not relevant_patterns:
            return None

        best_pattern = max(relevant_patterns, key=lambda p: p.confidence)

        predicted_amount = np.mean(best_pattern.amounts)
        predicted_time_delta = np.mean(best_pattern.time_deltas)
        predicted_risk = np.mean(best_pattern.risk_scores)

        return {
            "predicted_amount": predicted_amount,
            "predicted_time": datetime.fromisoformat(most_recent["timestamp"])
            + timedelta(seconds=predicted_time_delta),
            "predicted_risk_score": predicted_risk,
            "confidence": best_pattern.confidence,
            "pattern_type": best_pattern.pattern_type,
        }
