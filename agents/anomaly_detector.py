from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import zscore
from .base_agent import BaseAgent, AgentMessage


class AnomalyDetector(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(
            agent_id=agent_id, agent_type="anomaly_detector", config=config
        )
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.lof = LocalOutlierFactor(contamination=0.1, novelty=True)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.anomaly_cache = defaultdict(list)
        self.baseline_stats = defaultdict(dict)
        self.detection_methods = {
            "isolation_forest": True,
            "lof": True,
            "zscore": True,
            "pca": True,
        }

    async def initialize(self) -> bool:
        try:
            self.register_message_handler(
                "detect_anomalies", self._handle_anomaly_detection
            )
            self.register_message_handler("update_methods", self._handle_method_update)
            self.register_message_handler(
                "update_baseline", self._handle_baseline_update
            )
            return True
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            return False

    def _extract_features(self, transaction_data: Dict[str, Any]) -> np.ndarray:
        features = []
        amount = float(transaction_data["amount"])
        hour = datetime.fromisoformat(transaction_data["timestamp"]).hour

        features.extend(
            [
                amount,
                hour,
                float(transaction_data.get("latitude", 0)),
                float(transaction_data.get("longitude", 0)),
                float(transaction_data.get("velocity_1h", 0)),
                float(transaction_data.get("velocity_24h", 0)),
                float(transaction_data.get("distance_from_last", 0)),
            ]
        )

        return np.array(features).reshape(1, -1)

    def _detect_isolation_forest(self, features: np.ndarray) -> Tuple[bool, float]:
        score = self.isolation_forest.score_samples(features)[0]
        is_anomaly = self.isolation_forest.predict(features)[0] == -1
        return is_anomaly, score

    def _detect_lof(self, features: np.ndarray) -> Tuple[bool, float]:
        score = self.lof.score_samples(features)[0]
        is_anomaly = score < self.lof.offset_
        return is_anomaly, score

    def _detect_zscore(
        self, features: np.ndarray, threshold: float = 3.0
    ) -> Tuple[bool, List[float]]:
        z_scores = zscore(features, axis=0)
        is_anomaly = np.any(np.abs(z_scores) > threshold)
        return is_anomaly, z_scores.tolist()

    def _detect_pca(self, features: np.ndarray) -> Tuple[bool, float]:
        transformed = self.pca.transform(features)
        reconstructed = self.pca.inverse_transform(transformed)
        reconstruction_error = np.mean((features - reconstructed) ** 2)
        is_anomaly = reconstruction_error > self.pca.explained_variance_.mean()
        return is_anomaly, reconstruction_error

    def _update_baseline_stats(self, customer_id: str, features: np.ndarray):
        if customer_id not in self.baseline_stats:
            self.baseline_stats[customer_id] = {
                "mean": features.mean(axis=0),
                "std": features.std(axis=0),
                "min": features.min(axis=0),
                "max": features.max(axis=0),
                "last_update": datetime.utcnow(),
            }
        else:
            alpha = 0.1
            stats = self.baseline_stats[customer_id]
            stats["mean"] = (1 - alpha) * stats["mean"] + alpha * features.mean(axis=0)
            stats["std"] = (1 - alpha) * stats["std"] + alpha * features.std(axis=0)
            stats["min"] = np.minimum(stats["min"], features.min(axis=0))
            stats["max"] = np.maximum(stats["max"], features.max(axis=0))
            stats["last_update"] = datetime.utcnow()

    async def _handle_anomaly_detection(self, message: AgentMessage) -> Dict[str, Any]:
        data = message.content
        success, result = await self.process(data)

        if success:
            customer_id = data["transaction_data"].get("customer_id")
            if customer_id:
                self.anomaly_cache[customer_id].append(
                    {"timestamp": datetime.utcnow(), "anomaly_details": result}
                )
            return {"status": "success", "detection_result": result}
        return {"status": "error", "message": "Anomaly detection failed"}

    async def _handle_method_update(self, message: AgentMessage) -> Dict[str, Any]:
        new_methods = message.content.get("methods", {})
        self.detection_methods.update(new_methods)
        return {"status": "success", "message": "Detection methods updated"}

    async def _handle_baseline_update(self, message: AgentMessage) -> Dict[str, Any]:
        customer_id = message.content.get("customer_id")
        baseline_data = message.content.get("baseline_data", [])

        if customer_id and baseline_data:
            features = np.array([self._extract_features(t) for t in baseline_data])
            self._update_baseline_stats(customer_id, features)
            return {"status": "success", "message": "Baseline updated"}
        return {"status": "error", "message": "Invalid baseline update data"}

    def _combine_anomaly_scores(self, detection_results: Dict[str, Any]) -> float:
        scores = []
        weights = {"isolation_forest": 0.3, "lof": 0.3, "zscore": 0.2, "pca": 0.2}

        for method, result in detection_results.items():
            if method in weights and "score" in result:
                scores.append(result["score"] * weights[method])

        return sum(scores) / sum(weights.values()) if scores else 0.0

    async def process(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        try:
            transaction_data = data["transaction_data"]
            customer_id = transaction_data.get("customer_id", "")
            features = self._extract_features(transaction_data)
            scaled_features = self.scaler.fit_transform(features)

            detection_results = {}
            anomaly_types = set()

            if self.detection_methods["isolation_forest"]:
                is_anomaly, score = self._detect_isolation_forest(scaled_features)
                detection_results["isolation_forest"] = {
                    "is_anomaly": is_anomaly,
                    "score": score,
                }
                if is_anomaly:
                    anomaly_types.add("isolation_forest")

            if self.detection_methods["lof"]:
                is_anomaly, score = self._detect_lof(scaled_features)
                detection_results["lof"] = {"is_anomaly": is_anomaly, "score": score}
                if is_anomaly:
                    anomaly_types.add("local_outlier")

            if self.detection_methods["zscore"]:
                is_anomaly, z_scores = self._detect_zscore(scaled_features)
                detection_results["zscore"] = {
                    "is_anomaly": is_anomaly,
                    "scores": z_scores,
                }
                if is_anomaly:
                    anomaly_types.add("statistical_outlier")

            if self.detection_methods["pca"]:
                is_anomaly, error = self._detect_pca(scaled_features)
                detection_results["pca"] = {"is_anomaly": is_anomaly, "score": error}
                if is_anomaly:
                    anomaly_types.add("reconstruction_error")

            combined_score = self._combine_anomaly_scores(detection_results)
            is_anomaly = len(anomaly_types) >= 2 or combined_score > 0.7

            if customer_id:
                self._update_baseline_stats(customer_id, features)

            result = {
                "is_anomaly": is_anomaly,
                "anomaly_score": combined_score,
                "anomaly_types": list(anomaly_types),
                "detection_results": detection_results,
                "timestamp": datetime.utcnow().isoformat(),
                "feature_importance": {
                    "amount": features[0][0],
                    "hour": features[0][1],
                    "location": np.sqrt(features[0][2] ** 2 + features[0][3] ** 2),
                    "velocity": max(features[0][4], features[0][5]),
                    "distance": features[0][6],
                },
            }

            return True, result

        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return False, {"error": str(e)}

    async def cleanup(self) -> bool:
        try:
            self.anomaly_cache.clear()
            self.baseline_stats.clear()
            return True
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
            return False
