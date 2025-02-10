from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
from sklearn.ensemble import IsolationForest
import pandas as pd
from scipy.stats import combine_pvalues
from .base_agent import BaseAgent, AgentMessage


class RiskAssessor(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id=agent_id, agent_type="risk_assessor", config=config)
        self.risk_cache = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.risk_weights = {
            "transaction_score": 0.3,
            "pattern_score": 0.3,
            "historical_score": 0.2,
            "behavioral_score": 0.2,
        }
        self.risk_thresholds = {"critical": 0.9, "high": 0.7, "medium": 0.4, "low": 0.2}
        self.historical_patterns = defaultdict(list)
        self.behavioral_profiles = {}

    async def initialize(self) -> bool:
        try:
            self.register_message_handler("assess_risk", self._handle_risk_assessment)
            self.register_message_handler("update_weights", self._handle_weight_update)
            self.register_message_handler(
                "update_thresholds", self._handle_threshold_update
            )
            return True
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            return False

    def _calculate_behavioral_score(
        self, customer_id: str, transaction_data: Dict[str, Any]
    ) -> float:
        profile = self.behavioral_profiles.get(customer_id, {})
        if not profile:
            return 0.5

        current_hour = datetime.fromisoformat(transaction_data["timestamp"]).hour
        current_amount = float(transaction_data["amount"])
        current_location = (
            float(transaction_data.get("latitude", 0)),
            float(transaction_data.get("longitude", 0)),
        )

        time_score = abs(profile.get("typical_hours", [12]) - current_hour) / 12
        amount_score = abs(
            current_amount - profile.get("avg_amount", current_amount)
        ) / max(profile.get("avg_amount", current_amount), 1)

        location_score = 1.0
        if profile.get("typical_locations"):
            min_distance = float("inf")
            for loc in profile["typical_locations"]:
                dist = np.sqrt(
                    (loc[0] - current_location[0]) ** 2
                    + (loc[1] - current_location[1]) ** 2
                )
                min_distance = min(min_distance, dist)
            location_score = min(1.0, min_distance / 100)

        return (time_score + amount_score + location_score) / 3

    def _calculate_historical_score(
        self, customer_id: str, transaction_data: Dict[str, Any]
    ) -> float:
        patterns = self.historical_patterns[customer_id]
        if not patterns:
            return 0.5

        recent_patterns = [
            p
            for p in patterns
            if datetime.now() - datetime.fromisoformat(p["timestamp"])
            < timedelta(days=30)
        ]

        if not recent_patterns:
            return 0.5

        risk_scores = [p["risk_score"] for p in recent_patterns]
        trend_factor = np.polyfit(range(len(risk_scores)), risk_scores, 1)[0]
        current_amount = float(transaction_data["amount"])
        amount_stats = [float(p["amount"]) for p in recent_patterns]

        amount_zscore = (current_amount - np.mean(amount_stats)) / max(
            np.std(amount_stats), 1
        )
        pattern_score = np.mean(risk_scores) + trend_factor

        return min(1.0, max(0.0, (pattern_score + abs(amount_zscore)) / 2))

    def _analyze_transaction_patterns(self, patterns: Dict[str, Any]) -> float:
        if not patterns:
            return 0.5

        pattern_scores = []

        for pattern_type, pattern_list in patterns.items():
            if pattern_type == "velocity":
                for pattern in pattern_list:
                    time_diff = pattern["time_difference"]
                    amount = pattern["total_amount"]
                    score = min(1.0, (amount / 10000) * (1 / max(time_diff / 3600, 1)))
                    pattern_scores.append(score)

            elif pattern_type == "location":
                for pattern in pattern_list:
                    radius = pattern.get("radius", 0)
                    txn_count = len(pattern["transactions"])
                    score = min(1.0, (txn_count / 5) * (1 / max(radius / 100, 1)))
                    pattern_scores.append(score)

            elif pattern_type == "amount":
                for pattern in pattern_list:
                    z_scores = pattern["z_scores"].values()
                    score = min(1.0, max(abs(z) for z in z_scores) / 3)
                    pattern_scores.append(score)

            elif pattern_type == "graph":
                for pattern in pattern_list:
                    density = pattern.get("density", 0)
                    txn_count = len(pattern["transactions"])
                    score = min(1.0, density * (txn_count / 5))
                    pattern_scores.append(score)

        return np.mean(pattern_scores) if pattern_scores else 0.5

    async def _handle_risk_assessment(self, message: AgentMessage) -> Dict[str, Any]:
        data = message.content
        success, result = await self.process(data)

        if success:
            customer_id = data["transaction_data"].get("customer_id")
            if customer_id:
                self.risk_cache[customer_id] = {
                    "timestamp": datetime.utcnow(),
                    "risk_assessment": result,
                }
            return {"status": "success", "assessment": result}
        return {"status": "error", "message": "Risk assessment failed"}

    async def _handle_weight_update(self, message: AgentMessage) -> Dict[str, Any]:
        new_weights = message.content.get("weights", {})
        if sum(new_weights.values()) != 1.0:
            return {"status": "error", "message": "Weights must sum to 1.0"}
        self.risk_weights.update(new_weights)
        return {"status": "success", "message": "Weights updated"}

    async def _handle_threshold_update(self, message: AgentMessage) -> Dict[str, Any]:
        new_thresholds = message.content.get("thresholds", {})
        self.risk_thresholds.update(new_thresholds)
        return {"status": "success", "message": "Thresholds updated"}

    def _combine_risk_scores(self, scores: Dict[str, float]) -> float:
        weighted_scores = []
        weights = []

        for score_type, score in scores.items():
            weight = self.risk_weights.get(score_type, 0.25)
            weighted_scores.append(score)
            weights.append(weight)

        return np.average(weighted_scores, weights=weights)

    def _determine_risk_level(self, risk_score: float) -> str:
        if risk_score >= self.risk_thresholds["critical"]:
            return "critical"
        elif risk_score >= self.risk_thresholds["high"]:
            return "high"
        elif risk_score >= self.risk_thresholds["medium"]:
            return "medium"
        elif risk_score >= self.risk_thresholds["low"]:
            return "low"
        return "minimal"

    async def process(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        try:
            transaction_data = data["transaction_data"]
            transaction_score = data.get("transaction_score", 0.5)
            patterns = data.get("patterns", {})
            customer_id = transaction_data.get("customer_id", "")

            pattern_score = self._analyze_transaction_patterns(patterns)
            historical_score = self._calculate_historical_score(
                customer_id, transaction_data
            )
            behavioral_score = self._calculate_behavioral_score(
                customer_id, transaction_data
            )

            risk_scores = {
                "transaction_score": transaction_score,
                "pattern_score": pattern_score,
                "historical_score": historical_score,
                "behavioral_score": behavioral_score,
            }

            combined_risk_score = self._combine_risk_scores(risk_scores)
            risk_level = self._determine_risk_level(combined_risk_score)

            feature_vector = [
                transaction_score,
                pattern_score,
                historical_score,
                behavioral_score,
            ]

            anomaly_score = self.anomaly_detector.fit_predict([feature_vector])[0]

            assessment = {
                "risk_score": combined_risk_score,
                "risk_level": risk_level,
                "component_scores": risk_scores,
                "anomaly_detected": anomaly_score == -1,
                "confidence_score": 1 - np.std(list(risk_scores.values())),
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "weights_used": self.risk_weights,
                    "thresholds_used": self.risk_thresholds,
                },
            }

            if customer_id:
                self.historical_patterns[customer_id].append(
                    {
                        "timestamp": transaction_data["timestamp"],
                        "amount": transaction_data["amount"],
                        "risk_score": combined_risk_score,
                    }
                )

            return True, assessment

        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return False, {"error": str(e)}

    async def cleanup(self) -> bool:
        try:
            self.risk_cache.clear()
            self.historical_patterns.clear()
            self.behavioral_profiles.clear()
            return True
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
            return False
