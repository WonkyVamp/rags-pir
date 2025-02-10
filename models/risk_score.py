from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class RiskCategory(Enum):
    TRANSACTION = "transaction"
    BEHAVIORAL = "behavioral"
    HISTORICAL = "historical"
    NETWORK = "network"
    DEVICE = "device"
    LOCATION = "location"
    PATTERN = "pattern"


class RiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskFactor:
    name: str
    category: RiskCategory
    weight: float
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskThresholds:
    minimal: float = 0.2
    low: float = 0.4
    medium: float = 0.6
    high: float = 0.8
    critical: float = 0.9


@dataclass
class RiskHistory:
    scores: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    factors: List[List[RiskFactor]] = field(default_factory=list)
    levels: List[RiskLevel] = field(default_factory=list)


class RiskScore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = RiskThresholds(**config.get("thresholds", {}))
        self.category_weights = self._initialize_category_weights()
        self.risk_history = RiskHistory()
        self.current_factors: List[RiskFactor] = []
        self.last_updated: datetime = datetime.utcnow()
        self.decay_rate = config.get("decay_rate", 0.1)
        self.smoothing_factor = config.get("smoothing_factor", 0.3)

    def _initialize_category_weights(self) -> Dict[RiskCategory, float]:
        default_weights = {
            RiskCategory.TRANSACTION: 0.25,
            RiskCategory.BEHAVIORAL: 0.20,
            RiskCategory.HISTORICAL: 0.15,
            RiskCategory.NETWORK: 0.15,
            RiskCategory.DEVICE: 0.10,
            RiskCategory.LOCATION: 0.10,
            RiskCategory.PATTERN: 0.05,
        }
        return {
            RiskCategory(k): v
            for k, v in self.config.get("category_weights", default_weights).items()
        }

    def add_risk_factor(self, factor: RiskFactor):
        self.current_factors.append(factor)
        self.last_updated = datetime.utcnow()

    def calculate_risk_score(self) -> Tuple[float, RiskLevel]:
        if not self.current_factors:
            return 0.0, RiskLevel.MINIMAL

        category_scores = defaultdict(list)
        for factor in self.current_factors:
            category_scores[factor.category].append(factor.value * factor.weight)

        weighted_scores = []
        for category, scores in category_scores.items():
            category_weight = self.category_weights[category]
            category_score = np.mean(scores)
            weighted_scores.append(category_score * category_weight)

        final_score = np.sum(weighted_scores)
        risk_level = self._determine_risk_level(final_score)

        self._update_history(final_score, risk_level)
        return final_score, risk_level

    def _determine_risk_level(self, score: float) -> RiskLevel:
        if score >= self.thresholds.critical:
            return RiskLevel.CRITICAL
        elif score >= self.thresholds.high:
            return RiskLevel.HIGH
        elif score >= self.thresholds.medium:
            return RiskLevel.MEDIUM
        elif score >= self.thresholds.low:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _update_history(self, score: float, level: RiskLevel):
        self.risk_history.scores.append(score)
        self.risk_history.timestamps.append(datetime.utcnow())
        self.risk_history.factors.append(self.current_factors.copy())
        self.risk_history.levels.append(level)

        # Keep only last 1000 entries
        max_history = 1000
        if len(self.risk_history.scores) > max_history:
            self.risk_history.scores = self.risk_history.scores[-max_history:]
            self.risk_history.timestamps = self.risk_history.timestamps[-max_history:]
            self.risk_history.factors = self.risk_history.factors[-max_history:]
            self.risk_history.levels = self.risk_history.levels[-max_history:]

    def calculate_trend(self, window_hours: int = 24) -> float:
        if len(self.risk_history.scores) < 2:
            return 0.0

        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        recent_scores = [
            score
            for score, timestamp in zip(
                self.risk_history.scores, self.risk_history.timestamps
            )
            if timestamp >= cutoff_time
        ]

        if len(recent_scores) < 2:
            return 0.0

        return np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

    def get_risk_factors_by_category(self) -> Dict[RiskCategory, List[RiskFactor]]:
        factors_by_category = defaultdict(list)
        for factor in self.current_factors:
            factors_by_category[factor.category].append(factor)
        return dict(factors_by_category)

    def get_highest_risk_factors(self, n: int = 5) -> List[RiskFactor]:
        return sorted(
            self.current_factors, key=lambda x: x.value * x.weight, reverse=True
        )[:n]

    def calculate_velocity_risk(
        self, recent_transactions: List[Dict[str, Any]], window_minutes: int = 60
    ) -> float:
        if not recent_transactions:
            return 0.0

        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        relevant_transactions = [
            tx
            for tx in recent_transactions
            if datetime.fromisoformat(tx["timestamp"]) >= cutoff_time
        ]

        if not relevant_transactions:
            return 0.0

        transaction_count = len(relevant_transactions)
        total_amount = sum(float(tx["amount"]) for tx in relevant_transactions)

        count_risk = min(
            transaction_count / self.config.get("max_transactions_per_hour", 10), 1.0
        )
        amount_risk = min(
            total_amount / self.config.get("max_amount_per_hour", 10000), 1.0
        )

        return max(count_risk, amount_risk)

    def calculate_location_risk(
        self,
        current_location: Tuple[float, float],
        location_history: List[Tuple[float, float]],
    ) -> float:
        if not location_history:
            return 0.5

        def haversine_distance(
            loc1: Tuple[float, float], loc2: Tuple[float, float]
        ) -> float:
            R = 6371  # Earth's radius in kilometers
            lat1, lon1 = loc1
            lat2, lon2 = loc2

            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(np.radians(lat1))
                * np.cos(np.radians(lat2))
                * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c

        distances = [
            haversine_distance(current_location, hist_loc)
            for hist_loc in location_history
        ]

        min_distance = min(distances)
        max_normal_distance = self.config.get("max_normal_distance_km", 100)

        return min(min_distance / max_normal_distance, 1.0)

    def calculate_behavioral_risk(
        self, current_behavior: Dict[str, Any], historical_behavior: Dict[str, Any]
    ) -> float:
        risk_scores = []

        # Time pattern risk
        if "hour" in current_behavior and "typical_hours" in historical_behavior:
            current_hour = current_behavior["hour"]
            typical_hours = historical_behavior["typical_hours"]
            time_risk = min(abs(current_hour - np.mean(typical_hours)) / 12, 1.0)
            risk_scores.append(time_risk)

        # Amount pattern risk
        if "amount" in current_behavior and "typical_amounts" in historical_behavior:
            current_amount = current_behavior["amount"]
            typical_amounts = historical_behavior["typical_amounts"]
            amount_mean = np.mean(typical_amounts)
            amount_std = (
                np.std(typical_amounts) if len(typical_amounts) > 1 else amount_mean
            )
            amount_risk = min(
                (
                    abs(current_amount - amount_mean) / (3 * amount_std)
                    if amount_std > 0
                    else 0
                ),
                1.0,
            )
            risk_scores.append(amount_risk)

        return np.mean(risk_scores) if risk_scores else 0.5

    def apply_temporal_decay(self, days: int = 30):
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        for factor in self.current_factors:
            time_diff = (datetime.utcnow() - factor.timestamp).total_seconds()
            decay = np.exp(-self.decay_rate * time_diff / (24 * 3600))
            factor.value *= decay

    def get_smoothed_risk_score(self, window_size: int = 5) -> float:
        if len(self.risk_history.scores) < window_size:
            return self.risk_history.scores[-1] if self.risk_history.scores else 0.0

        recent_scores = self.risk_history.scores[-window_size:]
        weights = [
            self.smoothing_factor * (1 - self.smoothing_factor) ** (window_size - i - 1)
            for i in range(window_size)
        ]
        weights = [w / sum(weights) for w in weights]

        return sum(score * weight for score, weight in zip(recent_scores, weights))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_score": self.calculate_risk_score()[0],
            "risk_level": self.calculate_risk_score()[1].value,
            "trend": self.calculate_trend(),
            "high_risk_factors": [
                {
                    "name": f.name,
                    "category": f.category.value,
                    "value": f.value,
                    "weight": f.weight,
                }
                for f in self.get_highest_risk_factors()
            ],
            "last_updated": self.last_updated.isoformat(),
            "smoothed_score": self.get_smoothed_risk_score(),
        }
