from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import asyncio
from collections import defaultdict
import json

from .base_agent import BaseAgent, AgentMessage


class TransactionFeatures:
    """Extracts and computes features from transaction data"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            "amount_scaled",
            "hour_of_day",
            "day_of_week",
            "distance_from_last_transaction",
            "velocity_1h",
            "velocity_24h",
            "amount_deviation_from_mean",
            "frequency_merchant_7d",
            "frequency_category_7d",
        ]

    def compute_temporal_features(self, transaction_time: datetime) -> Dict[str, float]:
        """Compute time-based features"""
        return {
            "hour_of_day": transaction_time.hour / 24.0,  # Normalized
            "day_of_week": transaction_time.weekday() / 6.0,  # Normalized
        }

    def compute_velocity_features(
        self,
        current_transaction: Dict[str, Any],
        transaction_history: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute transaction velocity features"""
        current_time = datetime.fromisoformat(current_transaction["timestamp"])
        amount = float(current_transaction["amount"])

        # 1-hour velocity
        one_hour_ago = current_time - timedelta(hours=1)
        transactions_1h = [
            t
            for t in transaction_history
            if current_time - datetime.fromisoformat(t["timestamp"])
            <= timedelta(hours=1)
        ]
        velocity_1h = sum(float(t["amount"]) for t in transactions_1h)

        # 24-hour velocity
        twenty_four_hours_ago = current_time - timedelta(hours=24)
        transactions_24h = [
            t
            for t in transaction_history
            if current_time - datetime.fromisoformat(t["timestamp"])
            <= timedelta(hours=24)
        ]
        velocity_24h = sum(float(t["amount"]) for t in transactions_24h)

        return {"velocity_1h": velocity_1h, "velocity_24h": velocity_24h}

    def compute_location_features(
        self,
        current_transaction: Dict[str, Any],
        last_transaction: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute location-based features"""
        if not last_transaction:
            return {"distance_from_last_transaction": 0.0}

        def haversine_distance(
            lat1: float, lon1: float, lat2: float, lon2: float
        ) -> float:
            R = 6371  # Earth's radius in kilometers

            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c

            return distance

        try:
            current_lat = float(current_transaction.get("latitude", 0))
            current_lon = float(current_transaction.get("longitude", 0))
            last_lat = float(last_transaction.get("latitude", 0))
            last_lon = float(last_transaction.get("longitude", 0))

            distance = haversine_distance(current_lat, current_lon, last_lat, last_lon)
        except (ValueError, TypeError):
            distance = 0.0

        return {"distance_from_last_transaction": distance}

    def compute_merchant_features(
        self,
        current_transaction: Dict[str, Any],
        transaction_history: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute merchant-related features"""
        current_time = datetime.fromisoformat(current_transaction["timestamp"])
        seven_days_ago = current_time - timedelta(days=7)

        # Filter transactions from last 7 days
        recent_transactions = [
            t
            for t in transaction_history
            if current_time - datetime.fromisoformat(t["timestamp"])
            <= timedelta(days=7)
        ]

        # Count merchant frequency
        merchant_id = current_transaction.get("merchant_id", "")
        merchant_frequency = sum(
            1 for t in recent_transactions if t.get("merchant_id", "") == merchant_id
        )

        # Count category frequency
        category = current_transaction.get("category", "")
        category_frequency = sum(
            1 for t in recent_transactions if t.get("category", "") == category
        )

        return {
            "frequency_merchant_7d": merchant_frequency,
            "frequency_category_7d": category_frequency,
        }

    def compute_amount_features(
        self,
        current_transaction: Dict[str, Any],
        transaction_history: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute amount-related features"""
        amount = float(current_transaction["amount"])

        if not transaction_history:
            return {"amount_scaled": amount, "amount_deviation_from_mean": 0.0}

        historical_amounts = [float(t["amount"]) for t in transaction_history]
        mean_amount = np.mean(historical_amounts)
        std_amount = np.std(historical_amounts) if len(historical_amounts) > 1 else 1.0

        # Scale amount and compute deviation
        amount_scaled = (
            self.scaler.fit_transform([[amount]])[0][0]
            if historical_amounts
            else amount
        )

        amount_deviation = (amount - mean_amount) / std_amount if std_amount != 0 else 0

        return {
            "amount_scaled": amount_scaled,
            "amount_deviation_from_mean": amount_deviation,
        }


class TransactionAnalyzer(BaseAgent):
    """Agent responsible for analyzing individual transactions for fraud indicators"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(
            agent_id=agent_id, agent_type="transaction_analyzer", config=config
        )
        self.feature_extractor = TransactionFeatures()
        self.transaction_history = defaultdict(list)  # Customer ID -> transactions
        self.model = None
        self.risk_thresholds = {"low": 0.3, "medium": 0.6, "high": 0.8}

    async def initialize(self) -> bool:
        """Initialize the transaction analyzer agent"""
        try:
            # Load pre-trained model if specified in config
            model_path = self.config.get("model_path")
            if model_path:
                self.model = joblib.load(model_path)
                self.logger.info(f"Loaded model from {model_path}")

            # Register specialized message handlers
            self.register_message_handler(
                "analyze_transaction", self._handle_transaction_analysis
            )
            self.register_message_handler(
                "update_risk_thresholds", self._handle_risk_threshold_update
            )
            self.register_message_handler("clear_history", self._handle_clear_history)

            return True

        except Exception as e:
            self.logger.error(f"Error initializing TransactionAnalyzer: {str(e)}")
            return False

    async def _handle_transaction_analysis(
        self, message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle incoming transaction analysis requests"""
        transaction_data = message.content.get("transaction", {})
        customer_id = transaction_data.get("customer_id")

        if not customer_id:
            return {
                "status": "error",
                "message": "Missing customer_id in transaction data",
            }

        success, result = await self.process(transaction_data)

        if success:
            # Update transaction history
            self.transaction_history[customer_id].append(transaction_data)
            # Keep only last 1000 transactions per customer
            self.transaction_history[customer_id] = self.transaction_history[
                customer_id
            ][-1000:]

        return {"status": "success" if success else "error", "analysis_result": result}

    async def _handle_risk_threshold_update(
        self, message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle updates to risk thresholds"""
        new_thresholds = message.content.get("thresholds", {})

        if not all(k in new_thresholds for k in ["low", "medium", "high"]):
            return {"status": "error", "message": "Invalid threshold format"}

        self.risk_thresholds.update(new_thresholds)
        return {"status": "success", "message": "Risk thresholds updated"}

    async def _handle_clear_history(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle clearing transaction history"""
        customer_id = message.content.get("customer_id")

        if customer_id:
            self.transaction_history.pop(customer_id, None)
            message = f"Cleared history for customer {customer_id}"
        else:
            self.transaction_history.clear()
            message = "Cleared all transaction history"

        return {"status": "success", "message": message}

    async def process(
        self, transaction_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Process a single transaction and return analysis results"""
        try:
            customer_id = transaction_data.get("customer_id")
            customer_history = self.transaction_history.get(customer_id, [])

            # Extract features
            features = {}

            # Temporal features
            timestamp = datetime.fromisoformat(transaction_data["timestamp"])
            features.update(self.feature_extractor.compute_temporal_features(timestamp))

            # Velocity features
            features.update(
                self.feature_extractor.compute_velocity_features(
                    transaction_data, customer_history
                )
            )

            # Location features
            last_transaction = customer_history[-1] if customer_history else None
            features.update(
                self.feature_extractor.compute_location_features(
                    transaction_data, last_transaction
                )
            )

            # Merchant features
            features.update(
                self.feature_extractor.compute_merchant_features(
                    transaction_data, customer_history
                )
            )

            # Amount features
            features.update(
                self.feature_extractor.compute_amount_features(
                    transaction_data, customer_history
                )
            )

            # Prepare feature vector
            feature_vector = [
                features[name] for name in self.feature_extractor.feature_names
            ]

            # Calculate risk score
            risk_score = 0.0
            if self.model:
                risk_score = float(self.model.predict_proba([feature_vector])[0][1])
            else:
                # Fallback heuristic if no model is loaded
                amount_factor = min(features["amount_scaled"] / 10.0, 1.0)
                velocity_factor = min(features["velocity_24h"] / 10000.0, 1.0)
                distance_factor = min(
                    features["distance_from_last_transaction"] / 1000.0, 1.0
                )
                risk_score = (amount_factor + velocity_factor + distance_factor) / 3

            # Determine risk level
            risk_level = "low"
            if risk_score >= self.risk_thresholds["high"]:
                risk_level = "high"
            elif risk_score >= self.risk_thresholds["medium"]:
                risk_level = "medium"

            analysis_result = {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "features": features,
                "timestamp": datetime.utcnow().isoformat(),
                "transaction_id": transaction_data.get("transaction_id", ""),
                "customer_id": customer_id,
            }

            # Add feature importance if model supports it
            if hasattr(self.model, "feature_importances_"):
                analysis_result["feature_importance"] = dict(
                    zip(
                        self.feature_extractor.feature_names,
                        self.model.feature_importances_,
                    )
                )

            return True, analysis_result

        except Exception as e:
            self.logger.error(f"Error processing transaction: {str(e)}")
            return False, {"error": str(e)}

    async def cleanup(self) -> bool:
        """Cleanup resources when shutting down"""
        try:
            # Save any necessary state or clear caches
            self.transaction_history.clear()
            self.model = None
            return True
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return False
