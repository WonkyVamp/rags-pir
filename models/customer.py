from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
from uuid import uuid4


class CustomerStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BLOCKED = "blocked"
    UNDER_REVIEW = "under_review"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ContactInfo:
    email: str
    phone: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    preferred_contact: str = "email"
    verified_email: bool = False
    verified_phone: bool = False


@dataclass
class AuthenticationProfile:
    password_hash: str
    last_login: Optional[datetime] = None
    last_password_change: Optional[datetime] = None
    two_factor_enabled: bool = False
    failed_login_attempts: int = 0
    security_questions: Dict[str, str] = field(default_factory=dict)
    trusted_devices: List[Dict[str, Any]] = field(default_factory=list)
    authentication_methods: List[str] = field(default_factory=list)


@dataclass
class BehavioralProfile:
    typical_transaction_times: List[int] = field(default_factory=list)
    typical_amounts: List[float] = field(default_factory=list)
    frequent_merchants: Dict[str, int] = field(default_factory=dict)
    frequent_locations: List[Dict[str, float]] = field(default_factory=list)
    device_fingerprints: Set[str] = field(default_factory=set)
    usual_ip_addresses: Set[str] = field(default_factory=set)
    transaction_velocity: Dict[str, float] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskProfile:
    risk_level: RiskLevel
    risk_score: float
    risk_factors: List[str] = field(default_factory=list)
    monitoring_level: str = "standard"
    last_assessment: datetime = field(default_factory=datetime.utcnow)
    risk_history: List[Dict[str, Any]] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    review_required: bool = False


@dataclass
class TransactionHistory:
    last_transaction: Optional[datetime] = None
    total_transactions: int = 0
    total_amount: float = 0.0
    average_amount: float = 0.0
    transaction_count_30d: int = 0
    transaction_amount_30d: float = 0.0
    declined_transactions: int = 0
    flagged_transactions: int = 0
    countries_traded: Set[str] = field(default_factory=set)
    merchant_categories: Dict[str, int] = field(default_factory=dict)


@dataclass
class Customer:
    customer_id: str
    username: str
    contact_info: ContactInfo
    status: CustomerStatus
    created_at: datetime
    auth_profile: AuthenticationProfile
    behavioral_profile: BehavioralProfile
    risk_profile: RiskProfile
    transaction_history: TransactionHistory
    metadata: Dict[str, Any] = field(default_factory=dict)
    kyc_verified: bool = False
    last_updated: datetime = field(default_factory=datetime.utcnow)
    segments: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.customer_id:
            self.customer_id = str(uuid4())

    def to_dict(self) -> Dict:
        return {
            k: (
                v.value
                if isinstance(v, Enum)
                else (
                    v.isoformat()
                    if isinstance(v, datetime)
                    else (
                        list(v)
                        if isinstance(v, set)
                        else asdict(v) if hasattr(v, "__dataclass_fields__") else v
                    )
                )
            )
            for k, v in asdict(self).items()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def update_risk_profile(self, new_risk_data: Dict[str, Any]):
        current_risk = self.risk_profile.to_dict()
        current_risk["last_assessment"] = datetime.utcnow()
        self.risk_profile.risk_history.append(current_risk)

        self.risk_profile.risk_level = RiskLevel(new_risk_data["risk_level"])
        self.risk_profile.risk_score = new_risk_data["risk_score"]
        self.risk_profile.risk_factors = new_risk_data.get("risk_factors", [])
        self.risk_profile.monitoring_level = new_risk_data.get(
            "monitoring_level", "standard"
        )
        self.risk_profile.restrictions = new_risk_data.get("restrictions", [])
        self.risk_profile.review_required = new_risk_data.get("review_required", False)

        self.last_updated = datetime.utcnow()

    def update_behavioral_profile(self, transaction_data: Dict[str, Any]):
        profile = self.behavioral_profile

        # Update typical transaction times
        hour = datetime.fromisoformat(transaction_data["timestamp"]).hour
        profile.typical_transaction_times.append(hour)
        if len(profile.typical_transaction_times) > 1000:
            profile.typical_transaction_times = profile.typical_transaction_times[
                -1000:
            ]

        # Update typical amounts
        amount = float(transaction_data["amount"])
        profile.typical_amounts.append(amount)
        if len(profile.typical_amounts) > 1000:
            profile.typical_amounts = profile.typical_amounts[-1000:]

        # Update merchant frequency
        merchant_id = transaction_data["merchant_info"]["merchant_id"]
        profile.frequent_merchants[merchant_id] = (
            profile.frequent_merchants.get(merchant_id, 0) + 1
        )

        # Update location data
        location = {
            "lat": transaction_data["location"]["latitude"],
            "lon": transaction_data["location"]["longitude"],
        }
        profile.frequent_locations.append(location)
        if len(profile.frequent_locations) > 100:
            profile.frequent_locations = profile.frequent_locations[-100:]

        # Update device and IP data
        if "device" in transaction_data:
            profile.device_fingerprints.add(transaction_data["device"]["device_id"])
            if transaction_data["device"].get("ip_address"):
                profile.usual_ip_addresses.add(transaction_data["device"]["ip_address"])

        # Update velocity metrics
        current_time = datetime.utcnow()
        for window in ["1h", "24h", "7d"]:
            key = f"velocity_{window}"
            profile.transaction_velocity[key] = self._calculate_velocity(window)

        profile.last_update = current_time
        self.last_updated = current_time

    def update_transaction_history(self, transaction_data: Dict[str, Any]):
        history = self.transaction_history
        amount = float(transaction_data["amount"])
        timestamp = datetime.fromisoformat(transaction_data["timestamp"])

        history.last_transaction = timestamp
        history.total_transactions += 1
        history.total_amount += amount
        history.average_amount = history.total_amount / history.total_transactions

        # Update 30-day metrics
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        if timestamp >= thirty_days_ago:
            history.transaction_count_30d += 1
            history.transaction_amount_30d += amount

        # Update status-based counters
        if transaction_data["status"] == "declined":
            history.declined_transactions += 1
        elif transaction_data["status"] == "flagged":
            history.flagged_transactions += 1

        # Update geographic and merchant data
        if "location" in transaction_data and "country" in transaction_data["location"]:
            history.countries_traded.add(transaction_data["location"]["country"])

        merchant_category = transaction_data["merchant_info"]["category"]
        history.merchant_categories[merchant_category] = (
            history.merchant_categories.get(merchant_category, 0) + 1
        )

    def _calculate_velocity(self, window: str) -> float:
        current_time = datetime.utcnow()
        if window == "1h":
            start_time = current_time - timedelta(hours=1)
        elif window == "24h":
            start_time = current_time - timedelta(days=1)
        elif window == "7d":
            start_time = current_time - timedelta(days=7)
        else:
            raise ValueError(f"Invalid window: {window}")

        count = self.transaction_history.transaction_count_30d
        time_diff = (current_time - start_time).total_seconds()
        return count / time_diff if time_diff > 0 else 0

    def is_high_risk(self) -> bool:
        return self.risk_profile.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def requires_review(self) -> bool:
        return (
            self.risk_profile.review_required
            or self.risk_profile.risk_level == RiskLevel.CRITICAL
            or self.status == CustomerStatus.UNDER_REVIEW
        )

    def has_recent_suspicious_activity(self) -> bool:
        if not self.transaction_history.last_transaction:
            return False

        recent_window = datetime.utcnow() - timedelta(days=7)
        return self.transaction_history.last_transaction >= recent_window and (
            self.transaction_history.flagged_transactions > 0
            or self.risk_profile.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "Customer":
        data["status"] = CustomerStatus(data["status"])
        data["contact_info"] = ContactInfo(**data["contact_info"])
        data["auth_profile"] = AuthenticationProfile(**data["auth_profile"])
        data["behavioral_profile"] = BehavioralProfile(**data["behavioral_profile"])

        risk_data = data["risk_profile"]
        risk_data["risk_level"] = RiskLevel(risk_data["risk_level"])
        data["risk_profile"] = RiskProfile(**risk_data)

        data["transaction_history"] = TransactionHistory(**data["transaction_history"])

        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_updated" in data:
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])

        return cls(**data)

    def validate_contact_info(self) -> bool:
        if not self.contact_info.email:
            return False
        return True  # Add more validation as needed

    def validate_status_transition(self, new_status: CustomerStatus) -> bool:
        invalid_transitions = {
            CustomerStatus.BLOCKED: {CustomerStatus.ACTIVE, CustomerStatus.INACTIVE},
            CustomerStatus.SUSPENDED: {CustomerStatus.ACTIVE},
        }
        return new_status not in invalid_transitions.get(self.status, set())
