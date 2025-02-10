from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from uuid import uuid4
from enum import Enum
import json


class TransactionStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    FLAGGED = "flagged"
    UNDER_REVIEW = "under_review"


class TransactionType(Enum):
    PURCHASE = "purchase"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    REFUND = "refund"
    PAYMENT = "payment"


class PaymentMethod(Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CRYPTO = "cryptocurrency"
    OTHER = "other"


@dataclass
class Location:
    latitude: float
    longitude: float
    city: Optional[str] = None
    country: Optional[str] = None
    ip_address: Optional[str] = None
    timezone: Optional[str] = None


@dataclass
class Device:
    device_id: str
    device_type: str
    os: Optional[str] = None
    browser: Optional[str] = None
    ip_address: Optional[str] = None
    is_trusted: bool = False


@dataclass
class MerchantInfo:
    merchant_id: str
    name: str
    category: str
    mcc_code: Optional[str] = None
    location: Optional[Location] = None
    risk_level: Optional[str] = None


@dataclass
class PaymentDetails:
    method: PaymentMethod
    card_last4: Optional[str] = None
    card_issuer: Optional[str] = None
    card_type: Optional[str] = None
    wallet_provider: Optional[str] = None
    bank_name: Optional[str] = None
    is_international: bool = False


@dataclass
class RiskIndicators:
    velocity_check: Optional[bool] = None
    ip_risk: Optional[float] = None
    device_risk: Optional[float] = None
    amount_risk: Optional[float] = None
    location_risk: Optional[float] = None
    overall_risk: Optional[float] = None
    rules_triggered: List[str] = None

    def __post_init__(self):
        if self.rules_triggered is None:
            self.rules_triggered = []


@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    merchant_info: MerchantInfo
    amount: float
    currency: str
    timestamp: datetime
    status: TransactionStatus
    transaction_type: TransactionType
    payment_details: PaymentDetails
    location: Location
    device: Optional[Device]
    risk_indicators: Optional[RiskIndicators] = None
    metadata: Optional[Dict] = None
    created_at: datetime = None
    updated_at: datetime = None
    processed_by: Optional[List[str]] = None

    def __post_init__(self):
        if not self.transaction_id:
            self.transaction_id = str(uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.processed_by is None:
            self.processed_by = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            k: (
                v.value
                if isinstance(v, Enum)
                else (
                    v.isoformat()
                    if isinstance(v, datetime)
                    else asdict(v) if hasattr(v, "__dataclass_fields__") else v
                )
            )
            for k, v in asdict(self).items()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def update_status(self, new_status: TransactionStatus):
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def update_risk_indicators(self, risk_indicators: RiskIndicators):
        self.risk_indicators = risk_indicators
        self.updated_at = datetime.utcnow()

    def add_processor(self, processor_id: str):
        if processor_id not in self.processed_by:
            self.processed_by.append(processor_id)
            self.updated_at = datetime.utcnow()

    def is_high_risk(self) -> bool:
        if not self.risk_indicators or self.risk_indicators.overall_risk is None:
            return False
        return self.risk_indicators.overall_risk >= 0.7

    def is_international(self) -> bool:
        return self.payment_details.is_international or (
            self.location
            and self.merchant_info.location
            and self.location.country != self.merchant_info.location.country
        )

    def get_age(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds()

    @classmethod
    def from_dict(cls, data: Dict) -> "Transaction":
        data["status"] = TransactionStatus(data["status"])
        data["transaction_type"] = TransactionType(data["transaction_type"])
        data["payment_details"]["method"] = PaymentMethod(
            data["payment_details"]["method"]
        )

        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        data["merchant_info"] = MerchantInfo(**data["merchant_info"])
        data["location"] = Location(**data["location"])

        if data.get("device"):
            data["device"] = Device(**data["device"])
        if data.get("risk_indicators"):
            data["risk_indicators"] = RiskIndicators(**data["risk_indicators"])

        return cls(**data)

    @staticmethod
    def validate_amount(amount: float) -> bool:
        return amount > 0

    @staticmethod
    def validate_currency(currency: str) -> bool:
        valid_currencies = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD"}
        return currency.upper() in valid_currencies
