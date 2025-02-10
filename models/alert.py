import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import uuid4


class AlertStatus(Enum):
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class AlertPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertType(Enum):
    SUSPICIOUS_TRANSACTION = "suspicious_transaction"
    PATTERN_DETECTED = "pattern_detected"
    VELOCITY_BREACH = "velocity_breach"
    LOCATION_ANOMALY = "location_anomaly"
    DEVICE_ANOMALY = "device_anomaly"
    BEHAVIORAL_CHANGE = "behavioral_change"
    NETWORK_PATTERN = "network_pattern"
    RISK_SCORE_CHANGE = "risk_score_change"


@dataclass
class AlertTrigger:
    trigger_type: str
    threshold: float
    value: float
    rule_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertEvidence:
    evidence_type: str
    evidence_data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertAction:
    action_type: str
    performed_by: str
    timestamp: datetime
    result: str
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationInfo:
    level: int
    escalated_by: str
    escalated_to: str
    reason: str
    timestamp: datetime
    previous_assignee: Optional[str] = None
    due_time: Optional[datetime] = None


@dataclass
class ResolutionDetails:
    resolution_type: str
    resolved_by: str
    timestamp: datetime
    actions_taken: List[str]
    notes: Optional[str] = None
    feedback: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    alert_id: str
    customer_id: str
    transaction_id: Optional[str]
    alert_type: AlertType
    priority: AlertPriority
    status: AlertStatus
    created_at: datetime
    triggers: List[AlertTrigger]
    evidence: List[AlertEvidence]
    actions: List[AlertAction] = field(default_factory=list)
    assignee: Optional[str] = None
    escalation: Optional[EscalationInfo] = None
    resolution: Optional[ResolutionDetails] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    linked_alerts: List[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expiration: Optional[datetime] = None

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = str(uuid4())
        if not self.expiration:
            self.set_default_expiration()

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

    def set_default_expiration(self):
        priority_expiration = {
            AlertPriority.CRITICAL: timedelta(hours=2),
            AlertPriority.HIGH: timedelta(hours=4),
            AlertPriority.MEDIUM: timedelta(hours=12),
            AlertPriority.LOW: timedelta(hours=24),
        }
        self.expiration = datetime.utcnow() + priority_expiration[self.priority]

    def update_status(
        self, new_status: AlertStatus, updated_by: str, notes: Optional[str] = None
    ):
        action = AlertAction(
            action_type="status_change",
            performed_by=updated_by,
            timestamp=datetime.utcnow(),
            result=f"Status changed from {self.status.value} to {new_status.value}",
            notes=notes,
        )
        self.actions.append(action)
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def assign_alert(
        self, assignee: str, assigned_by: str, notes: Optional[str] = None
    ):
        previous_assignee = self.assignee
        self.assignee = assignee

        action = AlertAction(
            action_type="assignment",
            performed_by=assigned_by,
            timestamp=datetime.utcnow(),
            result=f"Assigned to {assignee}",
            notes=notes,
            metadata={"previous_assignee": previous_assignee},
        )
        self.actions.append(action)
        self.update_status(AlertStatus.ASSIGNED, assigned_by)

    def escalate_alert(
        self,
        escalation_info: EscalationInfo,
        escalated_by: str,
        notes: Optional[str] = None,
    ):
        self.escalation = escalation_info
        action = AlertAction(
            action_type="escalation",
            performed_by=escalated_by,
            timestamp=datetime.utcnow(),
            result=f"Escalated to level {escalation_info.level}",
            notes=notes,
            metadata={"escalation_info": asdict(escalation_info)},
        )
        self.actions.append(action)
        self.update_status(AlertStatus.ESCALATED, escalated_by)

    def add_evidence(self, evidence: AlertEvidence):
        self.evidence.append(evidence)
        self.updated_at = datetime.utcnow()

    def resolve_alert(self, resolution_details: ResolutionDetails):
        self.resolution = resolution_details
        action = AlertAction(
            action_type="resolution",
            performed_by=resolution_details.resolved_by,
            timestamp=resolution_details.timestamp,
            result=f"Alert resolved: {resolution_details.resolution_type}",
            notes=resolution_details.notes,
        )
        self.actions.append(action)
        self.update_status(AlertStatus.RESOLVED, resolution_details.resolved_by)

    def link_alert(self, related_alert_id: str):
        if related_alert_id not in self.linked_alerts:
            self.linked_alerts.append(related_alert_id)
            self.updated_at = datetime.utcnow()

    def add_action(self, action: AlertAction):
        self.actions.append(action)
        self.updated_at = datetime.utcnow()

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expiration if self.expiration else False

    def time_to_expiration(self) -> Optional[timedelta]:
        if not self.expiration:
            return None
        return self.expiration - datetime.utcnow()

    def requires_immediate_attention(self) -> bool:
        if self.status in [
            AlertStatus.RESOLVED,
            AlertStatus.CLOSED,
            AlertStatus.FALSE_POSITIVE,
        ]:
            return False

        if self.priority == AlertPriority.CRITICAL:
            return True

        if self.is_expired():
            return True

        if self.priority == AlertPriority.HIGH and not self.assignee:
            return True

        return False

    @classmethod
    def from_dict(cls, data: Dict) -> "Alert":
        data["alert_type"] = AlertType(data["alert_type"])
        data["priority"] = AlertPriority(data["priority"])
        data["status"] = AlertStatus(data["status"])

        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        if data.get("expiration"):
            data["expiration"] = datetime.fromisoformat(data["expiration"])

        triggers = []
        for trigger_data in data["triggers"]:
            triggers.append(AlertTrigger(**trigger_data))
        data["triggers"] = triggers

        evidence_list = []
        for evidence_data in data["evidence"]:
            evidence_data["timestamp"] = datetime.fromisoformat(
                evidence_data["timestamp"]
            )
            evidence_list.append(AlertEvidence(**evidence_data))
        data["evidence"] = evidence_list

        actions = []
        for action_data in data["actions"]:
            action_data["timestamp"] = datetime.fromisoformat(action_data["timestamp"])
            actions.append(AlertAction(**action_data))
        data["actions"] = actions

        if data.get("escalation"):
            data["escalation"] = EscalationInfo(**data["escalation"])

        if data.get("resolution"):
            data["resolution"] = ResolutionDetails(**data["resolution"])

        return cls(**data)

    def get_action_timeline(self) -> List[Dict[str, Any]]:
        timeline = []

        for action in self.actions:
            timeline.append(
                {
                    "timestamp": action.timestamp,
                    "type": "action",
                    "data": asdict(action),
                }
            )

        for evidence in self.evidence:
            timeline.append(
                {
                    "timestamp": evidence.timestamp,
                    "type": "evidence",
                    "data": asdict(evidence),
                }
            )

        if self.escalation:
            timeline.append(
                {
                    "timestamp": self.escalation.timestamp,
                    "type": "escalation",
                    "data": asdict(self.escalation),
                }
            )

        if self.resolution:
            timeline.append(
                {
                    "timestamp": self.resolution.timestamp,
                    "type": "resolution",
                    "data": asdict(self.resolution),
                }
            )

        return sorted(timeline, key=lambda x: x["timestamp"])
