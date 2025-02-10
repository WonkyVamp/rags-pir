import json
from datetime import datetime
from typing import Any, Dict, List

from fastapi import HTTPException

from models.alert import Alert, AlertStatus
from models.customer import Customer, CustomerStatus
from models.transaction import Transaction, TransactionStatus
from services.database_service import CollectionName, DatabaseService


class TransactionHandler:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    async def process_transaction(
        self, transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Basic validation
            if not all(
                k in transaction_data
                for k in ["transaction_id", "customer_id", "amount"]
            ):
                raise HTTPException(status_code=400, detail="Missing required fields")

            # Store transaction
            transaction_data["status"] = TransactionStatus.PENDING.value
            transaction_data["created_at"] = datetime.utcnow().isoformat()

            await self.db.insert_one(CollectionName.TRANSACTIONS, transaction_data)

            return {
                "transaction_id": transaction_data["transaction_id"],
                "status": "processed",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_transaction_status(self, transaction_id: str) -> Dict[str, Any]:
        try:
            transaction = await self.db.find_one(
                CollectionName.TRANSACTIONS, {"transaction_id": transaction_id}
            )

            if not transaction:
                raise HTTPException(status_code=404, detail="Transaction not found")

            return {
                "transaction_id": transaction_id,
                "status": transaction["status"],
                "updated_at": transaction.get("updated_at", transaction["created_at"]),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


class AlertHandler:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    async def create_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            alert_data["created_at"] = datetime.utcnow().isoformat()
            alert_data["status"] = AlertStatus.NEW.value

            alert_id = await self.db.insert_one(CollectionName.ALERTS, alert_data)

            return {"alert_id": alert_id, "status": "created"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def update_alert_status(
        self, alert_id: str, new_status: str
    ) -> Dict[str, Any]:
        try:
            result = await self.db.update_one(
                CollectionName.ALERTS,
                {"alert_id": alert_id},
                {"status": new_status, "updated_at": datetime.utcnow().isoformat()},
            )

            if not result:
                raise HTTPException(status_code=404, detail="Alert not found")

            return {"alert_id": alert_id, "status": new_status}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


class CustomerHandler:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    async def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        try:
            customer = await self.db.find_one(
                CollectionName.CUSTOMERS, {"customer_id": customer_id}
            )

            if not customer:
                raise HTTPException(status_code=404, detail="Customer not found")

            return customer

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def update_customer_status(
        self, customer_id: str, new_status: str
    ) -> Dict[str, Any]:
        try:
            result = await self.db.update_one(
                CollectionName.CUSTOMERS,
                {"customer_id": customer_id},
                {"status": new_status, "updated_at": datetime.utcnow().isoformat()},
            )

            if not result:
                raise HTTPException(status_code=404, detail="Customer not found")

            return {"customer_id": customer_id, "status": new_status}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


class RiskHandler:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    async def get_risk_assessment(self, customer_id: str) -> Dict[str, Any]:
        try:
            risk_data = await self.db.find_one(
                CollectionName.RISK_SCORES, {"customer_id": customer_id}
            )

            if not risk_data:
                return {
                    "customer_id": customer_id,
                    "risk_score": 0.0,
                    "risk_level": "UNKNOWN",
                }

            return risk_data

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def update_risk_score(
        self, customer_id: str, risk_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            risk_data["updated_at"] = datetime.utcnow().isoformat()

            await self.db.update_one(
                CollectionName.RISK_SCORES, {"customer_id": customer_id}, risk_data
            )

            return {
                "customer_id": customer_id,
                "risk_score": risk_data["risk_score"],
                "risk_level": risk_data["risk_level"],
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


class ReportHandler:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    async def generate_transaction_report(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        try:
            transactions = await self.db.find_many(
                CollectionName.TRANSACTIONS,
                {
                    "created_at": {
                        "$gte": start_date.isoformat(),
                        "$lte": end_date.isoformat(),
                    }
                },
            )

            return [
                {
                    "transaction_id": t["transaction_id"],
                    "amount": t["amount"],
                    "status": t["status"],
                    "created_at": t["created_at"],
                }
                for t in transactions
            ]

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def generate_alert_summary(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        try:
            alerts = await self.db.find_many(
                CollectionName.ALERTS,
                {
                    "created_at": {
                        "$gte": start_date.isoformat(),
                        "$lte": end_date.isoformat(),
                    }
                },
            )

            status_counts = {}
            for alert in alerts:
                status = alert["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "total_alerts": len(alerts),
                "status_breakdown": status_counts,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
