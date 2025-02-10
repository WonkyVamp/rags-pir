from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.security import OAuth2PasswordBearer
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
from pydantic import BaseModel, Field
import asyncio
from enum import Enum

from models.transaction import Transaction, TransactionStatus, TransactionType
from models.customer import Customer, CustomerStatus, RiskLevel
from models.alert import Alert, AlertStatus, AlertPriority
from models.risk_score import RiskScore

# Service imports
from services.openai_service import OpenAIService
from services.database_service import DatabaseService, DatabaseType, CollectionName
from services.notification_service import NotificationService, NotificationPriority
from services.audit_service import AuditService, AuditEventType, AuditSeverity

# Initialize routers
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Request/Response Models
class AnalyzeTransactionRequest(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    merchant_id: str
    timestamp: datetime
    location: Dict[str, float]
    device_info: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]

class RiskAssessmentResponse(BaseModel):
    risk_score: float
    risk_level: str
    factors: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime

class AlertCreationRequest(BaseModel):
    customer_id: str
    alert_type: str
    priority: AlertPriority
    details: Dict[str, Any]
    source: str

class TransactionBatchRequest(BaseModel):
    transactions: List[AnalyzeTransactionRequest]
    batch_id: str

class CustomerUpdateRequest(BaseModel):
    status: Optional[CustomerStatus]
    risk_level: Optional[RiskLevel]
    metadata: Optional[Dict[str, Any]]

class PatternAnalysisRequest(BaseModel):
    customer_id: str
    start_date: datetime
    end_date: datetime
    pattern_types: List[str]

# Dependency injection
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Implement user authentication
    pass

async def get_services():
    # Service initialization would happen at startup
    pass

# Transaction Routes
@router.post("/transactions/analyze", response_model=RiskAssessmentResponse)
async def analyze_transaction(
    request: AnalyzeTransactionRequest,
    background_tasks: BackgroundTasks,
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        db_service: DatabaseService = services['database']
        openai_service: OpenAIService = services['openai']
        audit_service: AuditService = services['audit']
        
        # Record audit event
        await audit_service.record_event(
            event_type=AuditEventType.TRANSACTION,
            actor=current_user['id'],
            action="analyze_transaction",
            resource=request.transaction_id,
            severity=AuditSeverity.INFO,
            details=request.dict()
        )
        
        # Store transaction
        transaction_data = request.dict()
        await db_service.insert_one(
            CollectionName.TRANSACTIONS,
            transaction_data
        )
        
        # Analyze with OpenAI
        analysis = await openai_service.analyze_transaction(transaction_data)
        
        # Process result in background
        background_tasks.add_task(
            process_transaction_analysis,
            analysis,
            request.customer_id,
            services
        )
        
        return RiskAssessmentResponse(
            risk_score=analysis['risk_score'],
            risk_level=analysis['risk_level'],
            factors=analysis['factors'],
            recommendations=analysis['recommendations'],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        await audit_service.record_event(
            event_type=AuditEventType.ERROR,
            actor=current_user['id'],
            action="analyze_transaction_error",
            resource=request.transaction_id,
            severity=AuditSeverity.ERROR,
            details={'error': str(e)}
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transactions/batch", response_model=Dict[str, Any])
async def process_transaction_batch(
    request: TransactionBatchRequest,
    background_tasks: BackgroundTasks,
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        results = []
        for transaction in request.transactions:
            result = await analyze_transaction(
                transaction,
                background_tasks,
                services,
                current_user
            )
            results.append({
                'transaction_id': transaction.transaction_id,
                'result': result
            })
            
        return {
            'batch_id': request.batch_id,
            'results': results,
            'timestamp': datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transactions/{transaction_id}", response_model=Dict[str, Any])
async def get_transaction(
    transaction_id: str = Path(..., title="The ID of the transaction to retrieve"),
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        db_service: DatabaseService = services['database']
        transaction = await db_service.find_one(
            CollectionName.TRANSACTIONS,
            {'transaction_id': transaction_id}
        )
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
            
        return transaction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Alert Routes
@router.post("/alerts", response_model=Dict[str, Any])
async def create_alert(
    request: AlertCreationRequest,
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        db_service: DatabaseService = services['database']
        notification_service: NotificationService = services['notification']
        
        alert_data = request.dict()
        alert_data['created_by'] = current_user['id']
        alert_data['created_at'] = datetime.utcnow()
        
        alert_id = await db_service.insert_one(
            CollectionName.ALERTS,
            alert_data
        )
        
        # Send notification if high priority
        if request.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]:
            await notification_service.send_notification(
                template_id="high_priority_alert",
                recipient=alert_data['customer_id'],
                variables=alert_data,
                priority=NotificationPriority.HIGH
            )
            
        return {'alert_id': alert_id, 'status': 'created'}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/{alert_id}", response_model=Dict[str, Any])
async def get_alert(
    alert_id: str = Path(..., title="The ID of the alert to retrieve"),
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        db_service: DatabaseService = services['database']
        alert = await db_service.find_one(
            CollectionName.ALERTS,
            {'alert_id': alert_id}
        )
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
            
        return alert
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/alerts/{alert_id}", response_model=Dict[str, Any])
async def update_alert(
    alert_id: str,
    status: AlertStatus,
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        db_service: DatabaseService = services['database']
        audit_service: AuditService = services['audit']
        
        result = await db_service.update_one(
            CollectionName.ALERTS,
            {'alert_id': alert_id},
            {'status': status.value, 'updated_by': current_user['id']}
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Alert not found")
            
        await audit_service.record_event(
            event_type=AuditEventType.ALERT,
            actor=current_user['id'],
            action="update_alert",
            resource=alert_id,
            severity=AuditSeverity.INFO,
            details={'new_status': status.value}
        )
        
        return {'alert_id': alert_id, 'status': 'updated'}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Customer Routes
@router.get("/customers/{customer_id}", response_model=Dict[str, Any])
async def get_customer(
    customer_id: str = Path(..., title="The ID of the customer to retrieve"),
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        db_service: DatabaseService = services['database']
        customer = await db_service.find_one(
            CollectionName.CUSTOMERS,
            {'customer_id': customer_id}
        )
        
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        return customer
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/customers/{customer_id}", response_model=Dict[str, Any])
async def update_customer(
    customer_id: str,
    request: CustomerUpdateRequest,
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        db_service: DatabaseService = services['database']
        audit_service: AuditService = services['audit']
        
        update_data = request.dict(exclude_unset=True)
        result = await db_service.update_one(
            CollectionName.CUSTOMERS,
            {'customer_id': customer_id},
            update_data
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        await audit_service.record_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            actor=current_user['id'],
            action="update_customer",
            resource=customer_id,
            severity=AuditSeverity.INFO,
            details=update_data
        )
        
        return {'customer_id': customer_id, 'status': 'updated'}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pattern Analysis Routes
@router.post("/patterns/analyze", response_model=Dict[str, Any])
async def analyze_patterns(
    request: PatternAnalysisRequest,
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        db_service: DatabaseService = services['database']
        openai_service: OpenAIService = services['openai']
        
        # Get customer transactions
        transactions = await db_service.find_many(
            CollectionName.TRANSACTIONS,
            {
                'customer_id': request.customer_id,
                'timestamp': {
                    '$gte': request.start_date,
                    '$lte': request.end_date
                }
            }
        )
        
        # Analyze patterns
        patterns = await analyze_transaction_patterns(
            transactions,
            request.pattern_types,
            services
        )
        
        return {
            'customer_id': request.customer_id,
            'patterns': patterns,
            'timestamp': datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Risk Assessment Routes
@router.get("/risk-assessment/{customer_id}", response_model=Dict[str, Any])
async def get_risk_assessment(
    customer_id: str = Path(..., title="The ID of the customer"),
    services: Dict = Depends(get_services),
    current_user: Dict = Depends(get_current_user)
):
    try:
        db_service: DatabaseService = services['database']
        risk_score = await db_service.find_one(
            CollectionName.RISK_SCORES,
            {'customer_id': customer_id}
        )
        
        if not risk_score:
            raise HTTPException(status_code=404, detail="Risk assessment not found")
            
        return risk_score
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility Functions
async def process_transaction_analysis(
    analysis: Dict[str, Any],
    customer_id: str,
    services: Dict[str, Any]
):
    try:
        db_service: DatabaseService = services['database']
        notification_service: NotificationService = services['notification']
        
        # Update risk score
        await db_service.update_one(
            CollectionName.CUSTOMERS,
            {'customer_id': customer_id},
            {'risk_score': analysis['risk_score']}
        )
        
        # Create alert if high risk
        if analysis['risk_score'] >= 0.7:
            alert_data = {
                'customer_id': customer_id,
                'alert_type': 'high_risk_transaction',
                'priority': AlertPriority.HIGH,
                'details': analysis
            }
            await create_alert(
                AlertCreationRequest(**alert_data),
                services,
                {'id': 'system'}
            )
            
