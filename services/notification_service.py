import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiosmtplib
import httpx
import jinja2
import redis.asyncio as redis
from twilio.rest import Client


class NotificationType(Enum):
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    PUSH = "push"
    SLACK = "slack"


class NotificationPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NotificationTemplate:
    template_id: str
    subject: str
    content: str
    variables: List[str]
    channel: NotificationType


@dataclass
class NotificationResult:
    success: bool
    notification_id: str
    timestamp: datetime
    channel: NotificationType
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NotificationService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        self.redis_client = redis.Redis(
            host=config["redis_host"], port=config["redis_port"], db=config["redis_db"]
        )
        self.smtp_config = config.get("smtp", {})
        self.twilio_client = Client(
            config["twilio_account_sid"], config["twilio_auth_token"]
        )
        self.template_engine = jinja2.Environment(
            loader=jinja2.FileSystemLoader("templates"), autoescape=True
        )
        self.http_client = httpx.AsyncClient()
        self.templates = self._load_templates()
        self.notification_queue = asyncio.Queue()
        self.webhook_retries = config.get("webhook_retries", 3)
        self._worker_task = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("notification_service")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_templates(self) -> Dict[str, NotificationTemplate]:
        templates = {}
        template_configs = self.config.get("templates", {})

        for template_id, config in template_configs.items():
            templates[template_id] = NotificationTemplate(
                template_id=template_id,
                subject=config["subject"],
                content=config["content"],
                variables=config["variables"],
                channel=NotificationType(config["channel"]),
            )
        return templates

    async def start(self):
        self._worker_task = asyncio.create_task(self._notification_worker())
        self.logger.info("Notification service started")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self.http_client.aclose()
        await self.redis_client.close()
        self.logger.info("Notification service stopped")

    async def _notification_worker(self):
        while True:
            try:
                notification = await self.notification_queue.get()
                await self._process_notification(notification)
                self.notification_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing notification: {str(e)}")
                await asyncio.sleep(1)

    async def _process_notification(self, notification: Dict[str, Any]):
        channel = NotificationType(notification["channel"])
        try:
            if channel == NotificationType.EMAIL:
                result = await self._send_email(notification)
            elif channel == NotificationType.SMS:
                result = await self._send_sms(notification)
            elif channel == NotificationType.WEBHOOK:
                result = await self._send_webhook(notification)
            elif channel == NotificationType.PUSH:
                result = await self._send_push(notification)
            elif channel == NotificationType.SLACK:
                result = await self._send_slack(notification)
            else:
                raise ValueError(f"Unsupported notification channel: {channel}")

            await self._store_notification_result(result)
        except Exception as e:
            self.logger.error(f"Failed to process {channel} notification: {str(e)}")
            raise

    async def _send_email(self, notification: Dict[str, Any]) -> NotificationResult:
        try:
            message = MIMEMultipart()
            message["From"] = self.smtp_config["from_email"]
            message["To"] = notification["recipient"]
            message["Subject"] = notification["subject"]

            template = self.template_engine.get_template(
                f"{notification['template_id']}.html"
            )
            content = template.render(**notification["variables"])
            message.attach(MIMEText(content, "html"))

            await aiosmtplib.send(
                message,
                hostname=self.smtp_config["host"],
                port=self.smtp_config["port"],
                username=self.smtp_config["username"],
                password=self.smtp_config["password"],
                use_tls=self.smtp_config.get("use_tls", True),
            )

            return NotificationResult(
                success=True,
                notification_id=notification["notification_id"],
                timestamp=datetime.utcnow(),
                channel=NotificationType.EMAIL,
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                notification_id=notification["notification_id"],
                timestamp=datetime.utcnow(),
                channel=NotificationType.EMAIL,
                error=str(e),
            )

    async def _send_sms(self, notification: Dict[str, Any]) -> NotificationResult:
        try:
            template = self.template_engine.get_template(
                f"{notification['template_id']}.txt"
            )
            content = template.render(**notification["variables"])

            message = self.twilio_client.messages.create(
                body=content,
                from_=self.config["twilio_phone_number"],
                to=notification["recipient"],
            )

            return NotificationResult(
                success=True,
                notification_id=notification["notification_id"],
                timestamp=datetime.utcnow(),
                channel=NotificationType.SMS,
                metadata={"message_sid": message.sid},
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                notification_id=notification["notification_id"],
                timestamp=datetime.utcnow(),
                channel=NotificationType.SMS,
                error=str(e),
            )

    async def _send_webhook(self, notification: Dict[str, Any]) -> NotificationResult:
        for attempt in range(self.webhook_retries):
            try:
                response = await self.http_client.post(
                    notification["webhook_url"],
                    json=notification["payload"],
                    headers=notification.get("headers", {}),
                    timeout=10.0,
                )
                response.raise_for_status()

                return NotificationResult(
                    success=True,
                    notification_id=notification["notification_id"],
                    timestamp=datetime.utcnow(),
                    channel=NotificationType.WEBHOOK,
                    metadata={"status_code": response.status_code},
                )
            except Exception as e:
                if attempt == self.webhook_retries - 1:
                    return NotificationResult(
                        success=False,
                        notification_id=notification["notification_id"],
                        timestamp=datetime.utcnow(),
                        channel=NotificationType.WEBHOOK,
                        error=str(e),
                    )
                await asyncio.sleep(2**attempt)

    async def _send_push(self, notification: Dict[str, Any]) -> NotificationResult:
        try:
            # Implement push notification logic (e.g., Firebase)
            return NotificationResult(
                success=True,
                notification_id=notification["notification_id"],
                timestamp=datetime.utcnow(),
                channel=NotificationType.PUSH,
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                notification_id=notification["notification_id"],
                timestamp=datetime.utcnow(),
                channel=NotificationType.PUSH,
                error=str(e),
            )

    async def _send_slack(self, notification: Dict[str, Any]) -> NotificationResult:
        try:
            response = await self.http_client.post(
                notification["webhook_url"],
                json={
                    "text": notification["message"],
                    "blocks": notification.get("blocks", []),
                },
            )
            response.raise_for_status()

            return NotificationResult(
                success=True,
                notification_id=notification["notification_id"],
                timestamp=datetime.utcnow(),
                channel=NotificationType.SLACK,
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                notification_id=notification["notification_id"],
                timestamp=datetime.utcnow(),
                channel=NotificationType.SLACK,
                error=str(e),
            )

    async def _store_notification_result(self, result: NotificationResult):
        key = f"notification:{result.notification_id}"
        await self.redis_client.set(
            key, json.dumps(result.__dict__), ex=86400  # 24 hours expiration
        )

    async def send_notification(
        self,
        template_id: str,
        recipient: str,
        variables: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.MEDIUM,
    ) -> str:
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        notification_id = f"{template_id}_{datetime.utcnow().timestamp()}"

        notification = {
            "notification_id": notification_id,
            "template_id": template_id,
            "recipient": recipient,
            "variables": variables,
            "channel": template.channel.value,
            "priority": priority.value,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.notification_queue.put(notification)
        return notification_id

    async def get_notification_status(
        self, notification_id: str
    ) -> Optional[NotificationResult]:
        key = f"notification:{notification_id}"
        result = await self.redis_client.get(key)
        if result:
            data = json.loads(result)
            return NotificationResult(**data)
        return None

    def add_template(self, template: NotificationTemplate):
        self.templates[template.template_id] = template

    async def get_pending_notifications(self) -> int:
        return self.notification_queue.qsize()

    async def clear_notification_history(self, older_than_days: int = 30):
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        async for key in self.redis_client.scan_iter("notification:*"):
            result = await self.redis_client.get(key)
            if result:
                data = json.loads(result)
                timestamp = datetime.fromisoformat(data["timestamp"])
                if timestamp < cutoff:
                    await self.redis_client.delete(key)
