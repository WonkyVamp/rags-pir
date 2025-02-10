import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AgentState:
    """Represents the current state of an agent"""

    agent_id: str
    status: str
    last_active: datetime
    current_task: Optional[str]
    performance_metrics: Dict[str, float]
    memory_buffer: List[Dict[str, Any]]


class AgentMessage:
    """Represents a message between agents"""

    def __init__(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        content: Dict[str, Any],
        priority: int = 1,
    ):
        self.message_id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.timestamp = datetime.utcnow()
        self.status = "created"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
        }


class BaseAgent(ABC):
    """Abstract base class for all agents in the fraud detection system"""

    def __init__(self, agent_id: str, agent_type: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.logger = self._setup_logger()
        self.state = AgentState(
            agent_id=agent_id,
            status="initialized",
            last_active=datetime.utcnow(),
            current_task=None,
            performance_metrics={
                "messages_processed": 0,
                "average_response_time": 0.0,
                "success_rate": 1.0,
                "error_count": 0,
            },
            memory_buffer=[],
        )
        self.message_queue = asyncio.Queue()
        self.running = False
        self._message_handlers = {}
        self._register_default_handlers()

    def _setup_logger(self) -> logging.Logger:
        """Sets up logging for the agent"""
        logger = logging.getLogger(f"agent.{self.agent_type}.{self.agent_id}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _register_default_handlers(self):
        """Registers default message handlers"""
        self._message_handlers = {
            "status_request": self._handle_status_request,
            "shutdown": self._handle_shutdown,
            "configuration_update": self._handle_config_update,
            "error_report": self._handle_error_report,
        }

    async def _handle_status_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handles status request messages"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.state.status,
            "last_active": self.state.last_active.isoformat(),
            "current_task": self.state.current_task,
            "performance_metrics": self.state.performance_metrics,
        }

    async def _handle_shutdown(self, message: AgentMessage) -> Dict[str, Any]:
        """Handles shutdown messages"""
        self.running = False
        return {"status": "shutting_down", "agent_id": self.agent_id}

    async def _handle_config_update(self, message: AgentMessage) -> Dict[str, Any]:
        """Handles configuration update messages"""
        try:
            new_config = message.content.get("config", {})
            self.config.update(new_config)
            return {"status": "config_updated", "agent_id": self.agent_id}
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _handle_error_report(self, message: AgentMessage) -> Dict[str, Any]:
        """Handles error report messages"""
        error_info = message.content.get("error_info", {})
        self.logger.error(f"Error reported: {json.dumps(error_info, indent=2)}")
        self.state.performance_metrics["error_count"] += 1
        return {"status": "error_logged", "agent_id": self.agent_id}

    async def send_message(
        self,
        receiver_id: str,
        message_type: str,
        content: Dict[str, Any],
        priority: int = 1,
    ) -> str:
        """Sends a message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority,
        )

        try:
            await self.message_queue.put(message)
            self.logger.debug(f"Message {message.message_id} queued for {receiver_id}")
            return message.message_id
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            raise

    async def receive_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Processes received messages"""
        start_time = datetime.utcnow()

        try:
            handler = self._message_handlers.get(
                message.message_type, self._handle_unknown_message
            )
            response = await handler(message)

            # Update performance metrics
            self.state.performance_metrics["messages_processed"] += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            current_avg = self.state.performance_metrics["average_response_time"]
            messages_processed = self.state.performance_metrics["messages_processed"]

            # Update rolling average of response time
            self.state.performance_metrics["average_response_time"] = (
                current_avg * (messages_processed - 1) + processing_time
            ) / messages_processed

            self.state.last_active = datetime.utcnow()
            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            self.state.performance_metrics["error_count"] += 1
            self.state.performance_metrics["success_rate"] = (
                messages_processed - self.state.performance_metrics["error_count"]
            ) / messages_processed
            raise

    async def _handle_unknown_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handles unknown message types"""
        self.logger.warning(f"Received unknown message type: {message.message_type}")
        return {
            "status": "error",
            "message": f"Unknown message type: {message.message_type}",
        }

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent with necessary setup"""
        pass

    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Process the main logic of the agent"""
        pass

    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources when shutting down"""
        pass

    async def run(self):
        """Main run loop for the agent"""
        try:
            await self.initialize()
            self.running = True
            self.state.status = "running"

            while self.running:
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=self.config.get("message_timeout", 1.0),
                    )

                    response = await self.receive_message(message)
                    self.state.memory_buffer.append(
                        {
                            "timestamp": datetime.utcnow(),
                            "message": message.to_dict(),
                            "response": response,
                        }
                    )

                    # Trim memory buffer if it exceeds configured size
                    max_buffer_size = self.config.get("max_memory_buffer_size", 1000)
                    if len(self.state.memory_buffer) > max_buffer_size:
                        self.state.memory_buffer = self.state.memory_buffer[
                            -max_buffer_size:
                        ]

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in run loop: {str(e)}")
                    self.state.performance_metrics["error_count"] += 1

            await self.cleanup()
            self.state.status = "shutdown"

        except Exception as e:
            self.logger.error(f"Critical error in agent run loop: {str(e)}")
            self.state.status = "error"
            raise

    def get_performance_metrics(self) -> Dict[str, float]:
        """Returns the current performance metrics of the agent"""
        return self.state.performance_metrics.copy()

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the agent"""
        return {
            "agent_id": self.state.agent_id,
            "status": self.state.status,
            "last_active": self.state.last_active.isoformat(),
            "current_task": self.state.current_task,
            "performance_metrics": self.state.performance_metrics,
            "memory_buffer_size": len(self.state.memory_buffer),
        }

    def register_message_handler(self, message_type: str, handler_func: callable):
        """Registers a new message handler"""
        self._message_handlers[message_type] = handler_func
        self.logger.info(f"Registered handler for message type: {message_type}")

    async def broadcast_message(
        self, message_type: str, content: Dict[str, Any], target_agents: List[str]
    ) -> List[str]:
        """Broadcasts a message to multiple agents"""
        message_ids = []
        for target_id in target_agents:
            try:
                message_id = await self.send_message(
                    receiver_id=target_id, message_type=message_type, content=content
                )
                message_ids.append(message_id)
            except Exception as e:
                self.logger.error(f"Error broadcasting to agent {target_id}: {str(e)}")

        return message_ids

    def update_performance_metrics(self, metric_name: str, value: float):
        """Updates a specific performance metric"""
        self.state.performance_metrics[metric_name] = value
        self.logger.debug(f"Updated metric {metric_name} to {value}")
