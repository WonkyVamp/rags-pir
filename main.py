import logging
from typing import Dict

import uvicorn
import yaml
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import (
    AuthMiddleware,
    ErrorHandlerMiddleware,
    LoggingMiddleware,
    RateLimitConfig,
    RateLimitMiddleware,
    ResponseFormatterMiddleware,
)
from api.routes import router
from services.audit_service import AuditService
from services.database_service import DatabaseConfig, DatabaseService, DatabaseType
from services.notification_service import NotificationService
from services.openai_service import OpenAIService

app = FastAPI(title="Fraud Detection System", version="1.0.0")

# Global services container
services = {}


def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict):
    logging.basicConfig(
        level=config["logging"]["level"],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def setup_middleware(app: FastAPI, config: Dict):
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["cors"]["allowed_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth
    app.add_middleware(
        AuthMiddleware,
        secret_key=config["auth"]["secret_key"],
        excluded_paths=set(config["auth"]["excluded_paths"]),
    )

    # Rate Limiting
    rate_limit_configs = {
        path: RateLimitConfig(**cfg) for path, cfg in config["rate_limits"].items()
    }
    app.add_middleware(
        RateLimitMiddleware, redis_url=config["redis"]["url"], config=rate_limit_configs
    )

    # Logging
    app.add_middleware(LoggingMiddleware, log_format=config["logging"]["format"])

    # Error Handling
    app.add_middleware(ErrorHandlerMiddleware)

    # Response Formatting
    app.add_middleware(
        ResponseFormatterMiddleware, include_metadata=config["api"]["include_metadata"]
    )


async def initialize_services(config: Dict):
    # Database
    db_config = DatabaseConfig(
        db_type=DatabaseType(config["database"]["type"]),
        host=config["database"]["host"],
        port=config["database"]["port"],
        database=config["database"]["name"],
        username=config["database"]["username"],
        password=config["database"]["password"],
    )
    services["database"] = DatabaseService(db_config)
    await services["database"].connect()

    # OpenAI
    services["openai"] = OpenAIService(
        {
            "openai_api_key": config["openai"]["api_key"],
            "model": config["openai"]["model"],
            "temperature": config["openai"]["temperature"],
        }
    )

    # Notification
    services["notification"] = NotificationService(
        {
            "smtp": config["smtp"],
            "twilio": config["twilio"],
            "redis": config["redis"],
            "templates": config["notification_templates"],
        }
    )
    await services["notification"].start()

    # Audit
    services["audit"] = AuditService(
        {
            "retention_days": config["audit"]["retention_days"],
            "batch_size": config["audit"]["batch_size"],
            "flush_interval": config["audit"]["flush_interval"],
        }
    )
    await services["audit"].start()


async def shutdown_services():
    if "database" in services:
        await services["database"].close()

    if "notification" in services:
        await services["notification"].stop()

    if "audit" in services:
        await services["audit"].stop()


@app.on_event("startup")
async def startup():
    config = load_config()
    setup_logging(config)
    setup_middleware(app, config)
    await initialize_services(config)

    # Mount routes
    app.include_router(router, prefix="/api/v1", dependencies=[Depends(get_services)])


@app.on_event("shutdown")
async def shutdown():
    await shutdown_services()


def get_services():
    return services


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {name: "running" for name in services.keys()},
    }


@app.get("/version")
async def version():
    return {"version": "1.0.0", "api_version": "v1", "environment": "production"}


def start():
    config = load_config()
    uvicorn.run(
        "main:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=config["server"]["reload"],
        workers=config["server"]["workers"],
    )


if __name__ == "__main__":
    start()
