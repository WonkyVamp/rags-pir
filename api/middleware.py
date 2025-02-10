from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.base import BaseHTTPMiddleware
from typing import Dict, Optional, Callable, Any
import time
import jwt
import json
import logging
from datetime import datetime
import asyncio
from dataclasses import dataclass
import redis.asyncio as redis
from fastapi.security import OAuth2PasswordBearer
import hashlib
import uuid


@dataclass
class RateLimitConfig:
    requests_per_minute: int
    burst_limit: int
    timeout: int


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        secret_key: str,
        excluded_paths: set = None,
        token_expire_minutes: int = 60,
    ):
        super().__init__(app)
        self.secret_key = secret_key
        self.excluded_paths = excluded_paths or {"/docs", "/redoc", "/openapi.json"}
        self.token_expire_minutes = token_expire_minutes
        self.logger = logging.getLogger("auth_middleware")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        try:
            token = self._extract_token(request)
            if not token:
                return JSONResponse(
                    status_code=401, content={"detail": "Missing authentication token"}
                )

            payload = self._validate_token(token)
            request.state.user = payload
            response = await call_next(request)
            return response

        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401, content={"detail": "Token has expired"}
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=401, content={"detail": "Invalid authentication token"}
            )
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return JSONResponse(
                status_code=500, content={"detail": "Internal server error"}
            )

    def _extract_token(self, request: Request) -> Optional[str]:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None
        parts = auth_header.split()
        if parts[0].lower() != "bearer" or len(parts) != 2:
            return None
        return parts[1]

    def _validate_token(self, token: str) -> Dict:
        return jwt.decode(token, self.secret_key, algorithms=["HS256"])


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str, config: Dict[str, RateLimitConfig]):
        super().__init__(app)
        self.redis = redis.from_url(redis_url)
        self.config = config
        self.logger = logging.getLogger("rate_limit_middleware")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host
        path = request.url.path
        key = f"ratelimit:{client_ip}:{path}"

        try:
            config = self.get_limit_config(path)
            allowed = await self._check_rate_limit(key, config)

            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Too many requests",
                        "retry_after": config.timeout,
                    },
                )

            response = await call_next(request)
            return response

        except Exception as e:
            self.logger.error(f"Rate limit error: {str(e)}")
            return await call_next(request)

    def get_limit_config(self, path: str) -> RateLimitConfig:
        for pattern, config in self.config.items():
            if pattern in path:
                return config
        return self.config.get("default")

    async def _check_rate_limit(self, key: str, config: RateLimitConfig) -> bool:
        async with self.redis.pipeline() as pipe:
            now = time.time()
            cleanup = now - 60  # Clean up entries older than 1 minute

            try:
                await pipe.zremrangebyscore(key, 0, cleanup)
                await pipe.zadd(key, {str(now): now})
                await pipe.zcard(key)
                await pipe.expire(key, 60)
                results = await pipe.execute()
                request_count = results[2]

                return request_count <= config.requests_per_minute

            except Exception as e:
                self.logger.error(f"Redis error: {str(e)}")
                return True


class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, log_format: str = None):
        super().__init__(app)
        self.logger = logging.getLogger("api_logger")
        self.log_format = (
            log_format or "{time} {method} {path} {status_code} {duration}ms"
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Capture request body if content-type is JSON
        body = None
        if request.headers.get("content-type") == "application/json":
            body = await self._get_request_body(request)

        try:
            response = await call_next(request)
            duration = round((time.time() - start_time) * 1000, 2)

            log_data = {
                "request_id": request_id,
                "time": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent"),
                "status_code": response.status_code,
                "duration": duration,
                "request_body": body,
                "response_body": await self._get_response_body(response),
            }

            self._log_request(log_data)
            return response

        except Exception as e:
            self.logger.error(
                f"Request failed: {str(e)}", extra={"request_id": request_id}
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )

    async def _get_request_body(self, request: Request) -> Optional[Dict]:
        try:
            body = await request.body()
            return json.loads(body) if body else None
        except Exception:
            return None

    async def _get_response_body(self, response: Response) -> Optional[Dict]:
        try:
            body = response.body
            return json.loads(body) if body else None
        except Exception:
            return None

    def _log_request(self, data: Dict[str, Any]):
        log_message = self.log_format.format(**data)

        if 200 <= data["status_code"] < 400:
            self.logger.info(log_message, extra=data)
        elif 400 <= data["status_code"] < 500:
            self.logger.warning(log_message, extra=data)
        else:
            self.logger.error(log_message, extra=data)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("error_handler")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response

        except jwt.InvalidTokenError:
            return self._create_error_response(
                401, "Invalid authentication token", "AUTH_ERROR"
            )

        except ValueError as e:
            return self._create_error_response(400, str(e), "VALIDATION_ERROR")

        except KeyError as e:
            return self._create_error_response(
                400, f"Missing required field: {str(e)}", "MISSING_FIELD"
            )

        except asyncio.TimeoutError:
            return self._create_error_response(
                504, "Request timed out", "TIMEOUT_ERROR"
            )

        except Exception as e:
            self.logger.exception("Unhandled error occurred")
            return self._create_error_response(
                500, "Internal server error", "INTERNAL_ERROR", str(e)
            )

    def _create_error_response(
        self, status_code: int, message: str, error_code: str, details: Any = None
    ) -> JSONResponse:
        error_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}{message}".encode()
        ).hexdigest()[:8]

        response_data = {
            "error": {
                "code": error_code,
                "message": message,
                "id": error_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        if details:
            response_data["error"]["details"] = details

        return JSONResponse(status_code=status_code, content=response_data)


class ResponseFormatterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, include_metadata: bool = True):
        super().__init__(app)
        self.include_metadata = include_metadata

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)

        if not response.headers.get("content-type") == "application/json":
            return response

        try:
            body = await self._get_response_body(response)
            if not body:
                return response

            formatted_response = {
                "data": body,
                "success": 200 <= response.status_code < 300,
            }

            if self.include_metadata:
                formatted_response["metadata"] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration": round((time.time() - start_time) * 1000, 2),
                    "path": request.url.path,
                    "method": request.method,
                }

            return JSONResponse(
                status_code=response.status_code,
                content=formatted_response,
                headers=dict(response.headers),
            )

        except Exception:
            return response

    async def _get_response_body(self, response: Response) -> Optional[Dict]:
        try:
            body = response.body
            return json.loads(body) if body else None
        except Exception:
            return None
