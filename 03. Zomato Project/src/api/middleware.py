"""
FastAPI middleware — error handling, request ID, rate limiting.

Implementation: Phase 5 (Hardening & Production Readiness)
"""

import time
import uuid
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from src.services.filter import ValidationError

logger = logging.getLogger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware to inject UUID request IDs and calculate execution latency."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        # Inject tracing metadata headers in response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            f"API Request: {request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Latency: {duration_ms:.2f}ms | "
            f"Request-ID: {request_id}"
        )
        return response


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle domain validation errors by mapping them to HTTP 400 Bad Request."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(f"Validation failure on request {request_id}: {exc.errors}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": "Preference validation failed",
            "errors": exc.errors,
            "request_id": request_id,
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled server exceptions mapping to HTTP 500."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Internal server error on request {request_id}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected server error occurred.",
            "request_id": request_id,
        },
    )

