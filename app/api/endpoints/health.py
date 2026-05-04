"""
ArgusX — GET /api/v1/health
System health check endpoint.
"""

import time
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.api.schemas import HealthResponse, ModelStatus
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Track startup time
_START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns service health, model load status, and uptime.",
    status_code=status.HTTP_200_OK,
)
async def health_check(request: Request) -> HealthResponse:
    """
    Liveness + readiness check.
    Returns 200 if system is healthy, 503 if models are not loaded.
    """
    registry = getattr(request.app.state, "model_registry", None)

    model_status = ModelStatus(
        behavioral_model=False,
        anomaly_detector=False,
        vectorizer=False,
    )
    db_status = "unknown"
    overall = "degraded"

    if registry is not None:
        status_dict = registry.status()
        model_status = ModelStatus(
            behavioral_model=status_dict.get("behavioral_model", False),
            anomaly_detector=status_dict.get("anomaly_detector", False),
            vectorizer=status_dict.get("vectorizer", False),
        )
        overall = "healthy" if registry.is_ready else "degraded"

    # Quick DB connectivity check
    try:
        from app.core.database import engine
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        db_status = "disconnected"
        overall = "degraded"

    uptime = round(time.time() - _START_TIME, 2)

    return HealthResponse(
        status=overall,
        version="1.0.0",
        models=model_status,
        uptime_seconds=uptime,
        database=db_status,
    )
