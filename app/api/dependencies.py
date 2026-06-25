"""
ArgusX — FastAPI Dependency Injection
"""

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.adaptive_detection_pipeline import AdaptiveDetectionPipeline


async def get_pipeline(request: Request) -> AdaptiveDetectionPipeline:
    """
    Retrieve the AdaptiveDetectionPipeline from app.state.
    Built once at startup; injected into every request handler.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None or not registry.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ArgusX models are not loaded. Service starting up.",
        )

    # Build pipeline lazily on first request, cache on app.state
    pipeline = getattr(request.app.state, "_pipeline", None)
    if pipeline is None:
        request.app.state._pipeline = AdaptiveDetectionPipeline(registry)
        pipeline = request.app.state._pipeline

    return pipeline
