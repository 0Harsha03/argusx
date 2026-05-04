"""
ArgusX — API Router
Aggregates all versioned endpoint routers.
"""

from fastapi import APIRouter

from app.api.endpoints.analyze import router as analyze_router
from app.api.endpoints.health  import router as health_router
from app.api.endpoints.logs    import router as logs_router

api_router = APIRouter()

api_router.include_router(analyze_router, tags=["Detection"])
api_router.include_router(health_router,  tags=["Health"])
api_router.include_router(logs_router,    tags=["Logs"])
