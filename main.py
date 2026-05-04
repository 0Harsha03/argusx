"""
ArgusX — Production Entry Point (FINAL)
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.database import engine, Base
from app.core.logging_config import configure_logging
from app.services.model_registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger = logging.getLogger("argusx.startup")
    logger.info("ArgusX starting — loading models and initializing DB…")

    # Create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready.")

    # Load model registry
    registry = ModelRegistry()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, registry.load_all)
    app.state.model_registry = registry
    logger.info("All models loaded. ArgusX is READY. 🛡️")

    yield

    logger.info("ArgusX shutting down…")
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ArgusX — LLM Security Firewall",
        description=(
            "**Production-grade multi-layer AI security system** for detecting "
            "prompt injection, adversarial inputs, jailbreaks, and instruction overrides.\n\n"
            "### Detection Layers\n"
            "1. **Pattern Detection** — Regex + named jailbreak rule library\n"
            "2. **Semantic Analysis** — TF-IDF cosine similarity (vectorizer.pkl)\n"
            "3. **Behavioral Analysis** — RandomForest classifier (behavioral_model.pkl)\n"
            "4. **Anomaly Detection** — LocalOutlierFactor (anomaly_detector.pkl)\n"
            "5. **Threat Scoring** — Weighted composite → ALLOW / FLAG / SANITIZE / BLOCK\n"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={"name": "ArgusX Security Team"},
        license_info={"name": "MIT"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
