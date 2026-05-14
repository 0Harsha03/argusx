"""
ArgusX — POST /api/v1/analyze
Core threat analysis endpoint.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import AnalyzeRequest, AnalyzeResponse, Decision, ScoreBreakdown, ThreatCategory
from app.api.dependencies import get_pipeline
from app.core.database import get_db
from app.models.db_models import AnalysisLog
from app.services.detection_pipeline import DetectionPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def _safe_threat_category(value: str) -> ThreatCategory:
    """Coerce a raw category string to ThreatCategory, falling back gracefully."""
    try:
        return ThreatCategory(value)
    except ValueError:
        logger.warning(
            "Unknown threat_category '%s' — defaulting to ADVERSARIAL_INPUT. "
            "Add it to the ThreatCategory enum in schemas.py.",
            value,
        )
        return ThreatCategory.ADVERSARIAL_INPUT


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a prompt for threats",
    description=(
        "Submit an LLM prompt for multi-layer threat analysis. "
        "Returns a full threat score breakdown, decision, and explanation."
    ),
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Analysis result with full threat breakdown"},
        422: {"description": "Invalid request body"},
        503: {"description": "Models not yet loaded"},
    },
)
async def analyze_prompt(
    body: AnalyzeRequest,
    request: Request,
    pipeline: DetectionPipeline = Depends(get_pipeline),
    db: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    """
    Run the full 4-layer ArgusX detection pipeline on the submitted prompt.

    Layers executed:
    1. Pattern detection (regex)
    2. Semantic similarity (TF-IDF)
    3. Behavioral analysis (RandomForest)
    4. Anomaly detection (LOF)
    """
    request_id = str(uuid.uuid4())

    # Extract real IP from proxy headers if present
    source_ip = body.source_ip or (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.client.host if request.client else None
    )

    try:
        result = pipeline.analyze(
            raw_prompt=body.prompt,
            session_id=body.session_id,
            source_ip=source_ip,
        )
    except Exception as exc:
        logger.error("Pipeline error for request %s: %s", request_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(exc)}",
        )

    # ── Persist to database ──────────────────────────────────────────────
    log_entry = AnalysisLog(
        id=request_id,
        prompt=result["prompt"],
        prompt_hash=result["prompt_hash"],
        prompt_length=result["prompt_length"],
        source_ip=source_ip,
        session_id=body.session_id,
        pattern_score=result["pattern_score"],
        semantic_score=result["semantic_score"],
        behavioral_score=result["behavioral_score"],
        anomaly_score=result["anomaly_score"],
        final_score=result["final_score"],
        decision=result["decision"],
        sanitized_prompt=result["sanitized_prompt"],
        matched_patterns=result["matched_patterns"],
        behavioral_flags=result["behavioral_flags"],
        explanation=result["explanation"],
        processing_ms=result["processing_ms"],
    )
    db.add(log_entry)
    # Commit handled by the get_db dependency context manager

    logger.info(
        "request_id=%s decision=%s score=%.1f ip=%s",
        request_id, result["decision"], result["final_score"], source_ip,
    )

    return AnalyzeResponse(
        request_id=request_id,
        decision=Decision(result["decision"]),
        threat_category=_safe_threat_category(result["threat_category"]),
        scores=ScoreBreakdown(
            pattern=result["pattern_score"],
            semantic=result["semantic_score"],
            behavioral=result["behavioral_score"],
            anomaly=result["anomaly_score"],
            final=result["final_score"],
        ),
        matched_patterns=result["matched_patterns"],
        behavioral_flags=result["behavioral_flags"],
        explanation=result["explanation"],
        sanitized_prompt=result["sanitized_prompt"],
        processing_ms=result["processing_ms"],
        timestamp=result["timestamp"],
    )
