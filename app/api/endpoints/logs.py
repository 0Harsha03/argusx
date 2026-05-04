"""
ArgusX — GET /api/v1/logs
Paginated structured log retrieval endpoint.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    Decision, LogEntry, LogsResponse, ScoreBreakdown
)
from app.core.database import get_db
from app.models.db_models import AnalysisLog

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/logs",
    response_model=LogsResponse,
    summary="Retrieve analysis logs",
    description=(
        "Paginated access to all analysis log entries. "
        "Filter by decision type or minimum threat score."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_logs(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    decision: Optional[str] = Query(
        None,
        description="Filter by decision: ALLOW, BLOCK, FLAG, SANITIZE",
        pattern="^(ALLOW|BLOCK|FLAG|SANITIZE)$",
    ),
    min_score: Optional[float] = Query(
        None, ge=0, le=100,
        description="Only return entries with final_score ≥ this value",
    ),
    db: AsyncSession = Depends(get_db),
) -> LogsResponse:
    """
    Returns paginated, optionally-filtered log entries sorted by newest first.
    """

    # ── Build query ───────────────────────────────────────────────────────
    base_query = select(AnalysisLog)

    if decision:
        base_query = base_query.where(AnalysisLog.decision == decision)
    if min_score is not None:
        base_query = base_query.where(AnalysisLog.final_score >= min_score)

    # Count total matching records
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Paginate
    offset = (page - 1) * page_size
    query = (
        base_query
        .order_by(desc(AnalysisLog.created_at))
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    # ── Serialise ─────────────────────────────────────────────────────────
    items = [
        LogEntry(
            id=row.id,
            prompt=row.prompt,
            prompt_hash=row.prompt_hash,
            decision=Decision(row.decision),
            scores=ScoreBreakdown(
                pattern=row.pattern_score,
                semantic=row.semantic_score,
                behavioral=row.behavioral_score,
                anomaly=row.anomaly_score,
                final=row.final_score,
            ),
            matched_patterns=row.matched_patterns or [],
            behavioral_flags=row.behavioral_flags or [],
            explanation=row.explanation or "",
            sanitized_prompt=row.sanitized_prompt,
            processing_ms=row.processing_ms,
            created_at=row.created_at,
        )
        for row in rows
    ]

    return LogsResponse(total=total, page=page, page_size=page_size, items=items)
