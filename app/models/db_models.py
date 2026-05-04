"""
ArgusX — SQLAlchemy ORM Models
Stores every analysis result with full audit trail.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, Text, JSON, Index
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from app.core.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisLog(Base):
    """
    Persisted record of every prompt analysis.
    Captures the full scoring breakdown plus the final decision.
    """
    __tablename__ = "analysis_logs"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # ── Input ──────────────────────────────────────────────────────────────
    prompt = Column(Text, nullable=False)
    prompt_hash = Column(String(64), nullable=False, index=True)  # SHA-256
    prompt_length = Column(Integer, nullable=False)
    source_ip = Column(String(45), nullable=True)   # IPv4 / IPv6
    session_id = Column(String(128), nullable=True)

    # ── Layer scores (0–100) ────────────────────────────────────────────────
    pattern_score = Column(Float, nullable=False, default=0.0)
    semantic_score = Column(Float, nullable=False, default=0.0)
    behavioral_score = Column(Float, nullable=False, default=0.0)
    anomaly_score = Column(Float, nullable=False, default=0.0)
    final_score = Column(Float, nullable=False, default=0.0)

    # ── Decision ──────────────────────────────────────────────────────────
    decision = Column(String(16), nullable=False)          # ALLOW | BLOCK | FLAG | SANITIZE
    sanitized_prompt = Column(Text, nullable=True)

    # ── Explainability ────────────────────────────────────────────────────
    matched_patterns = Column(JSON, nullable=True)         # list of triggered pattern names
    behavioral_flags = Column(JSON, nullable=True)         # list of behavioral signal names
    explanation = Column(Text, nullable=True)              # human-readable summary

    # ── Metadata ──────────────────────────────────────────────────────────
    processing_ms = Column(Float, nullable=True)           # inference time
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    __table_args__ = (
        Index("ix_logs_created_at", "created_at"),
        Index("ix_logs_decision", "decision"),
        Index("ix_logs_final_score", "final_score"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "prompt_length": self.prompt_length,
            "scores": {
                "pattern": self.pattern_score,
                "semantic": self.semantic_score,
                "behavioral": self.behavioral_score,
                "anomaly": self.anomaly_score,
                "final": self.final_score,
            },
            "decision": self.decision,
            "sanitized_prompt": self.sanitized_prompt,
            "matched_patterns": self.matched_patterns,
            "behavioral_flags": self.behavioral_flags,
            "explanation": self.explanation,
            "processing_ms": self.processing_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
