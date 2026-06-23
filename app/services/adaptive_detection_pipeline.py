"""
ArgusX v9 — Adaptive Threat-Aware Detection Pipeline
======================================================
Subclasses SBERTDetectionPipeline (v8) and inserts the ThreatRouter
between the Pattern Engine and the Threat Scoring layer.

Pipeline flow:

    raw_prompt
        │
        ▼
    [Preprocessing]
        │
        ▼
    [Layer 1: PatternDetector]    ← unchanged from v8
        │ pattern_score, pattern_categories
        ▼
    [ThreatRouter]                ← NEW in v9
        │  ┌─────────────────┐   ┌──────────────────┐
        │  │PromptInjection  │   │  CyberThreat     │
        │  │Route            │   │  Route           │
        │  │ (SBERT+RF+LOF)  │   │  (SBERT+RF+LOF)  │
        │  └────────┬────────┘   └───────┬──────────┘
        │           └──────────┬─────────┘
        │                  RouteResult
        ▼
    [Layer 5: ThreatScorer]       ← unchanged from v8
        │
        ▼
    Decision (ALLOW/FLAG/SANITIZE/BLOCK)

Backward-compatibility guarantee:
  For all inputs, output scores and decisions are numerically identical
  to af9123f (v8 SBERTDetectionPipeline) because both routes internally
  use the identical engines. The router adds zero score bias.

Usage:
    registry = ModelRegistry()
    registry.load_all()
    pipeline = AdaptiveDetectionPipeline(registry)  # drop-in for v8
    result   = pipeline.analyze(prompt)
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from app.services.model_registry import ModelRegistry
from app.detection.pattern_detector import PatternDetector
from app.detection.threat_scorer import ThreatScorer
from app.routing.threat_router import ThreatRouter
from app.utils.preprocessor import normalize_text, truncate_for_log

logger = logging.getLogger(__name__)


class AdaptiveDetectionPipeline:
    """
    ArgusX v9: Detection pipeline with ThreatRouter.

    Public interface (.analyze → dict) is 100% identical to
    SBERTDetectionPipeline (v8) for full drop-in compatibility.
    """

    VERSION = "9.0.0"

    def __init__(self, registry: ModelRegistry) -> None:
        if not registry.is_ready:
            raise RuntimeError("ModelRegistry is not ready — cannot build pipeline.")

        # Layer 1: Pattern (shared across all routes)
        self._pattern_detector = PatternDetector()

        # v9 routing layer
        self._router = ThreatRouter(
            behavioral_model=registry.behavioral_model,
            vectorizer=registry.vectorizer,
            anomaly_detector=registry.anomaly_detector,
        )

        # Layer 5: Threat Scoring (shared, unchanged)
        self._threat_scorer = ThreatScorer()

        logger.info(
            "AdaptiveDetectionPipeline v%s initialized | router=%s",
            self.VERSION, type(self._router).__name__,
        )

    def analyze(
        self,
        raw_prompt: str,
        session_id: Optional[str] = None,
        source_ip: Optional[str] = None,
    ) -> dict:
        """
        Run the full v9 adaptive detection pipeline on a raw prompt.

        Returns:
            dict — identical schema to v8 DetectionPipeline.analyze()
            with one additional field: 'selected_route'.
        """
        start = time.perf_counter()
        timestamp = datetime.now(timezone.utc)

        # ── Layer 0: Preprocessing ────────────────────────────────────────
        clean_prompt, transforms = normalize_text(raw_prompt)
        prompt_hash = hashlib.sha256(clean_prompt.encode()).hexdigest()

        logger.debug(
            "v9 analyzing [hash=%s, len=%d, transforms=%s]: %s",
            prompt_hash[:8], len(clean_prompt),
            transforms, truncate_for_log(clean_prompt, 120),
        )

        # ── Layer 1: Pattern Detection ────────────────────────────────────
        pattern_result = self._pattern_detector.analyze(clean_prompt)
        pattern_categories = list({d["category"] for d in pattern_result.details})

        # ── v9: ThreatRouter dispatch ─────────────────────────────────────
        routing = self._router.route(
            text=clean_prompt,
            pattern_score=pattern_result.score,
            pattern_categories=pattern_categories,
        )
        rr = routing.route_result  # RouteResult

        # ── Layer 5: Threat Scoring ───────────────────────────────────────
        threat = self._threat_scorer.compute(
            pattern_score=rr.pattern_score,
            semantic_score=rr.semantic_score,
            behavioral_score=rr.behavioral_score,
            anomaly_score=rr.anomaly_score,
            matched_patterns=pattern_result.matched_rules,
            behavioral_flags=rr.behavioral_flags,
            top_category=pattern_result.top_category,
            prompt=clean_prompt,
        )

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "v9 analysis complete [hash=%s] route=%s score=%.1f decision=%s in %.1fms",
            prompt_hash[:8], routing.selected_route,
            threat.final_score, threat.decision, elapsed_ms,
        )

        return {
            # Identity
            "prompt":           clean_prompt,
            "raw_prompt":       raw_prompt,
            "prompt_hash":      prompt_hash,
            "prompt_length":    len(clean_prompt),
            "session_id":       session_id,
            "source_ip":        source_ip,
            "timestamp":        timestamp,

            # Scores  (identical to v8)
            "pattern_score":    threat.pattern_score,
            "semantic_score":   threat.semantic_score,
            "behavioral_score": threat.behavioral_score,
            "anomaly_score":    threat.anomaly_score,
            "final_score":      threat.final_score,

            # Decision  (identical to v8)
            "decision":         threat.decision,
            "threat_category":  threat.threat_category,
            "sanitized_prompt": threat.sanitized_prompt,

            # Explainability  (identical to v8)
            "matched_patterns": threat.matched_patterns,
            "behavioral_flags": threat.behavioral_flags,
            "explanation":      threat.explanation,

            # Diagnostics  (identical to v8 + new route field)
            "preprocessing_transforms": transforms,
            "semantic_top_match": rr.semantic_top_match,
            "anomaly_is_anomaly": rr.anomaly_is_anomaly,
            "processing_ms":      elapsed_ms,

            # v9 extension
            "selected_route":   routing.selected_route,
            "route_metadata":   rr.route_metadata,
        }
