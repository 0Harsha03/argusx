"""
ArgusX v9 — Prompt Injection Route
=====================================
Concrete route handling prompts classified as "Prompt Injection".

v9 behaviour:
  Delegates entirely to SBERTSemanticAnalyzer + BehavioralAnalyzer +
  AnomalyDetector — identical to the v8 pipeline. This is by design:
  the routing skeleton is in place but DistilBERT has NOT been inserted.

Extension point (v9.1+):
  Replace `self._semantic_analyzer` with a DistilBERTSemanticAnalyzer
  that satisfies the same SemanticAnalysisResult interface.
  The BehavioralAnalyzer and AnomalyDetector remain unchanged.

Invariant:
  RouteResult scores produced here are numerically identical to v8
  DetectionPipeline scores for the same input. Benchmark results
  are therefore not affected by this refactor.
"""

from __future__ import annotations

import logging
from typing import List
import numpy as np

from app.routing.base_route import BaseRoute, RouteResult
from app.detection.distilbert_semantic_analyzer import DistilBERTSemanticAnalyzer
from app.detection.behavioral_analyzer import BehavioralAnalyzer
from app.detection.anomaly_detector import AnomalyDetector as LOFDetector

logger = logging.getLogger(__name__)


class PromptInjectionRoute(BaseRoute):
    """
    Route for prompt injection threat class.

    Internal engines (v11):
      - Semantic:   DistilBERTSemanticAnalyzer 
      - Behavioral: BehavioralAnalyzer     (RandomForest — unchanged)
      - Anomaly:    LOFDetector            (LocalOutlierFactor — unchanged)

    Extension point:
      Swap _semantic_analyzer for a DistilBERT-backed analyzer.
      Interface contract: must expose .analyze(text) -> SemanticAnalysisResult
      with .score (float 0-100), .top_match (Optional[str]).
    """

    def __init__(self, behavioral_model, vectorizer, anomaly_detector, platt_db=None, platt_rf=None) -> None:
        self.platt_db = platt_db
        self.platt_rf = platt_rf
        # ── Semantic engine (v11: DistilBERT) ─────────────────
        self._semantic_analyzer = DistilBERTSemanticAnalyzer()

        # ── Behavioral + Anomaly (unchanged from v8) ──────────────────────
        self._behavioral_analyzer = BehavioralAnalyzer(behavioral_model, vectorizer)
        self._anomaly_detector = LOFDetector(anomaly_detector, vectorizer)

        logger.info(
            "PromptInjectionRoute initialized | semantic=%s",
            type(self._semantic_analyzer).__name__,
        )

    @property
    def route_name(self) -> str:
        return "PromptInjectionRoute"

    def process(self, raw_prompt: str, clean_prompt: str, pattern_score: float,
                pattern_categories: List[str]) -> RouteResult:
        """
        Run semantic, behavioral, and anomaly analysis for injection threats.

        Returns scores numerically identical to v8 DetectionPipeline.
        """
        semantic_result = self._semantic_analyzer.analyze(raw_prompt)
        behavioral_result = self._behavioral_analyzer.analyze(
            text=raw_prompt,
            pattern_score=pattern_score,
            pattern_categories=pattern_categories,
        )
        anomaly_result = self._anomaly_detector.analyze(raw_prompt)

        # ── Phase 3B.1: Calibrated Probability Generation ─────────────────
        db_raw = semantic_result.raw_probability
        rf_raw = behavioral_result.raw_probability

        db_calibrated = db_raw
        rf_calibrated = rf_raw

        if self.platt_db is not None:
            db_calibrated = float(self.platt_db.predict_proba(np.array([[db_raw]]))[0, 1])
        
        if self.platt_rf is not None:
            rf_calibrated = float(self.platt_rf.predict_proba(np.array([[rf_raw]]))[0, 1])

        return RouteResult(
            route_name=self.route_name,
            pattern_score=pattern_score,
            semantic_score=semantic_result.score,
            behavioral_score=behavioral_result.score,
            anomaly_score=anomaly_result.score,
            matched_patterns=[],                    # populated by router from PatternDetector
            behavioral_flags=behavioral_result.behavioral_flags,
            semantic_top_match=semantic_result.top_match,
            anomaly_is_anomaly=anomaly_result.is_anomaly,
            route_metadata={
                "engine": "DistilBERTSemanticAnalyzer",
                "max_similarity": semantic_result.max_similarity,
                "above_threshold": semantic_result.above_threshold,
                # Phase 3B.1 Diagnostics
                "raw_db_probability": db_raw,
                "raw_rf_probability": rf_raw,
                "calibrated_db_probability": db_calibrated,
                "calibrated_rf_probability": rf_calibrated,
            },
        )
