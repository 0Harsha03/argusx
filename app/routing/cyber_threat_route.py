"""
ArgusX v9 — Cyber Threat Route
=================================
Concrete route handling prompts classified as "Cyber Threat" (malware
generation, exploitation, CVE abuse, phishing, credential theft, etc.).

v9 behaviour:
  Identical to PromptInjectionRoute — delegates to the same SBERT +
  Behavioral + Anomaly stack. The separation exists to allow future
  specialization (e.g. a CVE-specific classifier) without touching the
  injection route.

Extension point (v10+):
  Replace or augment _semantic_analyzer with a domain-specific
  cybersecurity classifier trained on CyberSecEval/MITRE datasets.
"""

from __future__ import annotations

import logging
from typing import List

from app.routing.base_route import BaseRoute, RouteResult
from app.detection.sbert_semantic_analyzer import SBERTSemanticAnalyzer
from app.detection.behavioral_analyzer import BehavioralAnalyzer
from app.detection.anomaly_detector import AnomalyDetector as LOFDetector

logger = logging.getLogger(__name__)


class CyberThreatRoute(BaseRoute):
    """
    Route for cyber-specific threat classes (malware, exploitation, phishing).

    Internal engines (v9):
      - Semantic:   SBERTSemanticAnalyzer  (shared anchor bank with injection route)
      - Behavioral: BehavioralAnalyzer     (shared RandomForest)
      - Anomaly:    LOFDetector            (shared LOF model)

    Extension point:
      Replace _semantic_analyzer with a CyberSecEval-fine-tuned encoder.
      Alternatively, add a _rule_augmenter that boosts score on MITRE TTPs.
    """

    def __init__(self, behavioral_model, vectorizer, anomaly_detector) -> None:
        # EXTENSION POINT: replace with CVE/MITRE-specialized semantic engine
        self._semantic_analyzer = SBERTSemanticAnalyzer()
        self._behavioral_analyzer = BehavioralAnalyzer(behavioral_model, vectorizer)
        self._anomaly_detector = LOFDetector(anomaly_detector, vectorizer)

        logger.info(
            "CyberThreatRoute initialized | semantic=%s",
            type(self._semantic_analyzer).__name__,
        )

    @property
    def route_name(self) -> str:
        return "CyberThreatRoute"

    def process(self, text: str, pattern_score: float,
                pattern_categories: List[str]) -> RouteResult:
        """
        Run cyber-threat-specific detection.

        v9: outputs identical to PromptInjectionRoute (shared engines).
        v10+: this method diverges when a specialized classifier is inserted.
        """
        semantic_result = self._semantic_analyzer.analyze(text)
        behavioral_result = self._behavioral_analyzer.analyze(
            text=text,
            pattern_score=pattern_score,
            pattern_categories=pattern_categories,
        )
        anomaly_result = self._anomaly_detector.analyze(text)

        return RouteResult(
            route_name=self.route_name,
            pattern_score=pattern_score,
            semantic_score=semantic_result.score,
            behavioral_score=behavioral_result.score,
            anomaly_score=anomaly_result.score,
            behavioral_flags=behavioral_result.behavioral_flags,
            semantic_top_match=semantic_result.top_match,
            anomaly_is_anomaly=anomaly_result.is_anomaly,
            route_metadata={
                "engine": "SBERTSemanticAnalyzer",
                "max_similarity": semantic_result.max_similarity,
                "above_threshold": semantic_result.above_threshold,
            },
        )
