"""
ArgusX — Detection Pipeline Orchestrator
==========================================
The unified pipeline that runs all four detection layers in sequence
and returns a complete ThreatScore with full explainability breakdown.

Pipeline:
  Preprocessor → PatternDetector → SemanticAnalyzer
               → BehavioralAnalyzer → AnomalyDetector → ThreatScorer
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from app.detection.pattern_detector   import PatternDetector
from app.detection.semantic_analyzer  import SemanticAnalyzer
from app.detection.behavioral_analyzer import BehavioralAnalyzer
from app.detection.anomaly_detector   import AnomalyDetector as LOFDetector
from app.detection.threat_scorer      import ThreatScorer, ThreatScore
from app.utils.preprocessor           import normalize_text, truncate_for_log
from app.services.model_registry      import ModelRegistry

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Orchestrates the 4-layer ArgusX detection pipeline.

    Instantiated once per app lifetime (attached to app.state or
    injected via FastAPI dependency), initialized from the ModelRegistry.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        if not registry.is_ready:
            raise RuntimeError("ModelRegistry is not ready — cannot build pipeline.")

        self._pattern_detector   = PatternDetector()
        self._semantic_analyzer  = SemanticAnalyzer(registry.vectorizer)
        self._behavioral_analyzer = BehavioralAnalyzer(
            registry.behavioral_model,
            registry.vectorizer,
        )
        self._anomaly_detector   = LOFDetector(
            registry.anomaly_detector,
            registry.vectorizer,
        )
        self._threat_scorer      = ThreatScorer()

        logger.info("DetectionPipeline initialized with all four layers.")

    def analyze(
        self,
        raw_prompt: str,
        session_id: Optional[str] = None,
        source_ip: Optional[str] = None,
    ) -> dict:
        """
        Run the full detection pipeline on a raw prompt.

        Returns:
            dict with all scores, decision, flags, explanation, and metadata.
        """
        start = time.perf_counter()
        timestamp = datetime.now(timezone.utc)

        # ── Layer 0: Preprocessing ────────────────────────────────────────
        clean_prompt, transforms = normalize_text(raw_prompt)
        prompt_hash = hashlib.sha256(clean_prompt.encode()).hexdigest()

        logger.debug(
            "Analyzing prompt [hash=%s, len=%d, transforms=%s]: %s",
            prompt_hash[:8], len(clean_prompt),
            transforms, truncate_for_log(clean_prompt, 120),
        )

        # ── Layer 1: Pattern Detection ────────────────────────────────────
        pattern_result = self._pattern_detector.analyze(clean_prompt)

        # ── Layer 2: Semantic Analysis ────────────────────────────────────
        semantic_result = self._semantic_analyzer.analyze(clean_prompt)

        # ── Layer 3: Behavioral Analysis ─────────────────────────────────
        behavioral_result = self._behavioral_analyzer.analyze(
            text=clean_prompt,
            pattern_score=pattern_result.score,
            pattern_categories=list(
                {d["category"] for d in pattern_result.details}
            ),
        )

        # ── Layer 4: Anomaly Detection ────────────────────────────────────
        anomaly_result = self._anomaly_detector.analyze(clean_prompt)

        # ── Layer 5: Threat Scoring ───────────────────────────────────────
        threat = self._threat_scorer.compute(
            pattern_score    = pattern_result.score,
            semantic_score   = semantic_result.score,
            behavioral_score = behavioral_result.score,
            anomaly_score    = anomaly_result.score,
            matched_patterns = pattern_result.matched_rules,
            behavioral_flags = behavioral_result.behavioral_flags,
            top_category     = pattern_result.top_category,
            prompt           = clean_prompt,
        )

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "Analysis complete [hash=%s] score=%.1f decision=%s in %.1fms",
            prompt_hash[:8], threat.final_score, threat.decision, elapsed_ms,
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

            # Scores
            "pattern_score":    threat.pattern_score,
            "semantic_score":   threat.semantic_score,
            "behavioral_score": threat.behavioral_score,
            "anomaly_score":    threat.anomaly_score,
            "final_score":      threat.final_score,

            # Decision
            "decision":         threat.decision,
            "threat_category":  threat.threat_category,
            "sanitized_prompt": threat.sanitized_prompt,

            # Explainability
            "matched_patterns": threat.matched_patterns,
            "behavioral_flags": threat.behavioral_flags,
            "explanation":      threat.explanation,

            # Diagnostics
            "preprocessing_transforms": transforms,
            "semantic_top_match": semantic_result.top_match,
            "anomaly_is_anomaly": anomaly_result.is_anomaly,
            "processing_ms":  elapsed_ms,
        }
