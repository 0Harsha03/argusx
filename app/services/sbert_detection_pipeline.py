"""
ArgusX v7 — SBERT Detection Pipeline
======================================
Subclasses DetectionPipeline, replacing ONLY the SemanticAnalyzer
with SBERTSemanticAnalyzer. Every other engine is inherited unchanged.

Usage:
    registry = ModelRegistry()
    registry.load_all()
    pipeline = SBERTDetectionPipeline(registry)   # drop-in replacement
    result   = pipeline.analyze(prompt)

The public interface (analyze → dict) is 100% identical to DetectionPipeline.
"""

import logging

from app.services.detection_pipeline import DetectionPipeline
from app.services.model_registry import ModelRegistry
from app.detection.sbert_semantic_analyzer import SBERTSemanticAnalyzer

logger = logging.getLogger(__name__)


class SBERTDetectionPipeline(DetectionPipeline):
    """
    ArgusX v7: DetectionPipeline with SBERT Semantic Engine.

    Only _semantic_analyzer is swapped. All other layers (Pattern,
    Behavioral, Anomaly, ThreatScorer, Enforcement) are inherited
    directly from DetectionPipeline without modification.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        # Initialise base pipeline (TF-IDF semantic analyzer included)
        super().__init__(registry)

        # Override ONLY the semantic layer
        logger.info("v7: Replacing TF-IDF SemanticAnalyzer with SBERTSemanticAnalyzer…")
        self._semantic_analyzer = SBERTSemanticAnalyzer()
        logger.info("v7: SBERTDetectionPipeline ready.")
