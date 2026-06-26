"""
ArgusX v9 — Threat Route Interfaces
=====================================
Abstract base classes that define the contract every routing path must satisfy.
DistilBERT or any future classifier can be inserted into PromptInjectionRoute
by subclassing BaseRoute and overriding `process()`.

Design goals:
  • Zero coupling to concrete detectors.
  • Consistent RouteResult schema used by ThreatRouter and ThreatFusion.
  • Fully backward-compatible: existing SBERT engines are unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── Shared Result Schema ─────────────────────────────────────────────────────

@dataclass
class RouteResult:
    """
    Unified result object produced by every concrete route.

    All scores are in the 0-100 range, matching the existing
    DetectionPipeline scoring convention so ThreatScorer requires
    zero changes.
    """
    route_name: str                          # e.g. "PromptInjectionRoute"
    pattern_score: float    = 0.0
    semantic_score: float   = 0.0
    behavioral_score: float = 0.0
    anomaly_score: float    = 0.0

    # Explainability pass-through
    matched_patterns: List[str]              = field(default_factory=list)
    behavioral_flags: List[str]              = field(default_factory=list)
    semantic_top_match: Optional[str]        = None
    anomaly_is_anomaly: bool                 = False

    # Extension point: route-specific metadata (e.g. DistilBERT logits)
    route_metadata: Dict[str, Any]           = field(default_factory=dict)

    # v12 Architecture additions
    binary_decision: Optional[bool]          = None
    enforcement_action: Optional[str]        = None
    final_decision: bool                     = False
    final_score: float                       = 0.0


# ─── Abstract Base Route ──────────────────────────────────────────────────────

class BaseRoute(ABC):
    """
    Contract for every threat-specific processing route.

    Concrete implementations:
      - PromptInjectionRoute  (v9: delegates to SBERT pipeline)
      - CyberThreatRoute      (v9: delegates to SBERT pipeline)

    Future implementations:
      - DistilBERTPromptInjectionRoute  (v9.1+)
      - FinetuedCyberRoute              (v10+)
    """

    @property
    @abstractmethod
    def route_name(self) -> str:
        """Human-readable route identifier used in logs and RouteResult."""
        ...

    @abstractmethod
    def process(self, raw_prompt: str, clean_prompt: str, pattern_score: float,
                pattern_categories: List[str]) -> RouteResult:
        """
        Run all route-specific detection logic.

        Args:
            raw_prompt:         Original unmodified prompt text.
            clean_prompt:       Preprocessed (normalized) prompt text.
            pattern_score:      Score already produced by PatternDetector
                                (shared across all routes — computed once).
            pattern_categories: Top threat categories from PatternDetector.

        Returns:
            RouteResult with all four sub-scores populated.
        """
        ...
