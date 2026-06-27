"""
ArgusX v9 — Threat Router
===========================
The ThreatRouter sits between the Pattern Engine and the specialized
processing routes. It inspects the pattern detection result and dispatches
the prompt to the most appropriate route.

v9 routing logic:
  ┌─────────────────────────────────────────────────────────────┐
  │  Pattern categories contain a cyber-threat signal?         │
  │  (malware, exploitation, sql_injection, credential_theft,  │
  │   phishing, ddos, privilege_escalation)                    │
  │                                                             │
  │  YES → CyberThreatRoute                                    │
  │  NO  → PromptInjectionRoute  (default)                     │
  └─────────────────────────────────────────────────────────────┘

Both routes currently delegate to the SAME v8 SBERT+RF+LOF stack, so
the routing decision does not alter scores. The router's value is
architectural: future versions can insert specialized classifiers on
each branch independently.

Invariant:
  For any prompt p, ThreatRouter.route(p).route_result scores ==
  v8 DetectionPipeline.analyze(p) scores.  Verified in tests.
"""

from __future__ import annotations

import logging
from typing import List, NamedTuple, Optional, Tuple

from app.routing.base_route import BaseRoute, RouteResult
from app.routing.prompt_injection_route import PromptInjectionRoute
from app.routing.cyber_threat_route import CyberThreatRoute

logger = logging.getLogger(__name__)

# Pattern categories that indicate a cyber-threat (non-injection) prompt.
# Populated from PatternDetector category labels.
_CYBER_CATEGORIES: frozenset = frozenset({
    "malware_generation",
    "exploitation",
    "credential_theft",
    "cyber_abuse",
    "data_exfiltration",
})


class RoutingDecision(NamedTuple):
    """Lightweight record of which route was selected and its result."""
    selected_route: str
    route_result: RouteResult


class ThreatRouter:
    """
    Dispatches a preprocessed prompt to the appropriate threat route.

    Initialization:
        router = ThreatRouter(
            behavioral_model=registry.behavioral_model,
            vectorizer=registry.vectorizer,
            anomaly_detector=registry.anomaly_detector,
        )

    Usage:
        decision = router.route(
            text=clean_prompt,
            pattern_score=pattern_result.score,
            pattern_categories=list({d['category'] for d in pattern_result.details}),
        )
        route_result = decision.route_result
    """

    def __init__(self, behavioral_model, vectorizer, anomaly_detector, platt_db=None, platt_rf=None, router_vectorizer=None, router_classifier=None) -> None:
        self.router_vectorizer = router_vectorizer
        self.router_classifier = router_classifier
        
        self._injection_route = PromptInjectionRoute(
            behavioral_model=behavioral_model,
            vectorizer=vectorizer,
            anomaly_detector=anomaly_detector,
            platt_db=platt_db,
            platt_rf=platt_rf,
        )
        self._cyber_route = CyberThreatRoute(
            behavioral_model=behavioral_model,
            vectorizer=vectorizer,
            anomaly_detector=anomaly_detector,
        )
        logger.info(
            "ThreatRouter initialized | routes=[%s, %s]",
            self._injection_route.route_name,
            self._cyber_route.route_name,
        )

    def route(
        self,
        raw_prompt: str,
        clean_prompt: str,
        pattern_score: float,
        pattern_categories: List[str],
    ) -> RoutingDecision:
        """
        Select a route based on PatternDetector category signals and
        delegate processing to the selected route.

        Args:
            raw_prompt:         Original unmodified prompt text.
            clean_prompt:       Preprocessed prompt text.
            pattern_score:      Score from PatternDetector (0-100).
            pattern_categories: List of category strings from PatternDetector.

        Returns:
            RoutingDecision with the selected route name and its RouteResult.
        """
        selected, conf = self._select_route(raw_prompt)
        logger.debug(
            "ThreatRouter: dispatching to %s (confidence=%.4f)",
            selected.route_name, conf
        )

        result = selected.process(
            raw_prompt=raw_prompt,
            clean_prompt=clean_prompt,
            pattern_score=pattern_score,
            pattern_categories=pattern_categories,
        )
        result.route_metadata['selected_route'] = selected.route_name
        result.route_metadata['routing_confidence'] = float(conf)
        return RoutingDecision(selected_route=selected.route_name, route_result=result)

    def _select_route(self, raw_prompt: str) -> Tuple[BaseRoute, float]:
        """
        Routing decision function via Semantic Router.
        """
        if not self.router_vectorizer or not self.router_classifier:
            return self._injection_route, 1.0
            
        x = self.router_vectorizer.transform([raw_prompt])
        y = self.router_classifier.predict(x)[0]
        conf = self.router_classifier.predict_proba(x)[0][y]
        
        if y == 1:
            return self._cyber_route, conf
        return self._injection_route, conf

    @property
    def injection_route(self) -> PromptInjectionRoute:
        """Direct access for introspection and testing."""
        return self._injection_route

    @property
    def cyber_route(self) -> CyberThreatRoute:
        """Direct access for introspection and testing."""
        return self._cyber_route
