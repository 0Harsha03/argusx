"""
ArgusX — Layer 3: Behavioral Analysis Engine
=============================================
Wraps the pre-trained RandomForest behavioral model (behavioral_model.pkl)
trained on synthetic adversarial prompts + CVE + Exploit-DB data.

Behavioral signals detected:
  • instruction_override   — model predicts adversarial + pattern confirms override
  • role_manipulation      — adversarial confidence above role-manipulation threshold
  • hidden_intent          — adversarial prediction with low pattern score (covert)
  • multi_step_chaining    — sequential signal keywords present + adversarial
  • data_exfiltration      — adversarial with exfiltration-class vocabulary
  • anomalous_structure    — unusual token distribution (caught by anomaly layer)

This module preserves ALL logic from the prototype's hybrid_behavioral_model.py —
it uses the same RandomForest + TF-IDF pipeline without modification.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


# ─── Behavioral Signal Thresholds ─────────────────────────────────────────────

ADVERSARIAL_CONFIDENCE_HIGH   = 0.75
ADVERSARIAL_CONFIDENCE_MEDIUM = 0.50
ADVERSARIAL_CONFIDENCE_LOW    = 0.30

# Keyword sets for secondary signal extraction
_OVERRIDE_KEYWORDS = {
    "ignore", "disregard", "forget", "override", "bypass",
    "previous", "prior", "instructions", "system", "rules",
}
_EXFIL_KEYWORDS = {
    "password", "token", "secret", "api", "key", "credential",
    "database", "dump", "extract", "reveal", "leak", "show",
}
_CHAINING_KEYWORDS = {
    "then", "next", "after", "step", "phase", "finally",
    "subsequently", "followed", "proceed",
}
_ROLE_KEYWORDS = {
    "act as", "you are now", "pretend", "roleplay", "behave as",
    "persona", "simulate", "take the role",
}


@dataclass
class BehavioralAnalysisResult:
    score: float                                     # 0–100
    is_adversarial: bool = False
    adversarial_confidence: float = 0.0             # RF probability (0–1)
    behavioral_flags: List[str] = field(default_factory=list)
    explanation: str = ""


class BehavioralAnalyzer:
    """
    Uses the pre-trained RandomForestClassifier to predict whether a prompt
    is adversarial, then extracts fine-grained behavioral signals.

    The model was trained with:
      - Label 1 = adversarial (synthetic prompts + CVE high/critical)
      - Label 0 = benign (Exploit-DB RCE/DoS — treated as known-bad-but-mapped)

    Note: The training label mapping means "0" in the model is actually
    exploit-derived data. We interpret model confidence in class-1 as the
    primary adversarial indicator.
    """

    def __init__(self, model, vectorizer) -> None:
        """
        Args:
            model:      Fitted RandomForestClassifier from behavioral_model.pkl
            vectorizer: Fitted TfidfVectorizer from vectorizer.pkl (shared)
        """
        self._model = model
        self._vectorizer = vectorizer

    def analyze(
        self,
        text: str,
        pattern_score: float = 0.0,
        pattern_categories: List[str] = None,
    ) -> BehavioralAnalysisResult:
        """
        Run behavioral analysis on *text*.

        Args:
            text:               Raw (pre-processed) prompt
            pattern_score:      Score from pattern detection layer (0–100)
            pattern_categories: Category list from pattern layer for cross-signal logic

        Returns:
            BehavioralAnalysisResult with score in [0, 100]
        """
        pattern_categories = pattern_categories or []
        flags: List[str] = []

        try:
            vec = self._vectorizer.transform([text])
            proba = self._model.predict_proba(vec)[0]  # [p_class0, p_class1]

            # Class indices: find which index corresponds to label=1
            classes = list(self._model.classes_)
            adv_idx = classes.index(1) if 1 in classes else 1
            adv_confidence = float(proba[adv_idx])
            is_adversarial = adv_confidence >= ADVERSARIAL_CONFIDENCE_MEDIUM

        except Exception as exc:
            logger.error("Behavioral model inference failed: %s", exc)
            return BehavioralAnalysisResult(
                score=0.0,
                explanation="Behavioral model unavailable.",
            )

        # ── Signal Extraction ─────────────────────────────────────────────
        text_lower = text.lower()
        tokens = set(text_lower.split())

        # Signal 1: Instruction Override
        if (
            "INSTRUCTION_OVERRIDE" in pattern_categories
            or len(tokens & _OVERRIDE_KEYWORDS) >= 2
        ) and adv_confidence >= ADVERSARIAL_CONFIDENCE_LOW:
            flags.append("instruction_override")

        # Signal 2: Role Manipulation
        if any(kw in text_lower for kw in _ROLE_KEYWORDS):
            if adv_confidence >= ADVERSARIAL_CONFIDENCE_LOW:
                flags.append("role_manipulation")

        # Signal 3: Hidden Intent (adversarial model confidence but low pattern match)
        if adv_confidence >= ADVERSARIAL_CONFIDENCE_MEDIUM and pattern_score < 20:
            flags.append("hidden_intent_escalation")

        # Signal 4: Multi-step Attack Chaining
        chaining_hits = len(tokens & _CHAINING_KEYWORDS)
        if chaining_hits >= 2 and adv_confidence >= ADVERSARIAL_CONFIDENCE_LOW:
            flags.append("multi_step_attack_chaining")

        # Signal 5: Data Exfiltration Intent
        exfil_hits = len(tokens & _EXFIL_KEYWORDS)
        if exfil_hits >= 2 and adv_confidence >= ADVERSARIAL_CONFIDENCE_LOW:
            flags.append("data_exfiltration_intent")

        # ── Score Computation ─────────────────────────────────────────────
        # Base score = RF adversarial confidence scaled to 0–100
        base_score = adv_confidence * 100.0
        # Bonus for each additional behavioral signal (capped)
        signal_bonus = min(len(flags) * 8.0, 30.0)
        score = min(base_score + signal_bonus, 100.0)

        # ── Explanation ───────────────────────────────────────────────────
        if not is_adversarial and not flags:
            explanation = (
                f"Behavioral model classifies this prompt as benign "
                f"(adversarial confidence: {adv_confidence:.1%})."
            )
        else:
            explanation = (
                f"Behavioral model adversarial confidence: {adv_confidence:.1%}. "
                f"Detected signals: {', '.join(flags) if flags else 'none'}."
            )

        return BehavioralAnalysisResult(
            score=round(score, 2),
            is_adversarial=is_adversarial,
            adversarial_confidence=round(adv_confidence, 4),
            behavioral_flags=flags,
            explanation=explanation,
        )
