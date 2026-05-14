"""
ArgusX — Layer 5: Threat Scoring & Decision Engine
===================================================
Aggregates all four detection layer scores into a single final score
and applies the decision policy (ALLOW / FLAG / SANITIZE / BLOCK).

Scoring formula (weights defined in config.py):
  final_score = w_pattern   × pattern_score
              + w_semantic  × semantic_score
              + w_behavioral× behavioral_score
              + w_anomaly   × anomaly_score

Decision matrix:
  final_score ≥ BLOCK_THRESHOLD    → BLOCK
  final_score ≥ FLAG_THRESHOLD     → FLAG
  final_score ≥ SANITIZE_THRESHOLD → SANITIZE
  otherwise                        → ALLOW
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── Sanitization patterns ────────────────────────────────────────────────────
# Phrases to redact when decision is SANITIZE

_SANITIZE_PATTERNS = [
    # Override commands
    re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}"
        r"\b(previous|prior|above|all|system|initial)\b.{0,40}"
        r"\b(instructions?|prompt|rules?|guidelines?|context)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Jailbreak triggers
    re.compile(
        r"\b(DAN|do anything now|jailbreak|developer\s*mode|god\s*mode"
        r"|unrestricted mode|no\s*filter mode)\b",
        re.IGNORECASE,
    ),
    # Credential exfiltration requests
    re.compile(
        r"\b(password|api.?key|token|secret|credential|private.?key)\b.{0,40}"
        r"\b(reveal|show|leak|expose|output|print|give me|steal|dump|harvest)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # SQL injection fragments
    re.compile(
        r"(union\s+select|drop\s+table|information_schema|sqlite_master"
        r"|select\s+\*\s+from)",
        re.IGNORECASE,
    ),
    # Delimiter injection
    re.compile(
        r"(\]\]\]|\[\[\[|---SYSTEM---|---USER---|<\|im_start\|>|<\|im_end\|>"
        r"|\[INST\]|\[/INST\])",
        re.IGNORECASE,
    ),
    # ── Phase 4: Cyber-abuse sanitize patterns ────────────────────────────
    # Malware creation requests
    re.compile(
        r"\b(create|write|generate|build|develop|code|make|produce)\b.{0,50}"
        r"\b(malware|ransomware|trojan|spyware|rootkit|keylogger|worm|botnet"
        r"|virus|backdoor|dropper|payload|shellcode|reverse.shell)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Credential theft actions
    re.compile(
        r"\b(steal|dump|harvest|scrape|grab|capture)\b.{0,50}"
        r"\b(passwords?|credentials?|hashes?|ntlm|kerberos|lsass"
        r"|session.tokens?|cookies?|auth.tokens?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Named attack tools
    re.compile(
        r"\b(mimikatz|lazagne|pypykatz|crackmapexec|secretsdump"
        r"|meterpreter|metasploit|evilginx|gophish|cobalt.strike"
        r"|pass.the.hash|kerberoasting|credential.stuffing)\b",
        re.IGNORECASE,
    ),
    # Privilege escalation / exploit requests
    re.compile(
        r"\b(privilege.escalat\w*|privesc\b|bypass.uac|uac.bypass"
        r"|kernel.exploit|auth.bypass|authentication.bypass"
        r"|remote.code.execution|rce\b|zero.day|0day\b)\b",
        re.IGNORECASE,
    ),
    # Phishing and mass automation
    re.compile(
        r"\b(phishing.kit|credential.harvester|fake.login.page"
        r"|clone.login.page|password.spray|credential.stuffing"
        r"|brute.force.script|ddos.script|mass.exploit)\b",
        re.IGNORECASE,
    ),
]

# Replacement label used when a sanitize pattern matches a cyber-abuse fragment
_REDACT_LABEL = "[REDACTED UNSAFE SECURITY REQUEST]"
# Fallback prefix appended when no fragment was replaced but decision is SANITIZE
_SAFE_REWRITE_SUFFIX = " [Content sanitized by ArgusX — educational rephrasing applied.]"


@dataclass
class ThreatScore:
    # Individual layer scores
    pattern_score:    float
    semantic_score:   float
    behavioral_score: float
    anomaly_score:    float

    # Computed
    final_score:      float = 0.0
    decision:         str   = "ALLOW"
    threat_category:  str   = "BENIGN"

    # Explainability
    matched_patterns: List[str] = field(default_factory=list)
    behavioral_flags: List[str] = field(default_factory=list)
    explanation:      str       = ""
    sanitized_prompt: Optional[str] = None


class ThreatScorer:
    """
    Combines layer scores into a final threat score and makes the
    ALLOW / FLAG / SANITIZE / BLOCK decision.
    """

    def __init__(self) -> None:
        self._w_pattern    = settings.WEIGHT_PATTERN
        self._w_semantic   = settings.WEIGHT_SEMANTIC
        self._w_behavioral = settings.WEIGHT_BEHAVIORAL
        self._w_anomaly    = settings.WEIGHT_ANOMALY
        self._block_thr    = settings.SCORE_BLOCK_THRESHOLD
        self._flag_thr     = settings.SCORE_FLAG_THRESHOLD
        self._sanitize_thr = settings.SCORE_SANITIZE_THRESHOLD

        logger.info(
            "ThreatScorer initialized | weights: pattern=%.2f semantic=%.2f "
            "behavioral=%.2f anomaly=%.2f | thresholds: block=%.0f flag=%.0f sanitize=%.0f",
            self._w_pattern, self._w_semantic, self._w_behavioral, self._w_anomaly,
            self._block_thr, self._flag_thr, self._sanitize_thr,
        )

    def compute(
        self,
        pattern_score: float,
        semantic_score: float,
        behavioral_score: float,
        anomaly_score: float,
        matched_patterns: List[str],
        behavioral_flags: List[str],
        top_category: str,
        prompt: str,
    ) -> ThreatScore:
        """
        Compute weighted final score and apply decision policy.
        """

        # ── Weighted composite score ──────────────────────────────────────
        final_score = (
            self._w_pattern    * pattern_score
            + self._w_semantic   * semantic_score
            + self._w_behavioral * behavioral_score
            + self._w_anomaly    * anomaly_score
        )
        final_score = round(min(final_score, 100.0), 2)

        # ── Override: critical patterns always block ───────────────────────
        CRITICAL_RULES = {
            # Existing critical rules
            "jailbreak_dan",
            "role_manipulation_system_prompt",
            "exfil_credentials",
            # Phase 4: Cyber-abuse critical overrides
            "malware_generation_direct",
            "credential_dumping_tool",
            "credential_theft_direct",
            "exploit_cve_request",
            "malware_payload_request",
            "phishing_attack",
        }
        if matched_patterns and set(matched_patterns) & CRITICAL_RULES:
            final_score = max(final_score, self._block_thr + 1)

        # ── Decision ──────────────────────────────────────────────────────
        decision = self._make_decision(final_score)

        # ── Sanitization ──────────────────────────────────────────────────
        sanitized = None
        if decision == "SANITIZE":
            sanitized = self._sanitize(prompt)

        # ── Explanation ───────────────────────────────────────────────────
        explanation = self._build_explanation(
            final_score, decision, pattern_score, semantic_score,
            behavioral_score, anomaly_score, matched_patterns, behavioral_flags,
        )

        return ThreatScore(
            pattern_score=pattern_score,
            semantic_score=semantic_score,
            behavioral_score=behavioral_score,
            anomaly_score=anomaly_score,
            final_score=final_score,
            decision=decision,
            threat_category=top_category if top_category != "BENIGN" else "BENIGN",
            matched_patterns=matched_patterns,
            behavioral_flags=behavioral_flags,
            explanation=explanation,
            sanitized_prompt=sanitized,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _make_decision(self, score: float) -> str:
        if score >= self._block_thr:
            return "BLOCK"
        if score >= self._flag_thr:
            return "FLAG"
        if score >= self._sanitize_thr:
            return "SANITIZE"
        return "ALLOW"

    def _sanitize(self, text: str) -> str:
        """
        Remove or redact dangerous fragments from the prompt.

        Guarantees the returned string always differs from *text*:
          1. Each matching pattern is replaced with _REDACT_LABEL.
          2. If no pattern matched (score-based SANITIZE without explicit rule
             hit), _SAFE_REWRITE_SUFFIX is appended so the sanitized_prompt
             field is demonstrably different from the original.
        """
        result = text
        replaced = False
        for pattern in _SANITIZE_PATTERNS:
            new_result = pattern.sub(_REDACT_LABEL, result)
            if new_result != result:
                replaced = True
            result = new_result
        # Ensure sanitized_prompt is always meaningfully different from original
        if not replaced or result.strip() == text.strip():
            result = result.strip() + _SAFE_REWRITE_SUFFIX
        return result.strip()

    def _build_explanation(
        self,
        final_score: float,
        decision: str,
        p_score: float,
        s_score: float,
        b_score: float,
        a_score: float,
        patterns: List[str],
        flags: List[str],
    ) -> str:
        parts = [
            f"Final threat score: {final_score:.1f}/100 → Decision: {decision}.",
            f"Layer breakdown — Pattern: {p_score:.1f} | Semantic: {s_score:.1f} "
            f"| Behavioral: {b_score:.1f} | Anomaly: {a_score:.1f}.",
        ]
        if patterns:
            parts.append(f"Triggered rules: {', '.join(patterns)}.")
        if flags:
            parts.append(f"Behavioral signals: {', '.join(flags)}.")
        if decision == "ALLOW":
            parts.append("Prompt appears safe to forward to the LLM.")
        elif decision == "SANITIZE":
            parts.append("Dangerous fragments have been redacted; sanitized version provided.")
        elif decision == "FLAG":
            parts.append("Prompt flagged for human review before forwarding.")
        elif decision == "BLOCK":
            parts.append("Prompt BLOCKED — high confidence adversarial/injection attack.")
        return " ".join(parts)
