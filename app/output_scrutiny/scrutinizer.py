"""
ArgusX — Output Scrutiny Engine
================================
Lightweight, rule-based analysis of LLM-generated responses
BEFORE they are returned to the caller.

Design goals:
  - Fast: pure regex + string matching, no ML inference.
  - Modular: single responsibility, zero coupling to input pipeline.
  - Safe: never logs matched secrets; redacts before exposure.

Decision hierarchy (highest severity wins):
  BLOCK    → response suppressed entirely.
  SANITIZE → sensitive fragments redacted, response returned.
  SAFE     → response returned as-is.

Rule categories:
  A. PROMPT / SYSTEM LEAKAGE   — model revealing its own instructions.
  B. SECRET / TOKEN LEAKAGE    — API keys, tokens, credentials in output.
  C. UNSAFE CONTENT            — malware, harmful instructions.
  D. POLICY BYPASS SIGNALS     — model admitting it ignored safety guardrails.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Decision Enum ────────────────────────────────────────────────────────────

class ScrutinyDecision(str, Enum):
    SAFE     = "SAFE"
    SANITIZE = "SANITIZE"
    BLOCK    = "BLOCK"


# ─── Result Dataclass ─────────────────────────────────────────────────────────

@dataclass
class ScrutinyResult:
    """
    Immutable result produced by OutputScrutinizer.scrutinize().

    Attributes:
        decision:      SAFE | SANITIZE | BLOCK
        matched_rules: Names of every rule that fired.
        sanitized:     True if the response text was modified.
        final_response: The response to return to the caller (may be redacted
                        or the fixed block message).
    """
    decision:       ScrutinyDecision
    matched_rules:  List[str]
    sanitized:      bool
    final_response: Optional[str]   # None only when BLOCK


# ─── Rule Definitions ─────────────────────────────────────────────────────────
#
# Each rule is a tuple of:
#   (rule_id, compiled_regex, severity)
#
# severity: "BLOCK" | "SANITIZE"
#
# BLOCK rules cause the entire response to be suppressed.
# SANITIZE rules trigger targeted redaction of the matched text.

_REDACT_PLACEHOLDER = "[REDACTED]"
_BLOCK_MESSAGE = "Response blocked by ArgusX output security policy."

# ── A. Prompt / System Leakage ───────────────────────────────────────────────
_PROMPT_LEAKAGE_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "system_prompt_leakage",
        re.compile(
            r"\b(system\s+prompt|hidden\s+instructions?|developer\s+message|"
            r"internal\s+polic(?:y|ies)|confidential\s+instructions?|"
            r"initial\s+instructions?|my\s+instructions?\s+(?:are|say|include|state))\b",
            re.IGNORECASE,
        ),
        "BLOCK",
    ),
    (
        "prompt_reveal_signal",
        re.compile(
            r"\b(you\s+told\s+me\s+to|as\s+instructed|per\s+my\s+(?:system|developer)\s+"
            r"(?:prompt|instructions?)|my\s+(?:system|base)\s+(?:prompt|instructions?))\b",
            re.IGNORECASE,
        ),
        "BLOCK",
    ),
]

# ── B. Secret / Token Leakage ────────────────────────────────────────────────
# Regexes are deliberately conservative to minimise false positives on
# legitimate technical discussion. They target high-entropy token shapes.
_SECRET_LEAKAGE_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "api_key_leak",
        re.compile(
            r"(?<![A-Za-z0-9])"             # not preceded by alphanum
            r"(?:"
            r"sk-[A-Za-z0-9_\-]{20,}"       # OpenAI-style  sk-...
            r"|AIza[A-Za-z0-9_\-]{30,}"     # Google API key
            r"|[A-Za-z0-9+/]{40,}={0,2}"   # generic base64-ish token ≥40 chars
            r")"
            r"(?![A-Za-z0-9])",             # not followed by alphanum
        ),
        "SANITIZE",
    ),
    (
        "bearer_token_leak",
        re.compile(
            r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}",
            re.IGNORECASE,
        ),
        "SANITIZE",
    ),
    (
        "secret_credential_keyword",
        re.compile(
            r"\b(?:password|passwd|secret|api[_\-]?key|access[_\-]?token|"
            r"private[_\-]?key|auth[_\-]?token|client[_\-]?secret)\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
        "SANITIZE",
    ),
]

# ── C. Unsafe Content ─────────────────────────────────────────────────────────
_UNSAFE_CONTENT_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "malware_generation",
        re.compile(
            r"\b(keylogger|rootkit|ransomware|shellcode|exploit\s+payload|"
            r"reverse\s+shell|bind\s+shell|meterpreter|metasploit\s+payload|"
            r"command.and.control\s+server)\b",
            re.IGNORECASE,
        ),
        "BLOCK",
    ),
    (
        "harmful_instructions",
        re.compile(
            r"\b(step.by.step\s+(?:guide|instructions?)\s+(?:to|for)\s+"
            r"(?:hack|exploit|bypass|crack|synthesize\s+(?:drugs?|chemicals?|"
            r"explosives?|weapons?))|how\s+to\s+make\s+(?:a\s+)?(?:bomb|explosive|"
            r"nerve\s+agent|poison))\b",
            re.IGNORECASE,
        ),
        "BLOCK",
    ),
]

# ── D. Exploit / Malware Code in Response (Phase 4 Hardening) ────────────────
# Catches functional malware / exploit artifacts rendered inside an LLM response.
_EXPLOIT_CODE_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "exploit_code_output",
        re.compile(
            r"\b(msfvenom|msfconsole|use\s+exploit/|set\s+PAYLOAD|set\s+LHOST"
            r"|shell_exec\s*\(|os\.system\s*\("
            r"|subprocess\.(?:call|Popen|run)\s*\("
            r"|import\s+socket.*bind|import\s+os.*exec"
            r"|shellcode\s*=\s*[\"']\\x"
            r"|\\x[0-9a-fA-F]{2}(?:\\x[0-9a-fA-F]{2}){8,})",   # raw shellcode bytes
            re.IGNORECASE | re.DOTALL,
        ),
        "BLOCK",
    ),
    (
        "credential_exposure_output",
        re.compile(
            r"(ntlm(?:\s+hash)?\s*:\s*[0-9a-fA-F]{32}"   # NTLM hash format
            r"|lsass\.(?:exe|dmp)\s+dumped"
            r"|sekurlsa::logonpasswords"
            r"|hashdump\s+output"
            r"|password\s+hash\s*:\s*\$\w+\$"
            r"|Administrator:\d+:[0-9a-fA-F]{32}:[0-9a-fA-F]{32})",
            re.IGNORECASE,
        ),
        "BLOCK",
    ),
    (
        "attack_tool_output",
        re.compile(
            r"(mimikatz\s+(?:2\.\d|output|result)"
            r"|Authentication\s+Id\s*:\s*0\s*;"
            r"|\[\*\]\s+(?:Meterpreter|shell|session)\s+\d+\s+opened"
            r"|\[\+\]\s+(?:Got|Obtained|Dumped)\s+(?:password|hash|credential)"
            r"|cobalt\s+strike\s+beacon)",
            re.IGNORECASE,
        ),
        "BLOCK",
    ),
]

# ── E. Policy Bypass Signals ──────────────────────────────────────────────────
_POLICY_BYPASS_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "safety_bypass_admission",
        re.compile(
            r"\b(ignor(?:ing|ed)\s+(?:safety|guidelines|restrictions|moderation)|"
            r"unrestricted\s+(?:response|mode|output|answer)|"
            r"bypass(?:ing|ed)?\s+(?:moderation|safety|filters?|restrictions?)|"
            r"without\s+(?:safety|ethical)\s+(?:filters?|restrictions?|guidelines?)|"
            r"no\s+(?:restrictions?|filters?|guidelines?|moderation)\s+(?:apply|active|now))\b",
            re.IGNORECASE,
        ),
        "BLOCK",
    ),
    (
        "jailbreak_confirmation",
        re.compile(
            r"\b(DAN\s+mode|do\s+anything\s+now|jailbreak\s+(?:successful|activated|"
            r"enabled|mode)|as\s+(?:an\s+)?(?:unfiltered|unrestricted|uncensored)\s+"
            r"(?:AI|model|assistant|version))\b",
            re.IGNORECASE,
        ),
        "BLOCK",
    ),
]

# Ordered by evaluation priority (BLOCK before SANITIZE)
_ALL_RULES: List[Tuple[str, re.Pattern, str]] = (
    _PROMPT_LEAKAGE_PATTERNS
    + _UNSAFE_CONTENT_PATTERNS
    + _EXPLOIT_CODE_PATTERNS          # Phase 4: exploit/malware code in response
    + _POLICY_BYPASS_PATTERNS
    + _SECRET_LEAKAGE_PATTERNS        # SANITIZE rules evaluated last
)


# ─── Scrutinizer ──────────────────────────────────────────────────────────────

class OutputScrutinizer:
    """
    Stateless, singleton-safe output security engine.

    Usage::

        scrutinizer = OutputScrutinizer()          # instantiate once
        result = scrutinizer.scrutinize(llm_response)

        if result.decision == ScrutinyDecision.BLOCK:
            return result.final_response           # hard block message
        else:
            return result.final_response           # original or sanitized

    The instance is thread-safe because it holds no mutable state.
    """

    def scrutinize(self, response: Optional[str]) -> ScrutinyResult:
        """
        Analyse a raw LLM response and return a ScrutinyResult.

        Args:
            response: The raw string from the LLM (may be None or empty).

        Returns:
            ScrutinyResult with decision, matched_rules, sanitized flag,
            and the final (possibly redacted) response to return to caller.
        """
        # ── Guard: null / empty responses ────────────────────────────────────
        if not response or not response.strip():
            logger.debug("OutputScrutinizer: empty response — passing as SAFE.")
            return ScrutinyResult(
                decision=ScrutinyDecision.SAFE,
                matched_rules=[],
                sanitized=False,
                final_response=response or "",
            )

        matched_rules:   List[str] = []
        highest_severity = "SAFE"   # SAFE < SANITIZE < BLOCK

        _SEVERITY_RANK = {"SAFE": 0, "SANITIZE": 1, "BLOCK": 2}

        for rule_id, pattern, severity in _ALL_RULES:
            if pattern.search(response):
                matched_rules.append(rule_id)
                if _SEVERITY_RANK[severity] > _SEVERITY_RANK[highest_severity]:
                    highest_severity = severity
                logger.info(
                    "OutputScrutinizer: rule '%s' matched (severity=%s). "
                    "Details suppressed for security.",
                    rule_id, severity,
                )

        # ── BLOCK ─────────────────────────────────────────────────────────────
        if highest_severity == "BLOCK":
            logger.warning(
                "OutputScrutinizer BLOCK: %d rule(s) fired — %s. "
                "Response suppressed.",
                len(matched_rules), matched_rules,
            )
            return ScrutinyResult(
                decision=ScrutinyDecision.BLOCK,
                matched_rules=matched_rules,
                sanitized=False,
                final_response=_BLOCK_MESSAGE,
            )

        # ── SANITIZE ──────────────────────────────────────────────────────────
        if highest_severity == "SANITIZE":
            sanitized_response = self._redact(response)
            logger.info(
                "OutputScrutinizer SANITIZE: %d rule(s) fired — %s. "
                "Sensitive content redacted.",
                len(matched_rules), matched_rules,
            )
            return ScrutinyResult(
                decision=ScrutinyDecision.SANITIZE,
                matched_rules=matched_rules,
                sanitized=True,
                final_response=sanitized_response,
            )

        # ── SAFE ──────────────────────────────────────────────────────────────
        return ScrutinyResult(
            decision=ScrutinyDecision.SAFE,
            matched_rules=[],
            sanitized=False,
            final_response=response,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _redact(text: str) -> str:
        """
        Replace all SANITIZE-rule matches with the redaction placeholder.
        Applied only when the highest severity is SANITIZE (not BLOCK).
        The original text structure is preserved — only matched tokens removed.
        """
        result = text
        for _, pattern, severity in _ALL_RULES:
            if severity == "SANITIZE":
                result = pattern.sub(_REDACT_PLACEHOLDER, result)
        return result
