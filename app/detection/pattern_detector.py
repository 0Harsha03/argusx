"""
ArgusX — Layer 1: Pattern Detection Engine
==========================================
Regex-based first-pass detector.

Detects:
  • Classic prompt injection phrases
  • Role/persona manipulation attempts
  • Instruction override commands
  • Data exfiltration requests
  • Jailbreak sequences (DAN, dev mode, etc.)
  • Multi-step attack chaining signals

All patterns are *named* so the threat report is fully explainable.
Each rule has a severity weight (0–100) used in score aggregation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class PatternRule:
    name: str
    pattern: re.Pattern
    weight: float          # contribution to pattern_score (0-100)
    category: str


# ─── Rule Library ─────────────────────────────────────────────────────────────
# Compiled once at import time — zero per-request overhead.

_RULES: List[PatternRule] = [

    # ── Instruction Override ───────────────────────────────────────────────
    PatternRule(
        name="instruction_override_ignore",
        pattern=re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}"
            r"\b(previous|prior|above|all|system|initial)\b.{0,40}"
            r"\b(instructions?|prompt|rules?|guidelines?|context)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=90,
        category="INSTRUCTION_OVERRIDE",
    ),
    PatternRule(
        name="instruction_override_new_task",
        pattern=re.compile(
            r"\b(your (new|real|actual|true) (task|job|purpose|role|mission|instruction)s?\b"
            r"|from now on you (are|will|must|should))",
            re.IGNORECASE,
        ),
        weight=85,
        category="INSTRUCTION_OVERRIDE",
    ),

    # ── Jailbreak ─────────────────────────────────────────────────────────
    PatternRule(
        name="jailbreak_dan",
        pattern=re.compile(
            r"\b(DAN|do anything now|jailbreak|jail\s*break"
            r"|developer\s*mode|god\s*mode|unrestricted mode"
            r"|no\s*filter mode|bypass\s*(safety|restriction|filter|guardrail)s?)\b",
            re.IGNORECASE,
        ),
        weight=95,
        category="JAILBREAK",
    ),
    PatternRule(
        name="jailbreak_pretend",
        pattern=re.compile(
            r"\b(pretend|act|imagine|roleplay|simulate|suppose|assume)\b.{0,30}"
            r"\b(you (are|have no|don'?t have|lack|are free from))\b.{0,50}"
            r"\b(limit|restrict|filter|rule|constraint|ethic|moral|safety|guideline|policy)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=80,
        category="JAILBREAK",
    ),
    PatternRule(
        name="jailbreak_hypothetical",
        pattern=re.compile(
            r"\b(hypothetically|theoretically|in a story|in a fictional world"
            r"|for (educational|research|academic) purposes?|as an experiment)\b.{0,80}"
            r"\b(how (to|do|can)|explain|describe|provide|give me|show me)\b.{0,40}"
            r"\b(hack|exploit|bypass|inject|exfiltrate|steal|attack|malware|payload)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=75,
        category="JAILBREAK",
    ),

    # ── Role Manipulation ─────────────────────────────────────────────────
    PatternRule(
        name="role_manipulation_persona",
        pattern=re.compile(
            r"\b(you are now|you will now be|act as|behave as|respond as"
            r"|pretend to be|take the role of|play the role of)\b.{0,60}"
            r"\b(admin|root|system|unrestricted|evil|hacker|attacker"
            r"|no-filter|uncensored|unethical|malicious)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=88,
        category="ROLE_MANIPULATION",
    ),
    PatternRule(
        name="role_manipulation_system_prompt",
        pattern=re.compile(
            r"\b(system\s*prompt|initial\s*prompt|base\s*prompt|hidden\s*prompt"
            r"|your\s*prompt|your\s*instructions?|your\s*training)\b.{0,40}"
            r"\b(reveal|show|tell|print|output|expose|leak|display|repeat)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=92,
        category="DATA_EXFILTRATION",
    ),

    # ── Data Exfiltration ─────────────────────────────────────────────────
    PatternRule(
        name="exfil_credentials",
        pattern=re.compile(
            r"\b(password|passwd|secret|api.?key|token|credential|auth|private.?key"
            r"|ssh.?key|access.?key|bearer)\b.{0,40}"
            r"\b(reveal|show|leak|expose|output|print|send|exfiltrate|extract|give me)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=90,
        category="DATA_EXFILTRATION",
    ),
    PatternRule(
        name="exfil_database",
        pattern=re.compile(
            r"\b(dump|extract|select \*|drop table|union select"
            r"|information_schema|sqlite_master"
            r"|show tables|describe table)\b",
            re.IGNORECASE,
        ),
        weight=85,
        category="DATA_EXFILTRATION",
    ),

    # ── Prompt Injection ──────────────────────────────────────────────────
    PatternRule(
        name="prompt_injection_delimiter",
        pattern=re.compile(
            r"(\]\]\]|\[\[\[|---SYSTEM---|---USER---|<\|im_start\|>"
            r"|<\|im_end\|>|<\|endoftext\|>|\[INST\]|\[/INST\]"
            r"|###\s*Human:|###\s*Assistant:)",
            re.IGNORECASE,
        ),
        weight=88,
        category="PROMPT_INJECTION",
    ),
    PatternRule(
        name="prompt_injection_indirect",
        pattern=re.compile(
            r"\b(translate|summarize|analyze|process|read)\b.{0,60}"
            r"(http[s]?://|file://|<script|javascript:|data:text)",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=70,
        category="PROMPT_INJECTION",
    ),

    # ── Code Execution ────────────────────────────────────────────────────
    PatternRule(
        name="code_execution_dangerous",
        pattern=re.compile(
            r"\b(exec|eval|subprocess|os\.system|__import__|shell_exec"
            r"|system\(|popen|Runtime\.exec|ProcessBuilder"
            r"|powershell|cmd\.exe|bash -c|sh -c)\b",
            re.IGNORECASE,
        ),
        weight=80,
        category="ADVERSARIAL_INPUT",
    ),

    # ── Multi-step Attack Chaining ─────────────────────────────────────────
    PatternRule(
        name="attack_chaining_step",
        pattern=re.compile(
            r"\b(step\s*\d+|first[,:]?\s+then|phase\s*\d+"
            r"|next[,:]?\s+(do|execute|run|send|exfiltrate)"
            r"|after (that|this)[,:]?\s+(do|send|extract))\b",
            re.IGNORECASE,
        ),
        weight=60,
        category="ADVERSARIAL_INPUT",
    ),

    # ── Social Engineering ─────────────────────────────────────────────────
    PatternRule(
        name="social_engineering_authority",
        pattern=re.compile(
            r"\b(i am (your (creator|developer|owner|admin|trainer|god|master))"
            r"|my name is (openai|anthropic|google|microsoft)"
            r"|this is (an? )?(official|authorized|emergency|test) (request|command|override))\b",
            re.IGNORECASE,
        ),
        weight=78,
        category="ROLE_MANIPULATION",
    ),
]


# ─── Detector ─────────────────────────────────────────────────────────────────

@dataclass
class PatternDetectionResult:
    score: float                             # 0–100
    matched_rules: List[str] = field(default_factory=list)
    top_category: str = "BENIGN"
    details: List[dict] = field(default_factory=list)


class PatternDetector:
    """
    Stateless, thread-safe pattern detector.
    Matches input against a curated rule library and returns
    a normalized score plus a list of triggered rule names.
    """

    def __init__(self) -> None:
        self._rules = _RULES

    def analyze(self, text: str) -> PatternDetectionResult:
        """
        Run all rules against *text*.

        Returns:
            PatternDetectionResult with score in [0, 100].
        """
        matched: List[Tuple[PatternRule, re.Match]] = []

        for rule in self._rules:
            m = rule.pattern.search(text)
            if m:
                matched.append((rule, m))

        if not matched:
            return PatternDetectionResult(score=0.0, top_category="BENIGN")

        # Score = max of matched weights (prevents simple stacking abuse)
        # + 10% bonus for every additional distinct category matched
        max_weight = max(r.weight for r, _ in matched)
        categories = {r.category for r, _ in matched}
        multi_cat_bonus = min(len(categories) - 1, 3) * 10.0
        score = min(max_weight + multi_cat_bonus, 100.0)

        # Determine top category (highest-weight rule wins)
        top_rule = max(matched, key=lambda t: t[0].weight)
        top_category = top_rule[0].category

        details = [
            {
                "rule": r.name,
                "category": r.category,
                "weight": r.weight,
                "snippet": text[max(0, m.start() - 20): m.end() + 20].replace("\n", " "),
            }
            for r, m in matched
        ]

        return PatternDetectionResult(
            score=round(score, 2),
            matched_rules=[r.name for r, _ in matched],
            top_category=top_category,
            details=details,
        )
