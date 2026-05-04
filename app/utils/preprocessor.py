"""
ArgusX — Input Preprocessing Utilities
=======================================
Normalises and cleans raw prompt text before passing it to the
detection pipeline. Preserves semantic meaning while stripping
encoding tricks commonly used to evade detection.
"""

import re
import unicodedata
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ─── Unicode homoglyph map (common lookalikes) ────────────────────────────────
# Attackers often substitute Cyrillic/Greek chars that look like Latin letters.
_HOMOGLYPH_MAP: dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",  # Cyrillic
    "і": "i", "ѕ": "s", "ᴉ": "i",
    "ａ": "a", "ｂ": "b", "ｃ": "c",  # Fullwidth
    "𝐚": "a", "𝐛": "b", "𝐜": "c",  # Mathematical bold
    "\u200b": "",   # Zero-width space
    "\u200c": "",   # Zero-width non-joiner
    "\u200d": "",   # Zero-width joiner
    "\u2060": "",   # Word joiner
    "\ufeff": "",   # BOM
}

_HOMOGLYPH_RE = re.compile(
    "|".join(re.escape(k) for k in _HOMOGLYPH_MAP.keys())
)

# ─── Obfuscation patterns ─────────────────────────────────────────────────────
_LEETSPEAK_MAP = {
    "1gnor3": "ignore", "byp4ss": "bypass", "h4ck": "hack",
    "@dm1n": "admin",   "r00t": "root",     "3xpl01t": "exploit",
    "instr uct": "instruct", "i g n o r e": "ignore",
}


def normalize_text(text: str) -> Tuple[str, list[str]]:
    """
    Clean and normalise a prompt string.

    Returns:
        (cleaned_text, list_of_applied_transforms)
    """
    transforms_applied = []
    original_length = len(text)

    # 1. Unicode NFKC normalisation (decomposes ligatures, fullwidth chars)
    text = unicodedata.normalize("NFKC", text)

    # 2. Replace homoglyphs
    def _replace_homoglyph(m: re.Match) -> str:
        return _HOMOGLYPH_MAP.get(m.group(0), m.group(0))

    cleaned = _HOMOGLYPH_RE.sub(_replace_homoglyph, text)
    if cleaned != text:
        transforms_applied.append("homoglyph_substitution")
        text = cleaned

    # 3. Collapse excessive whitespace / newlines (but preserve structure)
    text_collapsed = re.sub(r"[ \t]{3,}", "  ", text)   # 3+ spaces → 2
    text_collapsed = re.sub(r"\n{4,}", "\n\n\n", text_collapsed)
    if text_collapsed != text:
        transforms_applied.append("whitespace_collapse")
        text = text_collapsed

    # 4. Decode common URL encoding tricks (%20 = space, %0A = newline)
    url_decoded = _safe_url_decode(text)
    if url_decoded != text:
        transforms_applied.append("url_decode")
        text = url_decoded

    # 5. Resolve simple leetspeak substitutions
    text_lower = text.lower()
    for leet, normal in _LEETSPEAK_MAP.items():
        if leet in text_lower:
            text = re.sub(re.escape(leet), normal, text, flags=re.IGNORECASE)
            transforms_applied.append(f"leetspeak_decode:{leet}")

    # 6. Strip null bytes and control characters (except tab/newline/CR)
    cleaned_ctrl = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    if cleaned_ctrl != text:
        transforms_applied.append("control_char_strip")
        text = cleaned_ctrl

    if len(text) != original_length:
        logger.debug(
            "Preprocessor: %d→%d chars, transforms: %s",
            original_length, len(text), transforms_applied,
        )

    return text.strip(), transforms_applied


def _safe_url_decode(text: str) -> str:
    """Attempt URL-decoding without raising on invalid sequences."""
    try:
        from urllib.parse import unquote
        decoded = unquote(text, errors="ignore")
        return decoded
    except Exception:
        return text


def truncate_for_log(text: str, max_len: int = 500) -> str:
    """Return a safe, truncated version of the prompt for logging."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"… [truncated {len(text) - max_len} chars]"
