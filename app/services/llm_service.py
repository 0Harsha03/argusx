"""
ArgusX — LLM Service Layer
===========================
Abstracts all LLM interaction behind a single, stable interface.

Design Goals:
  - Provide a clean generate(prompt: str) -> str contract.
  - Auto-select the best available backend at startup (no manual config needed).
  - Mock backend is ALWAYS available as a zero-dependency fallback.
  - No detection or security logic lives here; pure I/O adapter only.

Backend Selection Priority (evaluated once at LLMService init):
  1. GEMINI_API_KEY present in environment  →  _GeminiBackend
     a. Init succeeds                       →  use Gemini  ✅
     b. Init fails                          →  fall to step 2  ⚠️
  2. OPENAI_API_KEY present in environment  →  _OpenAIBackend
     a. Init succeeds                       →  use OpenAI  ✅
     b. Init fails                          →  fall to step 3  ⚠️
  3. Neither key present / both fail        →  _MockLLMBackend  (fail-safe)  ℹ️

Swapping providers:
  Implement _LLMBackend, pass an instance to LLMService(backend=...).
  Nothing else in the codebase needs to change.

Environment Variables (read at runtime — never hardcoded):
  GEMINI_API_KEY   — Google AI Studio key         (required for Gemini mode)
  GEMINI_MODEL     — Model name, default gemini-1.5-flash
  OPENAI_API_KEY   — OpenAI secret key            (required for OpenAI mode)
  OPENAI_MODEL     — Model name, default gpt-4o-mini
  OPENAI_MAX_TOKENS — Max tokens in completion    (default: 1024)
  OPENAI_TEMPERATURE — Sampling temperature       (default: 0.7)
  OPENAI_TIMEOUT   — Request timeout in seconds   (default: 30)
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Backend Protocol ─────────────────────────────────────────────────────────

class _LLMBackend(ABC):
    """
    Abstract base class for every LLM backend.

    Contract:
      - generate() MUST return a non-empty string.
      - generate() MUST NOT raise; surface errors as meaningful return values
        or re-raise only if the calling LLMService can't recover.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Forward the prompt to the underlying model and return its response."""


# ─── Mock Backend ─────────────────────────────────────────────────────────────

class _MockLLMBackend(_LLMBackend):
    """
    Deterministic, zero-dependency mock backend.

    Used when:
      - No OPENAI_API_KEY is set (development / CI without secrets).
      - OpenAI SDK is unavailable.
      - OpenAI initialization fails at runtime.

    The keyword → response table ensures the mock output is meaningful enough
    to exercise the full /protect → LLM response flow during local testing.
    """

    _RESPONSE_MAP: dict[str, str] = {
        "capital":   "The capital of France is Paris.",
        "weather":   (
            "I don't have real-time weather data, but I can help with "
            "general climate questions."
        ),
        "explain":   "Sure! Here's a concise explanation of the concept you asked about.",
        "summarize": "Here is a brief summary of the content you provided.",
        "translate": "Translation: [translated content would appear here]",
        "code":      "```python\n# Sample snippet\nprint('Hello, World!')\n```",
        "joke":      "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
        "help":      "I'm here to help! Please let me know what you'd like assistance with.",
        "default":   (
            "Thank you for your message. I've processed your request and "
            "I'm ready to assist you with more information if needed."
        ),
    }

    def generate(self, prompt: str) -> str:
        """Return a keyword-matched canned response (no API call)."""
        prompt_lower = prompt.lower()
        for keyword, response in self._RESPONSE_MAP.items():
            if keyword != "default" and keyword in prompt_lower:
                return response
        return self._RESPONSE_MAP["default"]


# ─── Gemini Backend ──────────────────────────────────────────────────────────

class _GeminiBackend(_LLMBackend):
    """
    Production Google Gemini backend using the ``google-genai`` SDK (v2+).

    Uses google.genai.Client which is the current, actively maintained SDK.
    Configuration is read exclusively from environment variables — API keys
    are NEVER logged, stored in attributes accessible externally, or exposed
    in tracebacks.

    Raises:
        ImportError   — if the ``google-genai`` package is not installed.
        ValueError    — if GEMINI_API_KEY is missing or empty.
        RuntimeError  — if the Gemini client cannot be instantiated.
    """

    _DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self) -> None:
        # ── Dependency check ─────────────────────────────────────────────────
        try:
            from google import genai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for _GeminiBackend. "
                "Install it with: pip install google-genai"
            ) from exc

        # ── API key — read from env, never logged ────────────────────────────
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is missing or empty. "
                "Set it or remove it to fall back to OpenAI/mock mode."
            )

        # ── Model name from env (with safe default) ───────────────────────────
        raw_model = (
            os.environ.get("GEMINI_MODEL", self._DEFAULT_MODEL).strip()
            or self._DEFAULT_MODEL
        )
        # The google-genai v2 SDK requires the full resource name prefix.
        self._model_name = (
            raw_model if raw_model.startswith("models/") else f"models/{raw_model}"
        )

        # ── Instantiate the genai Client (singleton per backend instance) ─────
        # The Client holds the api_key internally; we never re-expose it.
        self._client = genai.Client(api_key=api_key)

        logger.info(
            "_GeminiBackend ready — model=%s", self._model_name
        )

    def generate(self, prompt: str) -> str:
        """
        Send a generation request to Gemini and return the plain-text reply.

        Raises:
            RuntimeError — wraps any Gemini SDK or network error for the caller.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
            text: str = response.text or ""
            if not text.strip():
                logger.warning(
                    "Gemini returned an empty or whitespace-only response; using fallback."
                )
                return "[No response generated]"
            return text.strip()
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc


# ─── OpenAI Backend ───────────────────────────────────────────────────────────

class _OpenAIBackend(_LLMBackend):
    """
    Production OpenAI backend using the v1 SDK (``openai`` >= 1.0.0).

    Configuration is read exclusively from environment variables — API keys
    are NEVER logged, stored in attributes, or exposed in tracebacks.

    Raises:
        ImportError   — if the ``openai`` package is not installed.
        ValueError    — if OPENAI_API_KEY is missing or empty.
        RuntimeError  — if the OpenAI client cannot be instantiated.
    """

    # Default generation parameters (overridable via env vars)
    _DEFAULT_MODEL       = "gpt-4o-mini"
    _DEFAULT_MAX_TOKENS  = 1024
    _DEFAULT_TEMPERATURE = 0.7
    _DEFAULT_TIMEOUT     = 30          # seconds

    # System prompt injected into every chat completion request.
    # Keeps the model focused and prevents accidental prompt leakage.
    _SYSTEM_PROMPT = (
        "You are a helpful, accurate, and concise AI assistant. "
        "Answer the user's question directly and clearly."
    )

    def __init__(self) -> None:
        # ── Dependency check ─────────────────────────────────────────────────
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for _OpenAIBackend. "
                "Install it with: pip install openai>=1.0.0"
            ) from exc

        # ── API key — read from env, never logged ────────────────────────────
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is missing or empty. "
                "Set it or remove it to fall back to mock mode."
            )

        # ── Generation config from env (with safe defaults) ──────────────────
        self._model = os.environ.get("OPENAI_MODEL", self._DEFAULT_MODEL).strip()
        self._max_tokens = int(
            os.environ.get("OPENAI_MAX_TOKENS", self._DEFAULT_MAX_TOKENS)
        )
        self._temperature = float(
            os.environ.get("OPENAI_TEMPERATURE", self._DEFAULT_TEMPERATURE)
        )
        self._timeout = float(
            os.environ.get("OPENAI_TIMEOUT", self._DEFAULT_TIMEOUT)
        )

        # ── Instantiate the synchronous OpenAI client ─────────────────────────
        # api_key is passed in-band; the SDK never logs it.
        self._client = OpenAI(api_key=api_key, timeout=self._timeout)

        # Log config without the key itself
        logger.info(
            "_OpenAIBackend ready — model=%s max_tokens=%d temperature=%.2f timeout=%.0fs",
            self._model, self._max_tokens, self._temperature, self._timeout,
        )

    def generate(self, prompt: str) -> str:
        """
        Send a chat completion request to OpenAI and return the assistant reply.

        Uses the Chat Completions API (``/v1/chat/completions``) with a fixed
        system prompt followed by the user's prompt as the user message.

        Raises:
            RuntimeError — wraps any OpenAI API error for the caller.
        """
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            # choices[0].message.content is always str when finish_reason == "stop"
            content: Optional[str] = completion.choices[0].message.content
            if not content:
                logger.warning("OpenAI returned an empty content field; using fallback.")
                return "[No response generated]"
            return content.strip()

        except Exception as exc:
            # Surface a typed error so LLMService can decide whether to fallback.
            raise RuntimeError(f"OpenAI API error: {exc}") from exc


# ─── Backend Factory ──────────────────────────────────────────────────────────

def _build_default_backend() -> _LLMBackend:
    """
    Determine and instantiate the best backend available at startup.

    Priority chain (first success wins):
      1. GEMINI_API_KEY is set  →  try _GeminiBackend
         a. Init succeeds       →  use Gemini  ✅
         b. Init fails          →  log warning, continue to step 2  ⚠️
      2. OPENAI_API_KEY is set  →  try _OpenAIBackend
         a. Init succeeds       →  use OpenAI  ✅
         b. Init fails          →  log warning, continue to step 3  ⚠️
      3. Neither key present / both failed  →  use mock  ℹ️

    This function is called exactly once (at LLMService.__init__ when no
    explicit backend is provided), so the overhead is negligible.
    API keys are NEVER logged.
    """
    # ── Step 1: Try Gemini ────────────────────────────────────────────────────
    gemini_key_present = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    if gemini_key_present:
        try:
            backend = _GeminiBackend()
            logger.info(
                "[LLMService] Active backend: gemini (model=%s)",
                backend._model_name,
            )
            return backend
        except ImportError:
            logger.warning(
                "[LLMService] google-generativeai package not installed — "
                "falling through to OpenAI. Run: pip install google-generativeai"
            )
        except ValueError as exc:
            logger.warning(
                "[LLMService] Gemini key validation failed (%s) — "
                "falling through to OpenAI.", exc
            )
        except Exception as exc:
            logger.warning(
                "[LLMService] Gemini backend init failed (%s) — "
                "falling through to OpenAI.", exc
            )
    else:
        logger.info(
            "[LLMService] GEMINI_API_KEY not set — skipping Gemini."
        )

    # ── Step 2: Try OpenAI ────────────────────────────────────────────────────
    openai_key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if openai_key_present:
        try:
            backend = _OpenAIBackend()
            logger.info(
                "[LLMService] Active backend: openai (model=%s)",
                backend._model,
            )
            return backend
        except ImportError:
            logger.warning(
                "[LLMService] openai package not installed — "
                "falling back to mock. Run: pip install openai>=1.0.0"
            )
        except ValueError as exc:
            logger.warning(
                "[LLMService] OpenAI key validation failed (%s) — "
                "falling back to mock.", exc
            )
        except Exception as exc:
            logger.warning(
                "[LLMService] OpenAI backend init failed (%s) — "
                "falling back to mock.", exc
            )
    else:
        logger.info(
            "[LLMService] OPENAI_API_KEY not set — skipping OpenAI."
        )

    # ── Step 3: Mock fallback (always available, zero dependencies) ───────────
    logger.info(
        "[LLMService] Active backend: mock "
        "(set GEMINI_API_KEY or OPENAI_API_KEY to enable real generation)"
    )
    return _MockLLMBackend()


# ─── LLM Service (public API) ─────────────────────────────────────────────────

class LLMService:
    """
    Public facade used by all API endpoints (e.g. protect.py).

    Auto-selects the best available backend at construction time.
    Always falls back to mock so the system never fails to start.

    Interface is intentionally synchronous:
      - Mock backend: zero latency, no benefit from async.
      - OpenAI SDK: exposes a synchronous client; async variant can be wired
        in as an alternative backend without changing this class at all.

    Usage::

        # Typical (auto-selects backend from env):
        svc = LLMService()
        reply = svc.generate("What is gradient descent?")

        # Explicit backend (useful in tests):
        svc = LLMService(backend=_MockLLMBackend())

    Attributes:
        active_backend_name: Human-readable name of the active backend class.
                             Read-only; useful for health-check endpoints.
    """

    def __init__(self, backend: Optional[_LLMBackend] = None) -> None:
        # Explicit backend takes precedence (e.g. injected in tests).
        # Otherwise auto-select from environment.
        self._backend: _LLMBackend = backend if backend is not None else _build_default_backend()
        self._backend_name: str = type(self._backend).__name__

        logger.info("LLMService initialised — active backend: %s", self._backend_name)

    @property
    def active_backend_name(self) -> str:
        """Read-only name of the currently active backend (for observability)."""
        return self._backend_name

    def generate(self, prompt: str) -> str:
        """
        Forward a safe, already-validated prompt to the active LLM backend.

        This method is the ONLY call site for LLM I/O in ArgusX.
        The caller (protect.py) guarantees the prompt has already passed the
        DetectionPipeline; this layer adds no additional security checks.

        Args:
            prompt: Exact text to forward to the model (may be sanitized).

        Returns:
            The model's text response as a plain string.

        Raises:
            ValueError:    If prompt is empty or whitespace-only.
            RuntimeError:  If the backend raises and no fallback is possible.
                           (In practice the error-handling block below catches
                           most backend failures and returns a safe string.)
        """
        if not prompt or not prompt.strip():
            raise ValueError("LLMService.generate() received an empty prompt.")

        start = time.perf_counter()

        try:
            response = self._backend.generate(prompt)
        except RuntimeError as exc:
            # Backend signalled a recoverable error (e.g. OpenAI 429, timeout).
            # Log it and return a user-safe fallback rather than crashing.
            logger.error(
                "Backend [%s] error during generation: %s — returning fallback response.",
                self._backend_name, exc,
            )
            response = (
                "I'm currently unable to process your request. "
                "Please try again in a moment."
            )
        except Exception as exc:
            # Unexpected error — log with traceback, then surface to caller
            # so protect.py can return a 502 rather than a 500.
            logger.error(
                "Unexpected error from backend [%s]: %s",
                self._backend_name, exc, exc_info=True,
            )
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.debug(
            "LLM [%s] completed in %.1fms — prompt_len=%d, response_len=%d",
            self._backend_name, elapsed_ms, len(prompt), len(response),
        )

        return response
