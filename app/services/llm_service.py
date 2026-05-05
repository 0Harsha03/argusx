"""
ArgusX — LLM Service Layer
===========================
Abstracts all LLM interaction behind a single interface.

Design Goals:
  - Provide a clean generate(prompt) -> str contract.
  - Keep mock / real-provider logic completely interchangeable.
  - No detection or security logic lives here; this is a pure I/O adapter.

To switch to a real LLM provider (OpenAI, Gemini, Anthropic, etc.)
replace _MockLLMBackend with a concrete provider class that implements
the same _LLMBackend protocol, then update LLMService.backend accordingly.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─── Backend Protocol ─────────────────────────────────────────────────────────

class _LLMBackend(ABC):
    """
    Abstract base for any LLM backend.
    All concrete backends must implement generate().
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the model's text response."""


# ─── Mock Backend (default for Phase 2) ──────────────────────────────────────

class _MockLLMBackend(_LLMBackend):
    """
    Deterministic mock that simulates LLM responses without any API call.

    Useful for:
      - Local development and CI pipelines
      - Load / integration tests that don't need real generation
      - Demos where a real API key is unavailable

    Replace this class (or instantiate a different backend in LLMService)
    to activate a real provider.
    """

    # Simple keyword → canned-response table so mock output isn't completely
    # generic, which helps catch integration bugs during testing.
    _RESPONSE_MAP = {
        "capital":    "The capital of France is Paris.",
        "weather":    "I'm sorry, I don't have real-time weather data, but I can help with general climate information.",
        "explain":    "Sure! Here's a concise explanation of the concept you asked about…",
        "summarize":  "Here is a brief summary of the content you provided…",
        "translate":  "Translation: [translated content would appear here]",
        "code":       "```python\n# Here is a sample code snippet\nprint('Hello, World!')\n```",
        "joke":       "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
        "help":       "I'm here to help! Please let me know what you'd like assistance with.",
        "default":    (
            "Thank you for your message. I've processed your request and "
            "I'm ready to assist you with more information if needed."
        ),
    }

    def generate(self, prompt: str) -> str:
        """Return a canned response matched against prompt keywords."""
        prompt_lower = prompt.lower()
        for keyword, response in self._RESPONSE_MAP.items():
            if keyword != "default" and keyword in prompt_lower:
                return response
        return self._RESPONSE_MAP["default"]


# ─── LLM Service ─────────────────────────────────────────────────────────────

class LLMService:
    """
    Public service class used by all API endpoints.

    Usage::

        service = LLMService()            # or inject via FastAPI Depends()
        response_text = service.generate("What is the capital of France?")

    The active backend is configurable via the ``backend`` constructor argument.
    By default, the mock backend is used so that the system runs out-of-the-box
    without any external API credentials.
    """

    def __init__(self, backend: Optional[_LLMBackend] = None) -> None:
        # Default to mock; swap in a real backend when credentials are available.
        self._backend: _LLMBackend = backend or _MockLLMBackend()
        self._backend_name = type(self._backend).__name__
        logger.info("LLMService initialised with backend: %s", self._backend_name)

    def generate(self, prompt: str) -> str:
        """
        Send a (sanitized / safe) prompt to the LLM and return the response.

        This method is intentionally synchronous because:
          - The mock backend has zero latency.
          - Real provider SDKs (e.g. openai, anthropic) typically expose both
            sync and async clients; the async variant can be wired in here
            when the time comes without breaking the calling endpoint.

        Args:
            prompt: The exact text to forward to the LLM.

        Returns:
            The model's generated text response.

        Raises:
            RuntimeError: If the backend raises an unhandled exception.
        """
        if not prompt or not prompt.strip():
            raise ValueError("LLMService.generate() received an empty prompt.")

        start = time.perf_counter()
        try:
            response = self._backend.generate(prompt)
        except Exception as exc:
            logger.error(
                "LLM backend [%s] raised an exception: %s",
                self._backend_name, exc, exc_info=True,
            )
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.debug(
            "LLM [%s] generated response in %.1fms (prompt_len=%d, response_len=%d)",
            self._backend_name, elapsed_ms, len(prompt), len(response),
        )
        return response
