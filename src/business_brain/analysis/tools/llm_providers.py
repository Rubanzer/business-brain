"""LLM Provider abstraction — Gemini, OpenAI, Anthropic behind one interface.

No external framework (no LangChain, no LiteLLM). Each provider wraps
its native SDK directly.  Synchronous generate() — the gateway wraps
with asyncio.to_thread().
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Synchronous text generation. Returns the model's text response."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...


class GeminiProvider(LLMProvider):
    """Google Gemini via google-genai SDK."""

    def __init__(self) -> None:
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.models.generate_content(
            model=kwargs.get("model", self._model),
            contents=prompt,
        )
        return response.text.strip()

    def name(self) -> str:
        return "gemini"


class OpenAIProvider(LLMProvider):
    """OpenAI via openai SDK."""

    def __init__(self) -> None:
        import openai

        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.3),
        )
        return response.choices[0].message.content.strip()

    def name(self) -> str:
        return "openai"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude via anthropic SDK."""

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.messages.create(
            model=kwargs.get("model", self._model),
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def name(self) -> str:
        return "anthropic"


# ---------------------------------------------------------------------------
# Provider resolution — singleton with auto-detection
# ---------------------------------------------------------------------------

_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Get the configured LLM provider (singleton).

    Resolution order:
    1. LLM_PROVIDER env var (explicit: "gemini" / "openai" / "anthropic")
    2. Auto-detect from which API keys are set (gemini first for backward compat)
    """
    global _provider
    if _provider is not None:
        return _provider

    explicit = (settings.llm_provider or "").lower()

    if explicit == "openai" and settings.openai_api_key:
        _provider = OpenAIProvider()
    elif explicit == "anthropic" and settings.anthropic_api_key:
        _provider = AnthropicProvider()
    elif explicit == "gemini" and settings.gemini_api_key:
        _provider = GeminiProvider()
    elif settings.gemini_api_key:
        _provider = GeminiProvider()
    elif settings.openai_api_key:
        _provider = OpenAIProvider()
    elif settings.anthropic_api_key:
        _provider = AnthropicProvider()
    else:
        raise RuntimeError(
            "No LLM provider configured. Set at least one of: "
            "GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
        )

    logger.info("LLM provider initialized: %s", _provider.name())
    return _provider


def reset_provider() -> None:
    """Reset singleton (for testing or key rotation)."""
    global _provider
    _provider = None
