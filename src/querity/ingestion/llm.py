from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from querity.ingestion.settings import IngestionSettings


class _FallbackChatModel:
    def __init__(self, models: Sequence[Any]) -> None:
        if not models:
            raise ValueError("At least one chat model is required")
        self._models = list(models)

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        primary, *fallbacks = self._models
        model = primary.with_fallbacks(fallbacks) if fallbacks else primary
        return model.invoke(messages, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        structured: list[Any] = []
        for model in self._models:
            try:
                structured.append(model.with_structured_output(schema, **kwargs))
            except Exception:
                continue
        if not structured:
            raise ValueError("No configured chat model supports structured output")
        primary, *fallbacks = structured
        return primary.with_fallbacks(fallbacks) if fallbacks else primary

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models[0], name)


def _build_google_chat_model(settings: IngestionSettings, model: str) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_output_tokens,
        "retries": 2,
        "request_timeout": 120,
    }
    if settings.google_api_key is not None:
        kwargs["api_key"] = settings.google_api_key.get_secret_value().strip()
    return ChatGoogleGenerativeAI(**kwargs)


def _build_nvidia_chat_model(settings: IngestionSettings, model: str) -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": settings.llm_temperature,
        "max_completion_tokens": settings.llm_max_output_tokens,
    }
    if settings.nvidia_api_key is not None:
        kwargs["api_key"] = settings.nvidia_api_key.get_secret_value().strip()
    return ChatNVIDIA(**kwargs)


def _build_single_chat_model(settings: IngestionSettings, model: str) -> Any:
    if settings.llm_provider == "google":
        return _build_google_chat_model(settings, model)
    if settings.llm_provider == "nvidia":
        return _build_nvidia_chat_model(settings, model)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def build_chat_model(settings: IngestionSettings) -> Any:
    """Build the chat model used by ingestion LLM features.

    The provider packages are imported lazily so heuristic-only runs do not pay
    their import cost and so optional providers stay optional at runtime.
    Gemini runs use a primary/secondary/tertiary fallback chain so quota errors
    on one model do not stop enrichment or query generation.
    """

    primary = _build_single_chat_model(settings, settings.llm_model)
    fallback_models = settings.llm_fallback_models if settings.llm_provider == "google" else []
    if not fallback_models:
        return primary
    fallbacks = [_build_single_chat_model(settings, model) for model in fallback_models]
    return _FallbackChatModel([primary, *fallbacks])
