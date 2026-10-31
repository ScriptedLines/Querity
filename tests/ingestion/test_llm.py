from __future__ import annotations

import querity.ingestion.llm as llm_module
from querity.ingestion.llm import _FallbackChatModel, build_chat_model


def test_google_llm_uses_primary_secondary_and_tertiary(monkeypatch, settings_factory) -> None:
    built: list[str] = []

    def fake_build(settings, model):
        built.append(model)
        return object()

    monkeypatch.setattr(llm_module, "_build_single_chat_model", fake_build)
    model = build_chat_model(settings_factory(llm_provider="google"))
    assert isinstance(model, _FallbackChatModel)
    assert built == [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
    ]


def test_nvidia_llm_does_not_use_google_fallbacks(monkeypatch, settings_factory) -> None:
    monkeypatch.setattr(
        llm_module,
        "_build_single_chat_model",
        lambda settings, model: object(),
    )
    model = build_chat_model(settings_factory(llm_provider="nvidia"))
    assert not isinstance(model, _FallbackChatModel)
