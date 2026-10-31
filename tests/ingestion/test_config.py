from __future__ import annotations

import pytest
from pydantic import ValidationError

from querity.ingestion.settings import IngestionSettings, normalize_database_url


def test_normalize_database_url_uses_psycopg() -> None:
    assert normalize_database_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_database_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert (
        normalize_database_url("postgresql+psycopg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    )


def test_normalize_database_url_rejects_other_drivers() -> None:
    with pytest.raises(ValueError):
        normalize_database_url("mysql://u:p@h/db")


def test_csv_settings(settings_factory) -> None:
    settings = settings_factory(
        source_schemas="public, analytics",
        table_include="public.fact_*, dim_*",
        table_exclude="*_tmp",
        sensitive_column_patterns=r"(^|_)salary(_|$), (^|_)cnic(_|$)",
    )
    assert settings.schemas == ["public", "analytics"]
    assert settings.include_patterns == ["public.fact_*", "dim_*"]
    assert settings.exclude_patterns == ["*_tmp"]
    assert len(settings.extra_sensitive_patterns) == 2


def test_invalid_schema_is_rejected(settings_factory) -> None:
    settings = settings_factory(source_schemas="Public")
    with pytest.raises(ValueError):
        _ = settings.schemas


def test_hnsw_dimension_limit(settings_factory) -> None:
    with pytest.raises(ValidationError):
        settings_factory(embedding_dimensions=3_072, embedding_index="hnsw")
    settings = settings_factory(embedding_dimensions=3_072, embedding_index="none")
    assert settings.embedding_dimensions == 3_072


def test_metadata_schema_must_be_identifier(settings_factory) -> None:
    with pytest.raises(ValidationError):
        settings_factory(metadata_schema="Text-To-SQL")


def test_source_id_validation(settings_factory) -> None:
    with pytest.raises(ValidationError):
        settings_factory(source_id="bad id with spaces")


def test_source_url_required_for_pipeline(settings_factory) -> None:
    settings = settings_factory(source_database_url=None)
    with pytest.raises(ValueError):
        _ = settings.source_url


def test_env_aliases(monkeypatch, settings_factory) -> None:
    monkeypatch.setenv("SOURCE_ID", "warehouse")
    monkeypatch.setenv("LOW_CARDINALITY_THRESHOLD", "77")
    settings = settings_factory()
    assert settings.source_id == "warehouse"
    assert settings.low_cardinality_threshold == 77


def test_settings_supports_field_names(settings_factory) -> None:
    settings = settings_factory(source_id="analytics")
    assert settings.source_id == "analytics"
    assert isinstance(settings, IngestionSettings)


def test_provider_api_keys_are_loaded_from_env(monkeypatch, settings_factory) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
    settings = settings_factory()
    assert settings.google_api_key is not None
    assert settings.google_api_key.get_secret_value() == "google-secret"
    assert settings.nvidia_api_key is not None
    assert settings.nvidia_api_key.get_secret_value() == "nvidia-secret"


def test_blank_provider_api_key_is_none(monkeypatch, settings_factory) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    settings = settings_factory()
    assert settings.google_api_key is None


def test_blank_database_url_is_rejected(settings_factory) -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        settings_factory(source_database_url="")


def test_refresh_unchanged_tables_defaults_to_false(settings_factory) -> None:
    assert settings_factory().refresh_unchanged_tables is False
    assert settings_factory(refresh_unchanged_tables=True).refresh_unchanged_tables is True


def test_semantic_layer_generation_defaults_to_false(settings_factory) -> None:
    assert settings_factory().generate_semantic_layer is False
    assert settings_factory(generate_semantic_layer=True).generate_semantic_layer is True


def test_nvidia_embeddings_are_the_default(settings_factory) -> None:
    settings = settings_factory()
    assert settings.embedding_provider == "nvidia"
    assert settings.embedding_model == "nvidia/nemotron-3-embed-1b"
    assert settings.embedding_dimensions == 2_048


def test_llm_fallback_models_default_to_secondary_and_tertiary(settings_factory) -> None:
    settings = settings_factory()
    assert settings.llm_model == "gemini-3.5-flash-lite"
    assert settings.llm_fallback_models == [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
    ]


def test_llm_fallback_models_skip_duplicates_and_primary(settings_factory) -> None:
    settings = settings_factory(
        llm_model="gemini-3.5-flash-lite",
        llm_secondary_model="gemini-3.5-flash-lite",
        llm_tertiary_model="gemini-2.5-flash-lite",
    )
    assert settings.llm_fallback_models == ["gemini-2.5-flash-lite"]


def test_blank_llm_fallback_models_are_ignored(settings_factory) -> None:
    settings = settings_factory(llm_secondary_model="", llm_tertiary_model=" ")
    assert settings.llm_fallback_models == []
