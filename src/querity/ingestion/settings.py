from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


def normalize_database_url(url: str) -> str:
    """Force SQLAlchemy to use psycopg 3 instead of silently requiring psycopg2. When we append +psycopg it uses psycopg3, wihtout it it deafutls to psycopg2"""
    url = url.strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgresql+psycopg://"):
        return url
    raise ValueError(
        "Database URL must start with postgresql://, postgres:// or postgresql+psycopg://"
    )


def _split_csv(value: str) -> list[str]:
    """splits the string on , then strip each element and save to a dict {element1:None,element2:None}, and key is unique so each unqiue element is extracted and sorted in the original order then transformed to list"""
    return list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def database_fingerprint(url: str) -> str:
    parsed = make_url(normalize_database_url(url))
    identity = (
        f"{parsed.drivername}://{parsed.username or ''}@"
        f"{parsed.host or 'local'}:{parsed.port or ''}/{parsed.database or ''}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_by_alias=True,
        validate_by_name=True,
    )

    # Connections -----------------------------------------------------------------
    source_database_url: SecretStr | None = Field(
        default=None,
        validation_alias="SOURCE_DATABASE_URL",
    )
    metadata_database_url: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_DATABASE_URL", "METADATA_DATABASE_URL"),
    )
    source_id: str = Field(default="default", validation_alias="SOURCE_ID")
    source_schemas: str = Field(default="public", validation_alias="SOURCE_SCHEMAS")
    include_views: bool = Field(default=False, validation_alias="INCLUDE_VIEWS")
    table_include: str = Field(default="", validation_alias="TABLE_INCLUDE")
    table_exclude: str = Field(default="", validation_alias="TABLE_EXCLUDE")

    # Profiling -------------------------------------------------------------------
    statement_timeout_ms: int = Field(
        default=15_000, ge=1_000, le=300_000, validation_alias="STATEMENT_TIMEOUT_MS"
    )
    profile_sample_rows: int = Field(default=3, ge=1, le=10, validation_alias="PROFILE_SAMPLE_ROWS")
    low_cardinality_threshold: int = Field(
        default=50, ge=1, le=200, validation_alias="LOW_CARDINALITY_THRESHOLD"
    )
    high_cardinality_sample_size: int = Field(
        default=3, ge=1, le=10, validation_alias="HIGH_CARDINALITY_SAMPLE_SIZE"
    )
    max_sample_value_chars: int = Field(
        default=120, ge=20, le=1_000, validation_alias="MAX_SAMPLE_VALUE_CHARS"
    )
    max_row_sample_columns: int = Field(
        default=15, ge=1, le=50, validation_alias="MAX_ROW_SAMPLE_COLUMNS"
    )
    max_prompt_chars: int = Field(
        default=24_000, ge=2_000, le=100_000, validation_alias="MAX_PROMPT_CHARS"
    )
    pii_masking: bool = Field(default=True, validation_alias="PII_MASKING")
    sensitive_column_patterns: str = Field(default="", validation_alias="SENSITIVE_COLUMN_PATTERNS")
    profile_numeric_ranges: bool = Field(default=False, validation_alias="PROFILE_NUMERIC_RANGES")
    refresh_unchanged_tables: bool = Field(
        default=False, validation_alias="REFRESH_UNCHANGED_TABLES"
    )

    # Metadata / vector storage ---------------------------------------------------
    metadata_schema: str = Field(default="querity", validation_alias="METADATA_SCHEMA")
    embedding_index: Literal["hnsw", "none"] = Field(
        default="none", validation_alias="EMBEDDING_INDEX"
    )

    # Enrichment LLM --------------------------------------------------------------
    enable_llm: bool = Field(default=True, validation_alias="ENABLE_LLM")
    llm_provider: Literal["google", "nvidia"] = Field(
        default="google", validation_alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="gemini-3.5-flash-lite", validation_alias="LLM_MODEL")
    llm_secondary_model: str | None = Field(
        default="gemini-3.1-flash-lite", validation_alias="LLM_SECONDARY_MODEL"
    )
    llm_tertiary_model: str | None = Field(
        default="gemini-2.5-flash-lite", validation_alias="LLM_TERTIARY_MODEL"
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0, validation_alias="LLM_TEMPERATURE")
    llm_max_output_tokens: int = Field(
        default=4_096, ge=256, le=32_768, validation_alias="LLM_MAX_OUTPUT_TOKENS"
    )
    llm_max_retries: int = Field(default=3, ge=1, le=10, validation_alias="LLM_MAX_RETRIES")
    llm_concurrency: int = Field(default=2, ge=1, le=16, validation_alias="LLM_CONCURRENCY")
    google_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    nvidia_api_key: SecretStr | None = Field(default=None, validation_alias="NVIDIA_API_KEY")

    # Embeddings ------------------------------------------------------------------
    embedding_provider: Literal["nvidia"] = Field(
        default="nvidia", validation_alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(
        default="nvidia/nemotron-3-embed-1b", validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(
        default=2_048, ge=64, le=16_000, validation_alias="EMBEDDING_DIMENSIONS"
    )
    embedding_batch_size: int = Field(
        default=64, ge=1, le=256, validation_alias="EMBEDDING_BATCH_SIZE"
    )
    embedding_max_chars: int = Field(
        default=12_000, ge=1_000, le=100_000, validation_alias="EMBEDDING_MAX_CHARS"
    )

    # Join graph / query logs -----------------------------------------------------
    mine_query_logs: bool = Field(default=False, validation_alias="MINE_QUERY_LOGS")
    query_log_limit: int = Field(default=500, ge=10, le=10_000, validation_alias="QUERY_LOG_LIMIT")

    # Semantic layer --------------------------------------------------------------
    semantic_layer_path: Path | None = Field(default=None, validation_alias="SEMANTIC_LAYER_PATH")
    generate_semantic_layer: bool = Field(
        default=False,
        validation_alias="GENERATE_SEMANTIC_LAYER",
        description=(
            "Let the LLM analyze the enriched catalog and generate metrics/filters for the "
            "semantic glossary. Curated entries from SEMANTIC_LAYER_PATH take precedence."
        ),
    )
    golden_queries_path: Path | None = Field(default=None, validation_alias="GOLDEN_QUERIES_PATH")
    validate_golden_sql: bool = Field(default=True, validation_alias="VALIDATE_GOLDEN_SQL")

    # Operations ------------------------------------------------------------------
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_json: bool = Field(default=False, validation_alias="LOG_JSON")

    @field_validator("google_api_key", "nvidia_api_key", mode="before")
    @classmethod
    def _blank_api_key_is_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_secondary_model", "llm_tertiary_model", mode="before")
    @classmethod
    def _blank_fallback_model_is_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("source_id")
    @classmethod
    def _validate_source_id(cls, value: str) -> str:
        if not _SOURCE_ID_RE.match(value):
            raise ValueError(
                "SOURCE_ID must be 1-63 chars and contain only letters, digits, '_', '.', '-'"
            )
        return value

    @field_validator("metadata_schema")
    @classmethod
    def _validate_metadata_schema(cls, value: str) -> str:
        if not _IDENTIFIER_RE.match(value):
            raise ValueError("METADATA_SCHEMA must be a lowercase PostgreSQL identifier")
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
        return value

    @model_validator(mode="after")
    def _validate_dimension_and_urls(self) -> IngestionSettings:
        if self.embedding_index == "hnsw" and self.embedding_dimensions > 2_000:
            raise ValueError(
                "pgvector HNSW indexes support at most 2000 dimensions. "
                "Lower EMBEDDING_DIMENSIONS or set EMBEDDING_INDEX=none."
            )
        for label, secret in (
            ("SOURCE_DATABASE_URL", self.source_database_url),
            ("SUPABASE_DATABASE_URL", self.metadata_database_url),
        ):
            if secret is not None:
                value = secret.get_secret_value().strip()
                if not value:
                    raise ValueError(f"{label} cannot be empty")
                normalize_database_url(value)
        return self

    @property
    def source_url(self) -> str:
        if self.source_database_url is None:
            raise ValueError("SOURCE_DATABASE_URL is required for indexing")
        return normalize_database_url(self.source_database_url.get_secret_value())

    @property
    def metadata_url(self) -> str:
        if self.metadata_database_url is None:
            raise ValueError("SUPABASE_DATABASE_URL is required for metadata storage")
        return normalize_database_url(self.metadata_database_url.get_secret_value())

    @property
    def schemas(self) -> list[str]:
        schemas = _split_csv(self.source_schemas)
        if not schemas:
            raise ValueError("SOURCE_SCHEMAS must contain at least one schema")
        for schema in schemas:
            if not _IDENTIFIER_RE.match(schema):
                raise ValueError(f"Invalid schema name in SOURCE_SCHEMAS: {schema!r}")
        return schemas

    @property
    def llm_fallback_models(self) -> list[str]:
        candidates = [self.llm_secondary_model, self.llm_tertiary_model]
        models = [value.strip() for value in candidates if value and value.strip()]
        return [model for model in dict.fromkeys(models) if model != self.llm_model]

    @property
    def include_patterns(self) -> list[str]:
        return _split_csv(self.table_include)

    @property
    def exclude_patterns(self) -> list[str]:
        return _split_csv(self.table_exclude)

    @property
    def extra_sensitive_patterns(self) -> list[str]:
        return _split_csv(self.sensitive_column_patterns)

    @property
    def effective_semantic_layer_path(self) -> Path | None:
        return self.semantic_layer_path

    @property
    def effective_golden_queries_path(self) -> Path | None:
        return self.golden_queries_path
