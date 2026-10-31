from __future__ import annotations

import contextlib
from pathlib import Path

from platformdirs import user_data_dir
from pydantic import Field

from querity.ingestion.settings import IngestionSettings, database_fingerprint


class AgentSettings(IngestionSettings):
    max_revisions: int = Field(default=2, ge=0, le=10, validation_alias="AGENT_MAX_REVISIONS")
    max_human_revisions: int = Field(
        default=10,
        ge=1,
        le=50,
        validation_alias="AGENT_MAX_HUMAN_REVISIONS",
    )
    max_tables: int = Field(default=10, ge=1, le=50, validation_alias="AGENT_MAX_TABLES")
    max_context_chars: int = Field(
        default=24_000,
        ge=4_000,
        le=120_000,
        validation_alias="AGENT_MAX_CONTEXT_CHARS",
    )
    max_result_rows: int = Field(
        default=500,
        ge=1,
        le=10_000,
        validation_alias="AGENT_MAX_RESULT_ROWS",
    )
    max_preview_rows: int = Field(
        default=50,
        ge=1,
        le=500,
        validation_alias="AGENT_MAX_PREVIEW_ROWS",
    )
    agent_statement_timeout_ms: int = Field(
        default=15_000,
        ge=1_000,
        le=300_000,
        validation_alias="AGENT_STATEMENT_TIMEOUT_MS",
    )
    review_model: str | None = Field(default=None, validation_alias="AGENT_REVIEW_MODEL")
    data_dir: Path | None = Field(default=None, validation_alias="AGENT_DATA_DIR")
    history_db_path: Path | None = Field(default=None, validation_alias="AGENT_HISTORY_DB")
    checkpoint_db_path: Path | None = Field(default=None, validation_alias="AGENT_CHECKPOINT_DB")
    cache_enabled: bool = Field(default=True, validation_alias="AGENT_CACHE_ENABLED")
    cache_db_path: Path | None = Field(default=None, validation_alias="AGENT_CACHE_DB")
    cache_max_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1_048_576,
        validation_alias="AGENT_CACHE_MAX_BYTES",
    )
    enable_clarification: bool = Field(default=True, validation_alias="AGENT_ENABLE_CLARIFICATION")
    enable_explain_guard: bool = Field(default=True, validation_alias="AGENT_ENABLE_EXPLAIN_GUARD")

    @property
    def effective_review_model(self) -> str:
        return self.review_model or self.llm_model

    @property
    def resolved_data_dir(self) -> Path:
        if self.data_dir is not None:
            return self.data_dir.expanduser()
        return Path(user_data_dir("querity", appauthor=False))

    @property
    def resolved_history_db_path(self) -> Path:
        if self.history_db_path is not None:
            return self.history_db_path.expanduser()
        return self.resolved_data_dir / "history.sqlite3"

    @property
    def resolved_checkpoint_db_path(self) -> Path:
        if self.checkpoint_db_path is not None:
            return self.checkpoint_db_path.expanduser()
        return self.resolved_data_dir / "checkpoints.sqlite3"

    @property
    def resolved_cache_db_path(self) -> Path:
        if self.cache_db_path is not None:
            return self.cache_db_path.expanduser()
        return self.resolved_data_dir / "cache.sqlite3"

    @property
    def db_fingerprint(self) -> str:
        return database_fingerprint(self.source_url)

    def ensure_local_dirs(self) -> None:
        directories = {
            self.resolved_data_dir,
            self.resolved_history_db_path.parent,
            self.resolved_checkpoint_db_path.parent,
            self.resolved_cache_db_path.parent,
        }
        for directory in directories:
            created = not directory.exists()
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            managed_default = self.data_dir is None and directory == self.resolved_data_dir
            if created or managed_default:
                with contextlib.suppress(OSError):
                    directory.chmod(0o700)
