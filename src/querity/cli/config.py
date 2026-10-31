from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir
from sqlalchemy.engine import make_url

from querity.agent.settings import AgentSettings
from querity.ingestion.settings import normalize_database_url

CONFIG_VERSION = 1


class ConfigError(RuntimeError):
    pass


class ConfigNotFound(ConfigError):
    pass


def config_path() -> Path:
    override = os.environ.get("QUERITY_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path(user_config_dir("querity", appauthor=False)) / "config.json"


def source_id_from_url(value: str) -> str:
    try:
        database = make_url(normalize_database_url(value)).database or "default"
    except Exception:
        database = "default"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", database).strip("_.-").lower()
    if not cleaned:
        cleaned = "default"
    if cleaned[0].isdigit():
        cleaned = f"db_{cleaned}"
    return cleaned[:63]


def database_label(value: str) -> str:
    if not value:
        return "(no database selected)"
    try:
        parsed = make_url(normalize_database_url(value))
    except Exception:
        return "(invalid database url)"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.host or 'local'}{port}/{parsed.database or ''}"


@dataclass
class CliConfig:
    database_url: str = ""
    source_id: str = "default"
    databases: list[dict[str, Any]] = field(default_factory=list)
    source_schemas: str = "public"
    metadata_schema: str = "querity"
    llm_provider: str = "google"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_secondary_model: str = "gemini-3.1-flash-lite"
    llm_tertiary_model: str = "gemini-2.5-flash-lite"
    review_model: str = ""
    google_api_key: str = ""
    nvidia_api_key: str = ""
    embedding_provider: str = "nvidia"
    embedding_model: str = "nvidia/nemotron-3-embed-1b"
    embedding_dimensions: int = 2048
    generate_semantic_layer: bool = True
    golden_queries_path: str = ""
    statement_timeout_ms: int = 15_000
    agent_statement_timeout_ms: int = 15_000
    max_result_rows: int = 500
    max_revisions: int = 2
    version: int = CONFIG_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_settings(
        self,
        *,
        database_url: str | None = None,
        source_id: str | None = None,
        source_schemas: str | None = None,
        metadata_schema: str | None = None,
        require_database: bool = True,
    ) -> AgentSettings:
        url = database_url if database_url is not None else self.database_url
        url = normalize_database_url(url) if url else ""
        if not url and require_database:
            raise ConfigError(
                "No database selected. Open the chat UI and use /newchat to choose a database."
            )
        current_url = ""
        if self.database_url:
            with contextlib.suppress(Exception):
                current_url = normalize_database_url(self.database_url)
        matched_entry: dict[str, str] = {}
        if url:
            for item in self.databases:
                if isinstance(item, dict):
                    with contextlib.suppress(Exception):
                        if normalize_database_url(str(item.get("url") or "")) == url:
                            matched_entry = item
                            break

        if source_id:
            resolved_source_id = source_id
        elif matched_entry.get("source_id"):
            resolved_source_id = str(matched_entry["source_id"])
        elif not database_url or not self.database_url or url == current_url:
            resolved_source_id = self.source_id
        else:
            resolved_source_id = source_id_from_url(url)
        if not resolved_source_id:
            resolved_source_id = source_id_from_url(url) if url else "default"

        resolved_source_schemas = (
            source_schemas
            if source_schemas is not None
            else (matched_entry.get("source_schemas") or self.source_schemas or "public")
        )
        resolved_metadata_schema = (
            metadata_schema
            if metadata_schema is not None
            else (matched_entry.get("metadata_schema") or self.metadata_schema or "querity")
        )

        values: dict[str, Any] = {
            "source_id": resolved_source_id,
            "source_schemas": resolved_source_schemas,
            "metadata_schema": resolved_metadata_schema,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_secondary_model": self.llm_secondary_model.strip() or None,
            "llm_tertiary_model": self.llm_tertiary_model.strip() or None,
            "google_api_key": self.google_api_key or None,
            "nvidia_api_key": self.nvidia_api_key or None,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "generate_semantic_layer": self.generate_semantic_layer,
            "statement_timeout_ms": self.statement_timeout_ms,
            "agent_statement_timeout_ms": self.agent_statement_timeout_ms,
            "max_result_rows": self.max_result_rows,
            "max_revisions": self.max_revisions,
        }
        if url:
            values["source_database_url"] = url
            values["metadata_database_url"] = url
        if self.review_model.strip():
            values["review_model"] = self.review_model.strip()
        if self.embedding_dimensions > 2_000:
            values["embedding_index"] = "none"
        for k, v in self.extra.items():
            if k not in values:
                values[k] = v
        return AgentSettings(_env_file=None, **values)

    def to_ingestion_settings(
        self,
        *,
        database_url: str | None = None,
        source_id: str | None = None,
        source_schemas: str | None = None,
        metadata_schema: str | None = None,
        require_database: bool = True,
    ) -> AgentSettings:
        return self.to_settings(
            database_url=database_url,
            source_id=source_id,
            source_schemas=source_schemas,
            metadata_schema=metadata_schema,
            require_database=require_database,
        )

    def masked_url(self) -> str:
        if not self.database_url:
            return "(no database selected)"
        try:
            parsed = make_url(self.database_url)
        except Exception:
            return "(invalid database url)"
        port = f":{parsed.port}" if parsed.port else ""
        return (
            f"{parsed.drivername}://{parsed.username or 'unknown'}@"
            f"{parsed.host or 'local'}{port}/{parsed.database or ''}"
        )

    def remember_database(
        self,
        url: str,
        *,
        source_id: str | None = None,
        source_schemas: str | None = None,
        metadata_schema: str | None = None,
        golden_queries_path: str | None = None,
        ingested: bool | None = None,
        table_count: int | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_database_url(url)
        resolved_source_id = source_id or source_id_from_url(normalized)
        existing_golden = ""
        existing_source_schemas = ""
        existing_metadata_schema = ""
        existing_ingested = False
        existing_table_count = 0
        for item in self.databases:
            if isinstance(item, dict) and item.get("url") == normalized:
                existing_golden = str(item.get("golden_queries_path") or "")
                existing_source_schemas = str(item.get("source_schemas") or "")
                existing_metadata_schema = str(item.get("metadata_schema") or "")
                existing_ingested = bool(item.get("ingested"))
                existing_table_count = int(item.get("table_count") or 0)
                break
        resolved_golden = (
            golden_queries_path
            if golden_queries_path is not None
            else (existing_golden or self.golden_queries_path)
        )
        resolved_source_schemas = (
            source_schemas
            if source_schemas is not None
            else (existing_source_schemas or self.source_schemas or "public")
        )
        resolved_metadata_schema = (
            metadata_schema
            if metadata_schema is not None
            else (existing_metadata_schema or self.metadata_schema or "querity")
        )
        resolved_ingested = ingested if ingested is not None else existing_ingested
        resolved_table_count = table_count if table_count is not None else existing_table_count
        entry: dict[str, Any] = {
            "url": normalized,
            "source_id": resolved_source_id,
            "label": database_label(normalized),
            "source_schemas": resolved_source_schemas,
            "metadata_schema": resolved_metadata_schema,
            "golden_queries_path": resolved_golden,
            "ingested": resolved_ingested,
            "table_count": resolved_table_count,
        }
        remaining = []
        for item in self.databases:
            existing_url = str(item.get("url") or "") if isinstance(item, dict) else ""
            normalized_existing = ""
            if existing_url:
                try:
                    normalized_existing = normalize_database_url(existing_url)
                except Exception:
                    normalized_existing = ""
            if normalized_existing and normalized_existing == normalized:
                continue
            remaining.append(item)
        self.databases = [*remaining, entry]
        self.database_url = normalized
        self.source_id = resolved_source_id
        self.source_schemas = resolved_source_schemas
        self.metadata_schema = resolved_metadata_schema
        if resolved_golden:
            self.golden_queries_path = resolved_golden
        return entry

    def mark_database_ingested(
        self,
        url: str,
        *,
        ingested: bool = True,
        table_count: int = 0,
    ) -> None:
        normalized = normalize_database_url(url)
        for item in self.databases:
            if isinstance(item, dict) and item.get("url") == normalized:
                item["ingested"] = ingested
                if table_count > 0:
                    item["table_count"] = table_count
                break

    def is_database_ingested(self, url: str) -> bool:
        normalized = normalize_database_url(url)
        for item in self.databases:
            if isinstance(item, dict) and item.get("url") == normalized:
                return bool(item.get("ingested"))
        return False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["database_url"] = self.database_url
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CliConfig:
        known = {field_name for field_name in cls.__dataclass_fields__}
        values = {key: value for key, value in payload.items() if key in known}
        extra = {key: value for key, value in payload.items() if key not in known}
        if not isinstance(values.get("databases"), list):
            values["databases"] = []
        config = cls(**values)
        if extra:
            config.extra.update(extra)
        if config.database_url and not config.databases:
            with contextlib.suppress(Exception):
                normalized = normalize_database_url(config.database_url)
                resolved_source_id = config.source_id or source_id_from_url(normalized)
                config.databases = [
                    {
                        "url": normalized,
                        "source_id": resolved_source_id,
                        "label": database_label(normalized),
                        "source_schemas": config.source_schemas or "public",
                        "metadata_schema": config.metadata_schema or "querity",
                        "golden_queries_path": config.golden_queries_path or "",
                    }
                ]
        return config

    @classmethod
    def load(cls, path: Path | None = None) -> CliConfig:
        target = path or config_path()
        if not target.exists():
            raise ConfigNotFound(f"No configuration found at {target}. Run `querity init` first.")
        with contextlib.suppress(OSError):
            target.chmod(0o600)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Could not read configuration at {target}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigError(f"Configuration at {target} is not a JSON object")
        return cls.from_dict(payload)

    def save(self, path: Path | None = None) -> Path:
        target = path or config_path()
        parent_created = not target.parent.exists()
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if parent_created:
            with contextlib.suppress(OSError):
                target.parent.chmod(0o700)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        handle, temp_name = tempfile.mkstemp(
            prefix=".config-",
            suffix=".json",
            dir=str(target.parent),
            text=True,
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.write("\n")
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        with contextlib.suppress(OSError):
            target.chmod(0o600)
        return target


def load_config() -> CliConfig:
    return CliConfig.load()
