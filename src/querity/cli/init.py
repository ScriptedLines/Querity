from __future__ import annotations

import contextlib
from pathlib import Path
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from querity.agent.settings import AgentSettings
from querity.cli.config import CliConfig, ConfigError, config_path
from querity.ingestion.db import create_source_engine, quote_ident
from querity.ingestion.settings import normalize_database_url

_GOOGLE_LLM = "gemini-3.5-flash-lite"
_GOOGLE_SECONDARY_LLM = "gemini-3.1-flash-lite"
_GOOGLE_TERTIARY_LLM = "gemini-2.5-flash-lite"
_NVIDIA_LLM = "meta/llama-3.3-70b-instruct"
_NVIDIA_EMBEDDING = "nvidia/nemotron-3-embed-1b"


@dataclass
class DatabaseProbe:
    database: str
    user: str
    server_version: str
    schemas: list[str] = field(default_factory=list)
    vector_installed: bool = False
    can_create_schema: bool = False
    metadata_table_catalog: bool = False
    existing_embedding_dimensions: int | None = None
    existing_tables: int = 0
    stored_source_ids: list[str] = field(default_factory=list)
    is_transaction_pooler: bool = False


def _source_id_from_database(database: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", database.strip()).strip("_.-").lower()
    if not value:
        value = "default"
    if value[0].isdigit():
        value = f"db_{value}"
    return value[:63]


def _probe_database(database_url: str, metadata_schema: str) -> DatabaseProbe:
    url = normalize_database_url(database_url)
    settings = AgentSettings(
        _env_file=None,
        source_database_url=url,
        metadata_database_url=url,
        source_id="probe",
        source_schemas="public",
        metadata_schema=metadata_schema,
    )
    engine = create_source_engine(settings)
    try:
        with engine.connect() as connection:
            database = str(connection.execute(sa.text("SELECT current_database()")).scalar_one())
            user = str(connection.execute(sa.text("SELECT current_user")).scalar_one())
            version = str(connection.execute(sa.text("SELECT version()")).scalar_one())
            schemas = [
                str(row[0])
                for row in connection.execute(
                    sa.text(
                        "SELECT schema_name FROM information_schema.schemata "
                        "WHERE schema_name NOT LIKE 'pg_%' "
                        "AND schema_name <> 'information_schema' "
                        "ORDER BY schema_name"
                    )
                )
            ]
            vector_installed = False
            with contextlib.suppress(Exception):
                vector_installed = (
                    connection.execute(
                        sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                    ).first()
                    is not None
                )
            can_create = False
            with contextlib.suppress(Exception):
                can_create = bool(
                    connection.execute(
                        sa.text(
                            "SELECT has_database_privilege("
                            "current_user, current_database(), 'CREATE')"
                        )
                    ).scalar_one()
                )
            metadata_table_catalog = False
            existing_dimensions: int | None = None
            existing_tables = 0
            stored_source_ids: list[str] = []
            with contextlib.suppress(Exception):
                metadata_table_catalog = (
                    connection.execute(
                        sa.text(
                            "SELECT 1 FROM information_schema.tables "
                            "WHERE table_schema = :schema AND table_name = 'table_catalog'"
                        ),
                        {"schema": metadata_schema},
                    ).first()
                    is not None
                )
            if metadata_table_catalog:
                qualified = f"{quote_ident(connection, metadata_schema)}.table_catalog"
                try:
                    dimensions = connection.execute(
                        sa.text(
                            f"SELECT vector_dims(embedding) FROM {qualified} "
                            "WHERE embedding IS NOT NULL LIMIT 1"
                        )
                    ).scalar_one_or_none()
                    existing_dimensions = int(dimensions) if dimensions is not None else None
                except Exception:
                    existing_dimensions = None
                if existing_dimensions is None:
                    try:
                        type_text = connection.execute(
                            sa.text(
                                "SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) "
                                "FROM pg_catalog.pg_attribute a "
                                "JOIN pg_catalog.pg_class c ON c.oid = a.attrelid "
                                "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                                "WHERE n.nspname = :schema AND c.relname = 'table_catalog' "
                                "AND a.attname = 'embedding' AND a.attnum > 0"
                            ),
                            {"schema": metadata_schema},
                        ).scalar_one_or_none()
                        match = re.search(r"\((\d+)\)", str(type_text or ""))
                        if match:
                            existing_dimensions = int(match.group(1))
                    except Exception:
                        existing_dimensions = None
                try:
                    existing_tables = int(
                        connection.execute(
                            sa.text(f"SELECT count(*) FROM {qualified} WHERE is_active = true")
                        ).scalar_one()
                    )
                except Exception:
                    existing_tables = 0
                try:
                    stored_source_ids = [
                        str(row[0])
                        for row in connection.execute(
                            sa.text(
                                f"SELECT DISTINCT source_id FROM {qualified} "
                                "WHERE is_active = true ORDER BY source_id"
                            )
                        )
                    ]
                except Exception:
                    stored_source_ids = []
    finally:
        engine.dispose()
    parsed = make_url_quiet(url)
    return DatabaseProbe(
        database=database,
        user=user,
        server_version=version.splitlines()[0] if version else "PostgreSQL",
        schemas=schemas,
        vector_installed=vector_installed,
        can_create_schema=can_create,
        metadata_table_catalog=metadata_table_catalog,
        existing_embedding_dimensions=existing_dimensions,
        existing_tables=existing_tables,
        stored_source_ids=stored_source_ids,
        is_transaction_pooler=bool(parsed and parsed.port == 6543),
    )


def make_url_quiet(url: str) -> Any:
    from sqlalchemy.engine import make_url

    try:
        return make_url(url)
    except Exception:
        return None


def _ask_secret(label: str, existing: str) -> str:
    suffix = " (press Enter to keep the existing key)" if existing else ""
    value = Prompt.ask(f"{label}{suffix}", password=True, default="")
    value = (value or "").strip()
    return value or existing


def _ask_required_secret(label: str, existing: str) -> str:
    while True:
        value = _ask_secret(label, existing)
        if value:
            return value
        Console().print("[red]A key is required for this provider.[/red]")


def _default_source_schemas(
    probe: DatabaseProbe,
    existing: CliConfig | None,
    metadata_schema: str,
) -> str:
    if existing:
        return existing.source_schemas
    if "public" in probe.schemas:
        return "public"
    candidates = [schema for schema in probe.schemas if schema not in {"public", metadata_schema}]
    return candidates[0] if candidates else "public"


def _test_credentials(config: CliConfig, console: Console) -> None:
    settings = config.to_settings()
    try:
        from querity.ingestion.embeddings import EmbeddingClient

        with console.status("Testing embedding credentials..."):
            EmbeddingClient(settings).embed_query("connection test")
        console.print("[green]Embedding credentials work.[/green]")
    except Exception as exc:
        console.print(f"[yellow]Embedding test failed:[/yellow] {exc}")
    try:
        from querity.ingestion.llm import build_chat_model

        with console.status("Testing LLM credentials..."):
            model = build_chat_model(settings)
            model.invoke("Reply with the single word OK.")
        console.print("[green]LLM credentials work.[/green]")
    except Exception as exc:
        console.print(f"[yellow]LLM test failed:[/yellow] {exc}")


def _configure(
    existing: CliConfig | None,
    console: Console,
) -> CliConfig:
    database_url = ""
    probe: DatabaseProbe | None = None
    for attempt in range(1, 4):
        default_hint = f" [{existing.masked_url()}]" if existing and attempt == 1 else ""
        entered = Prompt.ask(
            f"PostgreSQL URL (source + metadata){default_hint}",
            password=True,
            default="",
        )
        candidate = (entered or "").strip()
        if not candidate and existing:
            candidate = existing.database_url
        if not candidate:
            console.print("[red]A database URL is required.[/red]")
            continue
        try:
            database_url = normalize_database_url(candidate)
            metadata_schema = existing.metadata_schema if existing else "querity"
            with console.status("Connecting to PostgreSQL..."):
                probe = _probe_database(database_url, metadata_schema)
            break
        except Exception as exc:
            console.print(f"[red]Connection failed:[/red] {exc}")
            if attempt == 3:
                raise ConfigError("Could not connect to the database.") from exc
    assert probe is not None

    console.print(
        Panel(
            Text(
                f"Database : {probe.database}\n"
                f"User     : {probe.user}\n"
                f"Server   : {probe.server_version}\n"
                f"Schemas  : {', '.join(probe.schemas[:20]) or '(none)'}\n"
                f"pgvector : {'installed' if probe.vector_installed else 'not installed'}\n"
                f"DDL      : {'allowed' if probe.can_create_schema else 'not allowed'}",
                overflow="fold",
            ),
            title="Connection",
            border_style="green",
        )
    )
    if probe.is_transaction_pooler:
        console.print(
            "[yellow]This looks like a transaction pooler (port 6543). Ingestion runs DDL "
            "and may need the session pooler on port 5432.[/yellow]"
        )
    if probe.metadata_table_catalog:
        console.print(
            f"[cyan]Existing metadata catalog found with {probe.existing_tables} "
            "indexed tables.[/cyan]"
        )
    elif not probe.can_create_schema and metadata_schema not in probe.schemas:
        console.print(
            "[red]This role cannot create the metadata schema in this database. "
            "Grant CREATE on the database (and permission to enable pgvector) or "
            "pre-create the metadata schema and tables before ingestion.[/red]"
        )

    source_id_default = (
        existing.source_id
        if existing
        else (
            probe.stored_source_ids[0]
            if probe.stored_source_ids
            else _source_id_from_database(probe.database)
        )
    )
    if probe.stored_source_ids:
        console.print(f"[cyan]Catalog source ids: {', '.join(probe.stored_source_ids[:10])}[/cyan]")
    source_id = Prompt.ask("Source id", default=source_id_default).strip()
    metadata_schema = Prompt.ask(
        "Metadata schema",
        default=existing.metadata_schema if existing else "querity",
    ).strip()
    source_schemas = Prompt.ask(
        "Source schemas (comma separated)",
        default=_default_source_schemas(probe, existing, metadata_schema),
    ).strip()

    provider_default = existing.llm_provider if existing else "google"
    llm_provider = Prompt.ask(
        "LLM provider",
        choices=["google", "nvidia"],
        default=provider_default,
    )
    llm_default = (
        existing.llm_model
        if existing and existing.llm_provider == llm_provider
        else (_GOOGLE_LLM if llm_provider == "google" else _NVIDIA_LLM)
    )
    llm_model = Prompt.ask("LLM model", default=llm_default).strip()
    same_provider = existing is not None and existing.llm_provider == llm_provider
    secondary_default = (
        existing.llm_secondary_model
        if same_provider
        else (_GOOGLE_SECONDARY_LLM if llm_provider == "google" else "")
    )
    secondary_model = Prompt.ask(
        "Secondary model (Enter = no fallback)",
        default=secondary_default,
    ).strip()
    tertiary_default = (
        existing.llm_tertiary_model
        if same_provider
        else (_GOOGLE_TERTIARY_LLM if llm_provider == "google" else "")
    )
    tertiary_model = Prompt.ask(
        "Tertiary model (Enter = no fallback)",
        default=tertiary_default,
    ).strip()
    review_model = ""

    google_key = existing.google_api_key if existing else ""
    nvidia_key = existing.nvidia_api_key if existing else ""
    if llm_provider == "google":
        google_key = _ask_required_secret("Google API key (LLM)", google_key)
    else:
        nvidia_key = _ask_required_secret("NVIDIA API key (LLM)", nvidia_key)

    embedding_provider = "nvidia"
    embedding_model = _NVIDIA_EMBEDDING
    embedding_dimensions = 2_048
    if not nvidia_key:
        nvidia_key = _ask_required_secret("NVIDIA API key (embeddings)", nvidia_key)

    generate_semantics = True

    config = CliConfig(
        database_url=database_url,
        source_id=source_id,
        databases=existing.databases if existing else [],
        source_schemas=source_schemas,
        metadata_schema=metadata_schema,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_secondary_model=secondary_model,
        llm_tertiary_model=tertiary_model,
        review_model=review_model,
        google_api_key=google_key,
        nvidia_api_key=nvidia_key,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        generate_semantic_layer=generate_semantics,
        golden_queries_path=existing.golden_queries_path if existing else "",
        statement_timeout_ms=existing.statement_timeout_ms if existing else 15_000,
        agent_statement_timeout_ms=(existing.agent_statement_timeout_ms if existing else 15_000),
        max_result_rows=existing.max_result_rows if existing else 500,
        max_revisions=existing.max_revisions if existing else 2,
    )
    try:
        config.remember_database(
            database_url,
            source_id=source_id,
            source_schemas=source_schemas,
            metadata_schema=metadata_schema,
        )
        settings = config.to_settings()
        _ = settings.schemas
    except Exception as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
    return config


def run_init(
    *,
    force: bool = False,
    console: Console | None = None,
) -> CliConfig:
    console = console or Console()
    if not sys.stdin.isatty():
        raise ConfigError("`querity init` must be run in an interactive terminal.")
    path = config_path()
    existing: CliConfig | None = None
    if path.exists():
        try:
            existing = CliConfig.load(path)
        except ConfigError:
            existing = None
        if existing and not force:
            console.print(f"Configuration found at [bold]{path}[/bold]")
            if not Confirm.ask("Reconfigure this database?", default=False):
                return existing

    console.print(
        Panel(
            Text(
                "This wizard stores the database URL and API keys locally at\n"
                f"{path}\n"
                "The file is created with mode 0600 and project .env files are ignored.",
                overflow="fold",
            ),
            title="Querity setup",
            border_style="cyan",
        )
    )
    config = _configure(existing, console)
    saved = config.save(path)
    console.print(f"[green]Saved configuration to {saved}[/green]")

    if Confirm.ask("Test LLM and embedding credentials now?", default=True):
        _test_credentials(config, console)

    if Confirm.ask("Run ingestion now?", default=True):
        from querity.cli.ingest import run_ingestion

        run_ingestion(config.to_settings(), console=console)

    from querity.cli.config import load_config

    console.print(
        Panel(
            Text(
                f"Database: {load_config().masked_url()}\n"
                "Start chatting: querity chat\n"
                'One-off question: querity ask "top customers by revenue"\n'
                "Refresh catalog: querity ingest",
                overflow="fold",
            ),
            title="Ready",
            border_style="green",
        )
    )
    return config
