from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from querity.ingestion.settings import IngestionSettings


def _connect_args(*, application_name: str, statement_timeout_ms: int) -> dict[str, object]:
    return {
        "application_name": application_name,
        "connect_timeout": 10,
        # PgBouncer transaction pooling does not support named prepared statements.
        "prepare_threshold": None,
        "options": (
            f"-c statement_timeout={statement_timeout_ms} "
            "-c idle_in_transaction_session_timeout=60000 "
            "-c lock_timeout=5000"
        ),
    }


def create_source_engine(settings: IngestionSettings) -> Engine:
    connect_args = _connect_args(
        application_name="querity-ingest-readonly",
        statement_timeout_ms=settings.statement_timeout_ms,
    )
    connect_args["options"] = f"{connect_args['options']} -c default_transaction_read_only=on"
    return sa.create_engine(
        settings.source_url,
        poolclass=sa.pool.NullPool,
        connect_args=connect_args,
        future=True,
    )


def create_agent_engine(settings: IngestionSettings) -> Engine:
    timeout_ms = int(getattr(settings, "agent_statement_timeout_ms", 15_000))
    connect_args = _connect_args(
        application_name="querity-agent",
        statement_timeout_ms=timeout_ms,
    )
    connect_args["options"] = f"{connect_args['options']} -c default_transaction_read_only=on"
    return sa.create_engine(
        settings.source_url,
        poolclass=sa.pool.NullPool,
        connect_args=connect_args,
        future=True,
    )


def create_metadata_engine(settings: IngestionSettings) -> Engine:
    return sa.create_engine(
        settings.metadata_url,
        pool_size=2,
        max_overflow=2,
        pool_pre_ping=True,
        pool_recycle=1_800,
        connect_args=_connect_args(
            application_name="querity-ingest-metadata",
            statement_timeout_ms=max(settings.statement_timeout_ms * 4, 60_000),
        ),
        future=True,
    )


@contextmanager
def readonly_connection(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as connection:
        connection.execute(sa.text("SET TRANSACTION READ ONLY"))
        try:
            yield connection
        finally:
            connection.rollback()


def validate_identifier(identifier: str) -> str:
    if not identifier or "\x00" in identifier:
        raise ValueError("SQL identifier must be non-empty and cannot contain NUL bytes")
    return identifier


def quote_ident(engine: Engine | Connection, identifier: str) -> str:
    validate_identifier(identifier)
    dialect = engine.dialect
    return dialect.identifier_preparer.quote(identifier)


def qualified_name(engine: Engine | Connection, schema: str, table: str) -> str:
    return f"{quote_ident(engine, schema)}.{quote_ident(engine, table)}"


def sql_literal_quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
