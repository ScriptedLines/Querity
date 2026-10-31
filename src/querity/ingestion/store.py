from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from querity.common.logging import get_logger
from querity.ingestion.db import create_metadata_engine, quote_ident
from querity.ingestion.models import (
    ExistingGoldenState,
    ExistingSemanticState,
    ExistingTableState,
    TableCatalogRecord,
)
from querity.ingestion.semantic_layer import GoldenEntry, SemanticEntry
from querity.ingestion.settings import IngestionSettings

logger = get_logger(__name__)

_VECTOR_DIMENSION_RE = re.compile(r"\((\d+)\)")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def vector_literal(embedding: Sequence[float], dimensions: int) -> str:
    if len(embedding) != dimensions:
        raise ValueError(f"Embedding has {len(embedding)} dimensions; expected {dimensions}")
    values: list[str] = []
    for item in embedding:
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("Embedding contains a non-finite number")
        values.append(format(number, ".10g"))
    return "[" + ",".join(values) + "]"


class MetadataStore:
    def __init__(self, settings: IngestionSettings, engine: Engine | None = None) -> None:
        self._settings = settings
        self._schema = settings.metadata_schema
        self._engine = engine or create_metadata_engine(settings)
        self._vector_type: str | None = None

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def schema(self) -> str:
        return self._schema

    def vector_cast(self) -> str:
        return self._vector_cast()

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> MetadataStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _qualified(self, connection: Connection, table: str) -> str:
        return f"{quote_ident(connection, self._schema)}.{quote_ident(connection, table)}"

    def ensure_schema(self) -> None:
        with self._engine.begin() as connection:
            try:
                connection.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            except sa.exc.SQLAlchemyError as exc:
                raise RuntimeError(
                    "Could not enable the pgvector extension. Run this migration as a role "
                    "that can create extensions (or enable 'vector' in the Supabase dashboard)."
                ) from exc

            extension_schema = connection.execute(
                sa.text(
                    "SELECT n.nspname FROM pg_extension e "
                    "JOIN pg_namespace n ON n.oid = e.extnamespace "
                    "WHERE e.extname = 'vector'"
                )
            ).scalar_one_or_none()
            if not extension_schema:
                raise RuntimeError("pgvector is not installed in the target database")
            extension_schema_sql = quote_ident(connection, extension_schema)
            vector_type = f"{extension_schema_sql}.vector"
            self._vector_type = vector_type

            schema_sql = quote_ident(connection, self._schema)
            dimensions = self._settings.embedding_dimensions
            connection.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {schema_sql}"))

            connection.execute(
                sa.text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {schema_sql}.ingestion_runs (
                        run_id uuid PRIMARY KEY,
                        source_id text NOT NULL,
                        started_at timestamptz NOT NULL DEFAULT now(),
                        finished_at timestamptz,
                        status text NOT NULL DEFAULT 'running'
                            CHECK (status IN ('running', 'succeeded', 'partial', 'failed')),
                        stats jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                        error text
                    )
                    """
                )
            )
            connection.execute(
                sa.text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {schema_sql}.table_catalog (
                        id bigserial PRIMARY KEY,
                        source_id text NOT NULL,
                        schema_name text NOT NULL,
                        table_name text NOT NULL,
                        table_type text NOT NULL DEFAULT 'table',
                        business_description text NOT NULL,
                        m_schema text NOT NULL,
                        columns jsonb NOT NULL DEFAULT '[]'::jsonb,
                        enrichment jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                        sample_questions jsonb NOT NULL DEFAULT '[]'::jsonb,
                        join_keys jsonb NOT NULL DEFAULT '[]'::jsonb,
                        row_count_estimate bigint,
                        structural_hash text NOT NULL,
                        profile_hash text NOT NULL,
                        enrichment_hash text NOT NULL,
                        content text NOT NULL,
                        content_hash text NOT NULL,
                        embedding {vector_type}({dimensions}),
                        embedding_model text,
                        embedding_updated_at timestamptz,
                        is_active boolean NOT NULL DEFAULT true,
                        last_run_id uuid,
                        last_seen_at timestamptz NOT NULL DEFAULT now(),
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        UNIQUE (source_id, schema_name, table_name)
                    )
                    """
                )
            )
            connection.execute(
                sa.text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {schema_sql}.semantic_glossary (
                        id bigserial PRIMARY KEY,
                        source_id text NOT NULL,
                        kind text NOT NULL CHECK (kind IN ('metric', 'filter')),
                        name text NOT NULL,
                        synonyms jsonb NOT NULL DEFAULT '[]'::jsonb,
                        description text NOT NULL DEFAULT '',
                        required_tables jsonb NOT NULL DEFAULT '[]'::jsonb,
                        sql_formula text,
                        sql_condition text,
                        table_name text,
                        notes text,
                        content text NOT NULL,
                        content_hash text NOT NULL,
                        embedding {vector_type}({dimensions}),
                        embedding_model text,
                        embedding_updated_at timestamptz,
                        is_active boolean NOT NULL DEFAULT true,
                        last_run_id uuid,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        UNIQUE (source_id, kind, name)
                    )
                    """
                )
            )
            connection.execute(
                sa.text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {schema_sql}.golden_sql_examples (
                        id bigserial PRIMARY KEY,
                        source_id text NOT NULL,
                        example_key text NOT NULL,
                        question text NOT NULL,
                        sql_query text NOT NULL,
                        tables_used jsonb NOT NULL DEFAULT '[]'::jsonb,
                        tags jsonb NOT NULL DEFAULT '[]'::jsonb,
                        notes text,
                        content text NOT NULL,
                        content_hash text NOT NULL,
                        embedding {vector_type}({dimensions}),
                        embedding_model text,
                        embedding_updated_at timestamptz,
                        is_active boolean NOT NULL DEFAULT true,
                        last_run_id uuid,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        UNIQUE (source_id, example_key)
                    )
                    """
                )
            )
            connection.execute(
                sa.text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {schema_sql}.schema_graph (
                        source_id text PRIMARY KEY,
                        graph jsonb NOT NULL,
                        node_count integer NOT NULL,
                        edge_count integer NOT NULL,
                        last_run_id uuid,
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )

            self._validate_embedding_dimensions(connection, dimensions)

            connection.execute(
                sa.text(
                    "CREATE INDEX IF NOT EXISTS ingestion_runs_source_started_idx "
                    f"ON {schema_sql}.ingestion_runs (source_id, started_at DESC)"
                )
            )
            connection.execute(
                sa.text(
                    "CREATE INDEX IF NOT EXISTS table_catalog_source_active_idx "
                    f"ON {schema_sql}.table_catalog (source_id, is_active)"
                )
            )
            connection.execute(
                sa.text(
                    "CREATE INDEX IF NOT EXISTS semantic_glossary_source_active_idx "
                    f"ON {schema_sql}.semantic_glossary (source_id, is_active, kind)"
                )
            )
            connection.execute(
                sa.text(
                    "CREATE INDEX IF NOT EXISTS golden_sql_source_active_idx "
                    f"ON {schema_sql}.golden_sql_examples (source_id, is_active)"
                )
            )

            if self._settings.embedding_index == "hnsw":
                ops = f"{extension_schema_sql}.vector_cosine_ops"
                for table, index_name in (
                    ("table_catalog", "table_catalog_embedding_idx"),
                    ("semantic_glossary", "semantic_glossary_embedding_idx"),
                    ("golden_sql_examples", "golden_sql_embedding_idx"),
                ):
                    connection.execute(
                        sa.text(
                            f"CREATE INDEX IF NOT EXISTS {index_name} "
                            f"ON {schema_sql}.{table} USING hnsw (embedding {ops}) "
                            "WITH (m = 16, ef_construction = 64)"
                        )
                    )

            for table in (
                "ingestion_runs",
                "table_catalog",
                "semantic_glossary",
                "golden_sql_examples",
                "schema_graph",
            ):
                connection.execute(
                    sa.text(f"ALTER TABLE {schema_sql}.{table} ENABLE ROW LEVEL SECURITY")
                )

    def _validate_embedding_dimensions(self, connection: Connection, expected: int) -> None:
        for table in ("table_catalog", "semantic_glossary", "golden_sql_examples"):
            rendered_type = connection.execute(
                sa.text(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = :schema
                      AND c.relname = :table
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                    """
                ),
                {"schema": self._schema, "table": table},
            ).scalar_one_or_none()
            if rendered_type is None:
                continue
            match = _VECTOR_DIMENSION_RE.search(rendered_type)
            if not match or "vector" not in rendered_type.lower():
                raise RuntimeError(
                    f"{self._schema}.{table}.embedding is not a dimensioned pgvector column "
                    f"({rendered_type}). Recreate the metadata schema or migrate it."
                )
            actual = int(match.group(1))
            if actual != expected:
                raise RuntimeError(
                    f"Embedding dimension mismatch for {self._schema}.{table}: "
                    f"database has vector({actual}), configured EMBEDDING_DIMENSIONS={expected}. "
                    "Changing dimensions requires a new column/index migration."
                )

    def start_run(self, source_id: str) -> uuid.UUID:
        run_id = uuid.uuid4()
        with self._engine.begin() as connection:
            connection.execute(
                sa.text(
                    f"INSERT INTO {quote_ident(connection, self._schema)}.ingestion_runs "
                    "(run_id, source_id, status) "
                    "VALUES (CAST(:run_id AS uuid), :source_id, 'running')"
                ),
                {"run_id": str(run_id), "source_id": source_id},
            )
        return run_id

    def finish_run(
        self,
        run_id: uuid.UUID,
        status: str,
        stats: dict[str, Any],
        error: str | None = None,
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                sa.text(
                    f"UPDATE {quote_ident(connection, self._schema)}.ingestion_runs "
                    "SET finished_at = now(), status = :status, "
                    "stats = CAST(:stats AS jsonb), error = :error "
                    "WHERE run_id = CAST(:run_id AS uuid)"
                ),
                {
                    "run_id": str(run_id),
                    "status": status,
                    "stats": _json(stats),
                    "error": error,
                },
            )

    def get_table_states(self, source_id: str) -> dict[str, ExistingTableState]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                sa.text(
                    f"SELECT schema_name, table_name, business_description, m_schema, "
                    f"columns, enrichment, sample_questions, content, content_hash, "
                    f"structural_hash, profile_hash, enrichment_hash, embedding_model, "
                    f"(embedding IS NOT NULL) AS has_embedding "
                    f"FROM {quote_ident(connection, self._schema)}.table_catalog "
                    "WHERE source_id = :source_id"
                ),
                {"source_id": source_id},
            ).fetchall()

        states: dict[str, ExistingTableState] = {}
        for row in rows:
            state = ExistingTableState(
                schema_name=row.schema_name,
                table_name=row.table_name,
                business_description=row.business_description,
                m_schema=row.m_schema,
                columns=_json_field(row.columns, []),
                enrichment=_json_field(row.enrichment, {}),
                sample_questions=_json_field(row.sample_questions, []),
                content=row.content,
                content_hash=row.content_hash,
                structural_hash=row.structural_hash,
                profile_hash=row.profile_hash,
                enrichment_hash=row.enrichment_hash,
                embedding_model=row.embedding_model,
                has_embedding=bool(row.has_embedding),
            )
            states[state.key] = state
        return states

    def get_semantic_states(self, source_id: str) -> dict[str, ExistingSemanticState]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                sa.text(
                    f"SELECT kind, name, content_hash, embedding_model, "
                    f"(embedding IS NOT NULL) AS has_embedding, is_active "
                    f"FROM {quote_ident(connection, self._schema)}.semantic_glossary "
                    "WHERE source_id = :source_id"
                ),
                {"source_id": source_id},
            ).fetchall()
        return {
            f"{row.kind}:{row.name}": ExistingSemanticState(
                kind=row.kind,
                name=row.name,
                content_hash=row.content_hash,
                embedding_model=row.embedding_model,
                has_embedding=bool(row.has_embedding),
                is_active=bool(row.is_active),
            )
            for row in rows
        }

    def get_golden_states(self, source_id: str) -> dict[str, ExistingGoldenState]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                sa.text(
                    f"SELECT example_key, content_hash, embedding_model, "
                    f"(embedding IS NOT NULL) AS has_embedding, is_active "
                    f"FROM {quote_ident(connection, self._schema)}.golden_sql_examples "
                    "WHERE source_id = :source_id"
                ),
                {"source_id": source_id},
            ).fetchall()
        return {
            row.example_key: ExistingGoldenState(
                example_key=row.example_key,
                content_hash=row.content_hash,
                embedding_model=row.embedding_model,
                has_embedding=bool(row.has_embedding),
                is_active=bool(row.is_active),
            )
            for row in rows
        }

    def _vector_cast(self) -> str:
        if self._vector_type is None:
            with self._engine.connect() as connection:
                extension_schema = connection.execute(
                    sa.text(
                        "SELECT n.nspname FROM pg_extension e "
                        "JOIN pg_namespace n ON n.oid = e.extnamespace "
                        "WHERE e.extname = 'vector'"
                    )
                ).scalar_one_or_none()
                if not extension_schema:
                    raise RuntimeError(
                        "pgvector is not installed. Run init-db or enable the vector "
                        "extension in Supabase."
                    )
                self._validate_embedding_dimensions(connection, self._settings.embedding_dimensions)
                self._vector_type = f"{quote_ident(connection, extension_schema)}.vector"
        return self._vector_type

    def upsert_tables(
        self,
        source_id: str,
        run_id: uuid.UUID,
        records: Sequence[TableCatalogRecord],
    ) -> None:
        if not records:
            return
        vector_type = self._vector_cast()
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            sql = sa.text(
                f"""
                INSERT INTO {schema_sql}.table_catalog (
                    source_id, schema_name, table_name, table_type,
                    business_description, m_schema, columns, enrichment, sample_questions,
                    join_keys, row_count_estimate, structural_hash, profile_hash, enrichment_hash,
                    content, content_hash, embedding, embedding_model, embedding_updated_at,
                    is_active, last_run_id, last_seen_at, updated_at
                ) VALUES (
                    :source_id, :schema_name, :table_name, :table_type,
                    :business_description, :m_schema, CAST(:columns AS jsonb),
                    CAST(:enrichment AS jsonb), CAST(:sample_questions AS jsonb),
                    CAST(:join_keys AS jsonb),
                    :row_count_estimate, :structural_hash, :profile_hash, :enrichment_hash,
                    :content, :content_hash, CAST(:embedding AS {vector_type}), :embedding_model,
                    now(), true, CAST(:run_id AS uuid), now(), now()
                )
                ON CONFLICT (source_id, schema_name, table_name) DO UPDATE SET
                    table_type = EXCLUDED.table_type,
                    business_description = EXCLUDED.business_description,
                    m_schema = EXCLUDED.m_schema,
                    columns = EXCLUDED.columns,
                    enrichment = EXCLUDED.enrichment,
                    sample_questions = EXCLUDED.sample_questions,
                    join_keys = EXCLUDED.join_keys,
                    row_count_estimate = EXCLUDED.row_count_estimate,
                    structural_hash = EXCLUDED.structural_hash,
                    profile_hash = EXCLUDED.profile_hash,
                    enrichment_hash = EXCLUDED.enrichment_hash,
                    content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_updated_at = now(),
                    is_active = true,
                    last_run_id = EXCLUDED.last_run_id,
                    last_seen_at = now(),
                    updated_at = now()
                """
            )
            params = []
            for record in records:
                if record.embedding is None or not record.embedding_model:
                    raise ValueError(f"Table {record.key} is missing an embedding")
                params.append(
                    {
                        "source_id": source_id,
                        "schema_name": record.schema_name,
                        "table_name": record.table_name,
                        "table_type": record.table_type,
                        "business_description": record.business_description,
                        "m_schema": record.m_schema,
                        "columns": _json(record.columns),
                        "enrichment": _json(record.enrichment),
                        "sample_questions": _json(record.sample_questions),
                        "join_keys": _json(record.join_keys),
                        "row_count_estimate": record.row_count_estimate,
                        "structural_hash": record.structural_hash,
                        "profile_hash": record.profile_hash,
                        "enrichment_hash": record.enrichment_hash,
                        "content": record.content,
                        "content_hash": record.content_hash,
                        "embedding": vector_literal(
                            record.embedding, self._settings.embedding_dimensions
                        ),
                        "embedding_model": record.embedding_model,
                        "run_id": str(run_id),
                    }
                )
            connection.execute(sql, params)

    def touch_tables(
        self,
        source_id: str,
        run_id: uuid.UUID,
        tables: Sequence[tuple[str, str]],
    ) -> None:
        if not tables:
            return
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            connection.execute(
                sa.text(
                    f"UPDATE {schema_sql}.table_catalog "
                    "SET is_active = true, last_seen_at = now(), "
                    "last_run_id = CAST(:run_id AS uuid) "
                    "WHERE source_id = :source_id AND schema_name = :schema_name "
                    "AND table_name = :table_name"
                ),
                [
                    {
                        "source_id": source_id,
                        "run_id": str(run_id),
                        "schema_name": schema_name,
                        "table_name": table_name,
                    }
                    for schema_name, table_name in tables
                ],
            )

    def deactivate_missing_tables(
        self,
        source_id: str,
        active_tables: Sequence[tuple[str, str]],
    ) -> int:
        seen = [
            {"schema_name": schema_name, "table_name": table_name}
            for schema_name, table_name in active_tables
        ]
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            result = connection.execute(
                sa.text(
                    f"UPDATE {schema_sql}.table_catalog AS catalog "
                    "SET is_active = false, updated_at = now() "
                    "WHERE catalog.source_id = :source_id "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM jsonb_to_recordset(CAST(:seen AS jsonb)) "
                    "  AS seen(schema_name text, table_name text) "
                    "  WHERE seen.schema_name = catalog.schema_name "
                    "  AND seen.table_name = catalog.table_name"
                    ")"
                ),
                {"source_id": source_id, "seen": _json(seen)},
            )
            return int(result.rowcount or 0)

    def upsert_semantics(
        self,
        source_id: str,
        run_id: uuid.UUID,
        entries: Sequence[tuple[SemanticEntry, list[float], str]],
    ) -> None:
        if not entries:
            return
        vector_type = self._vector_cast()
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            sql = sa.text(
                f"""
                INSERT INTO {schema_sql}.semantic_glossary (
                    source_id, kind, name, synonyms, description, required_tables,
                    sql_formula, sql_condition, table_name, notes, content, content_hash,
                    embedding, embedding_model, embedding_updated_at, is_active,
                    last_run_id, updated_at
                ) VALUES (
                    :source_id, :kind, :name, CAST(:synonyms AS jsonb), :description,
                    CAST(:required_tables AS jsonb), :sql_formula, :sql_condition,
                    :table_name, :notes, :content, :content_hash,
                    CAST(:embedding AS {vector_type}), :embedding_model, now(), true,
                    CAST(:run_id AS uuid), now()
                )
                ON CONFLICT (source_id, kind, name) DO UPDATE SET
                    synonyms = EXCLUDED.synonyms,
                    description = EXCLUDED.description,
                    required_tables = EXCLUDED.required_tables,
                    sql_formula = EXCLUDED.sql_formula,
                    sql_condition = EXCLUDED.sql_condition,
                    table_name = EXCLUDED.table_name,
                    notes = EXCLUDED.notes,
                    content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_updated_at = now(),
                    is_active = true,
                    last_run_id = EXCLUDED.last_run_id,
                    updated_at = now()
                """
            )
            connection.execute(
                sql,
                [
                    {
                        "source_id": source_id,
                        "kind": entry.kind,
                        "name": entry.name,
                        "synonyms": _json(entry.synonyms),
                        "description": entry.description,
                        "required_tables": _json(entry.required_tables),
                        "sql_formula": entry.sql_formula,
                        "sql_condition": entry.sql_condition,
                        "table_name": entry.table_name,
                        "notes": entry.notes,
                        "content": entry.content,
                        "content_hash": entry.content_hash,
                        "embedding": vector_literal(embedding, self._settings.embedding_dimensions),
                        "embedding_model": embedding_model,
                        "run_id": str(run_id),
                    }
                    for entry, embedding, embedding_model in entries
                ],
            )

    def deactivate_missing_semantics(
        self,
        source_id: str,
        active_entries: Sequence[SemanticEntry],
    ) -> int:
        seen = [{"kind": entry.kind, "name": entry.name} for entry in active_entries]
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            result = connection.execute(
                sa.text(
                    f"UPDATE {schema_sql}.semantic_glossary AS glossary "
                    "SET is_active = false, updated_at = now() "
                    "WHERE glossary.source_id = :source_id "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM jsonb_to_recordset(CAST(:seen AS jsonb)) "
                    "  AS seen(kind text, name text) "
                    "  WHERE seen.kind = glossary.kind AND seen.name = glossary.name"
                    ")"
                ),
                {"source_id": source_id, "seen": _json(seen)},
            )
            return int(result.rowcount or 0)

    def touch_semantics(
        self,
        source_id: str,
        run_id: uuid.UUID,
        entries: Sequence[SemanticEntry],
    ) -> None:
        if not entries:
            return
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            connection.execute(
                sa.text(
                    f"UPDATE {schema_sql}.semantic_glossary "
                    "SET is_active = true, last_run_id = CAST(:run_id AS uuid), "
                    "updated_at = now() "
                    "WHERE source_id = :source_id AND kind = :kind AND name = :name"
                ),
                [
                    {
                        "source_id": source_id,
                        "run_id": str(run_id),
                        "kind": entry.kind,
                        "name": entry.name,
                    }
                    for entry in entries
                ],
            )

    def upsert_golden_queries(
        self,
        source_id: str,
        run_id: uuid.UUID,
        entries: Sequence[tuple[GoldenEntry, list[float], str]],
    ) -> None:
        if not entries:
            return
        vector_type = self._vector_cast()
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            sql = sa.text(
                f"""
                INSERT INTO {schema_sql}.golden_sql_examples (
                    source_id, example_key, question, sql_query, tables_used, tags, notes,
                    content, content_hash, embedding, embedding_model, embedding_updated_at,
                    is_active, last_run_id, updated_at
                ) VALUES (
                    :source_id, :example_key, :question, :sql_query,
                    CAST(:tables_used AS jsonb), CAST(:tags AS jsonb), :notes,
                    :content, :content_hash, CAST(:embedding AS {vector_type}),
                    :embedding_model, now(), true, CAST(:run_id AS uuid), now()
                )
                ON CONFLICT (source_id, example_key) DO UPDATE SET
                    question = EXCLUDED.question,
                    sql_query = EXCLUDED.sql_query,
                    tables_used = EXCLUDED.tables_used,
                    tags = EXCLUDED.tags,
                    notes = EXCLUDED.notes,
                    content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_updated_at = now(),
                    is_active = true,
                    last_run_id = EXCLUDED.last_run_id,
                    updated_at = now()
                """
            )
            connection.execute(
                sql,
                [
                    {
                        "source_id": source_id,
                        "example_key": entry.example_key,
                        "question": entry.question,
                        "sql_query": entry.sql,
                        "tables_used": _json(entry.tables),
                        "tags": _json(entry.tags),
                        "notes": entry.notes,
                        "content": entry.content,
                        "content_hash": entry.content_hash,
                        "embedding": vector_literal(embedding, self._settings.embedding_dimensions),
                        "embedding_model": embedding_model,
                        "run_id": str(run_id),
                    }
                    for entry, embedding, embedding_model in entries
                ],
            )

    def deactivate_missing_golden_queries(
        self,
        source_id: str,
        active_entries: Sequence[GoldenEntry],
    ) -> int:
        seen = [{"example_key": entry.example_key} for entry in active_entries]
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            result = connection.execute(
                sa.text(
                    f"UPDATE {schema_sql}.golden_sql_examples AS golden "
                    "SET is_active = false, updated_at = now() "
                    "WHERE golden.source_id = :source_id "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM jsonb_to_recordset(CAST(:seen AS jsonb)) "
                    "  AS seen(example_key text) "
                    "  WHERE seen.example_key = golden.example_key"
                    ")"
                ),
                {"source_id": source_id, "seen": _json(seen)},
            )
            return int(result.rowcount or 0)

    def touch_golden_queries(
        self,
        source_id: str,
        run_id: uuid.UUID,
        entries: Sequence[GoldenEntry],
    ) -> None:
        if not entries:
            return
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            connection.execute(
                sa.text(
                    f"UPDATE {schema_sql}.golden_sql_examples "
                    "SET is_active = true, last_run_id = CAST(:run_id AS uuid), "
                    "updated_at = now() "
                    "WHERE source_id = :source_id AND example_key = :example_key"
                ),
                [
                    {
                        "source_id": source_id,
                        "run_id": str(run_id),
                        "example_key": entry.example_key,
                    }
                    for entry in entries
                ],
            )

    def save_join_graph(
        self,
        source_id: str,
        run_id: uuid.UUID,
        graph_json: str,
        node_count: int,
        edge_count: int,
    ) -> None:
        with self._engine.begin() as connection:
            schema_sql = quote_ident(connection, self._schema)
            connection.execute(
                sa.text(
                    f"INSERT INTO {schema_sql}.schema_graph "
                    "(source_id, graph, node_count, edge_count, last_run_id, updated_at) "
                    "VALUES (:source_id, CAST(:graph AS jsonb), :node_count, :edge_count, "
                    "CAST(:run_id AS uuid), now()) "
                    "ON CONFLICT (source_id) DO UPDATE SET "
                    "graph = EXCLUDED.graph, node_count = EXCLUDED.node_count, "
                    "edge_count = EXCLUDED.edge_count, last_run_id = EXCLUDED.last_run_id, "
                    "updated_at = now()"
                ),
                {
                    "source_id": source_id,
                    "graph": graph_json,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "run_id": str(run_id),
                },
            )

    def catalog_version(self, source_id: str) -> str:
        with self._engine.connect() as connection:
            row = connection.execute(
                sa.text(
                    f"SELECT last_run_id, updated_at FROM "
                    f"{quote_ident(connection, self._schema)}.schema_graph "
                    "WHERE source_id = :source_id"
                ),
                {"source_id": source_id},
            ).first()
        if row is None or row.last_run_id is None:
            return ""
        updated_at = row.updated_at.isoformat() if row.updated_at else ""
        return f"{row.last_run_id}:{updated_at}"

    def catalog_table_count(self, source_id: str) -> int:
        with self._engine.connect() as connection:
            count = int(
                connection.execute(
                    sa.text(
                        f"SELECT count(*) FROM {quote_ident(connection, self._schema)}."
                        "table_catalog WHERE source_id = :source_id AND is_active = true"
                    ),
                    {"source_id": source_id},
                ).scalar_one()
            )
            if count == 0:
                count = int(
                    connection.execute(
                        sa.text(
                            f"SELECT count(*) FROM {quote_ident(connection, self._schema)}."
                            "table_catalog WHERE is_active = true"
                        )
                    ).scalar_one()
                )
            return count

    def primary_source_id(self) -> str | None:
        with self._engine.connect() as connection:
            row = connection.execute(
                sa.text(
                    f"SELECT source_id FROM {quote_ident(connection, self._schema)}."
                    "table_catalog WHERE is_active = true GROUP BY source_id "
                    "ORDER BY count(*) DESC LIMIT 1"
                )
            ).first()
            return str(row[0]) if row and row[0] else None

    def latest_run(self, source_id: str) -> dict[str, Any] | None:
        with self._engine.connect() as connection:
            row = connection.execute(
                sa.text(
                    f"SELECT run_id, status, started_at, finished_at, stats, error "
                    f"FROM {quote_ident(connection, self._schema)}.ingestion_runs "
                    "WHERE source_id = :source_id ORDER BY started_at DESC LIMIT 1"
                ),
                {"source_id": source_id},
            ).first()
            if row is None:
                row = connection.execute(
                    sa.text(
                        f"SELECT run_id, status, started_at, finished_at, stats, error "
                        f"FROM {quote_ident(connection, self._schema)}.ingestion_runs "
                        "ORDER BY started_at DESC LIMIT 1"
                    )
                ).first()
        if row is None:
            return None
        return {
            "run_id": str(row.run_id),
            "status": row.status,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "stats": _json_field(row.stats, {}),
            "error": row.error,
        }

    def health(self) -> dict[str, Any]:
        with self._engine.connect() as connection:
            vector_version = connection.execute(
                sa.text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
            schema_exists = connection.execute(
                sa.text("SELECT 1 FROM pg_namespace WHERE nspname = :schema"),
                {"schema": self._schema},
            ).scalar_one_or_none()
            counts: dict[str, int] = {}
            if schema_exists:
                for table in (
                    "table_catalog",
                    "semantic_glossary",
                    "golden_sql_examples",
                    "schema_graph",
                ):
                    counts[table] = int(
                        connection.execute(
                            sa.text(
                                f"SELECT count(*) FROM {quote_ident(connection, self._schema)}."
                                f"{quote_ident(connection, table)}"
                            )
                        ).scalar_one()
                    )
        return {
            "vector_version": vector_version,
            "schema_exists": bool(schema_exists),
            "counts": counts,
        }
