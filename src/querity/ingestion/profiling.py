from __future__ import annotations

import json
import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from querity.common.logging import get_logger
from querity.common.masking import mask_value, truncate_value
from querity.ingestion.db import qualified_name, quote_ident, readonly_connection
from querity.ingestion.models import ColumnProfile, RowSample, TableSchema
from querity.ingestion.settings import IngestionSettings

logger = get_logger(__name__)

_PROFILEABLE_KINDS = {"string", "numeric", "temporal", "boolean"}
_UNSAFE_SAMPLE_TYPES = ("BYTEA", "BLOB", "BINARY", "JSON", "ARRAY")


def _execute(
    connection: Connection,
    sql: sa.TextClause,
    params: dict[str, Any] | None = None,
) -> list[Any] | None:
    savepoint = connection.begin_nested()
    try:
        rows = connection.execute(sql, params or {}).fetchall()
    except sa.exc.SQLAlchemyError as exc:
        savepoint.rollback()
        logger.debug("Profiling statement failed: %s", exc)
        return None
    savepoint.commit()
    return rows


def _load_statistics(connection: Connection, table: TableSchema) -> dict[str, dict[str, Any]]:
    rows = _execute(
        connection,
        sa.text(
            "SELECT attname, n_distinct, null_frac "
            "FROM pg_stats WHERE schemaname = :schema AND tablename = :table"
        ),
        {"schema": table.schema_name, "table": table.table_name},
    )
    if not rows:
        return {}
    return {
        row.attname: {
            "n_distinct": float(row.n_distinct) if row.n_distinct is not None else None,
            "null_frac": float(row.null_frac) if row.null_frac is not None else None,
        }
        for row in rows
    }


def _estimated_distinct(
    raw_n_distinct: float | None,
    row_count_estimate: int | None,
) -> float | None:
    if raw_n_distinct is None:
        return None
    if raw_n_distinct >= 0:
        return raw_n_distinct
    if row_count_estimate is None:
        return None
    return abs(raw_n_distinct) * max(row_count_estimate, 0)


def _stringify(value: Any, *, max_chars: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date | time):
        return truncate_value(value.isoformat(), max_chars)
    if isinstance(value, Decimal):
        return truncate_value(str(value), max_chars)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes | bytearray | memoryview):
        return "<binary>"
    if isinstance(value, dict | list | tuple):
        try:
            return truncate_value(json.dumps(value, default=str, ensure_ascii=False), max_chars)
        except (TypeError, ValueError):
            return truncate_value(str(value), max_chars)
    return truncate_value(str(value), max_chars)


def _stringify_sampled(
    value: Any, column: ColumnProfile, settings: IngestionSettings
) -> str | None:
    if value is None:
        return None
    if column.is_sensitive and settings.pii_masking:
        return mask_value(value, column.name)
    return _stringify(value, max_chars=settings.max_sample_value_chars)


def _sample_values(
    connection: Connection,
    table: TableSchema,
    column: ColumnProfile,
    settings: IngestionSettings,
) -> list[str]:
    if column.value_kind not in _PROFILEABLE_KINDS:
        return []
    qualified = qualified_name(connection, table.schema_name, table.table_name)
    quoted_column = quote_ident(connection, column.name)
    limit = settings.high_cardinality_sample_size
    rows = _execute(
        connection,
        sa.text(
            f"SELECT {quoted_column} FROM {qualified} "
            f"WHERE {quoted_column} IS NOT NULL LIMIT {int(limit)}"
        ),
    )
    if not rows:
        return []
    values: list[str] = []
    for row in rows:
        rendered = _stringify_sampled(row[0], column, settings)
        if rendered is not None and rendered not in values:
            values.append(rendered)
    return values


def _distinct_values(
    connection: Connection,
    table: TableSchema,
    column: ColumnProfile,
    settings: IngestionSettings,
) -> list[str]:
    if column.is_sensitive or column.value_kind not in _PROFILEABLE_KINDS:
        return []
    qualified = qualified_name(connection, table.schema_name, table.table_name)
    quoted_column = quote_ident(connection, column.name)
    limit = settings.low_cardinality_threshold + 1
    rows = _execute(
        connection,
        sa.text(
            f"SELECT DISTINCT {quoted_column} FROM {qualified} "
            f"WHERE {quoted_column} IS NOT NULL "
            f"ORDER BY {quoted_column} LIMIT {int(limit)}"
        ),
    )
    if rows is None or len(rows) > settings.low_cardinality_threshold:
        return []
    values: list[str] = []
    for row in rows:
        rendered = _stringify(row[0], max_chars=settings.max_sample_value_chars)
        if rendered is not None and rendered not in values:
            values.append(rendered)
    return values


def _value_range(
    connection: Connection,
    table: TableSchema,
    column: ColumnProfile,
    settings: IngestionSettings,
) -> tuple[str | None, str | None]:
    if (
        column.is_sensitive
        or column.value_kind not in {"numeric", "temporal"}
        or column.primary_key
        or (column.value_kind == "numeric" and not settings.profile_numeric_ranges)
    ):
        return None, None
    qualified = qualified_name(connection, table.schema_name, table.table_name)
    quoted_column = quote_ident(connection, column.name)
    rows = _execute(
        connection,
        sa.text(
            f"SELECT MIN({quoted_column}) AS min_value, MAX({quoted_column}) AS max_value "
            f"FROM {qualified} WHERE {quoted_column} IS NOT NULL"
        ),
    )
    if not rows:
        return None, None
    row = rows[0]
    return (
        _stringify(row.min_value, max_chars=settings.max_sample_value_chars),
        _stringify(row.max_value, max_chars=settings.max_sample_value_chars),
    )


def _safe_sample_columns(table: TableSchema, settings: IngestionSettings) -> list[ColumnProfile]:
    candidates = [
        column
        for column in table.columns
        if not column.is_sensitive
        and column.value_kind != "other"
        and not any(token in column.data_type.upper() for token in _UNSAFE_SAMPLE_TYPES)
    ]

    def priority(column: ColumnProfile) -> tuple[int, int, str]:
        if column.primary_key or column.foreign_key:
            rank = 0
        elif column.value_kind == "temporal":
            rank = 1
        elif bool(column.allowed_values):
            rank = 2
        else:
            rank = 3
        return rank, 0, column.name

    candidates.sort(key=priority)
    return candidates[: settings.max_row_sample_columns]


def _row_sample(
    connection: Connection,
    table: TableSchema,
    settings: IngestionSettings,
) -> RowSample:
    columns = _safe_sample_columns(table, settings)
    if not columns:
        return RowSample()
    qualified = qualified_name(connection, table.schema_name, table.table_name)
    select_list = ", ".join(quote_ident(connection, column.name) for column in columns)
    rows = _execute(
        connection,
        sa.text(f"SELECT {select_list} FROM {qualified} LIMIT {int(settings.profile_sample_rows)}"),
    )
    if not rows:
        return RowSample(columns=[column.name for column in columns], rows=[])
    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered_rows.append(
            [_stringify(value, max_chars=settings.max_sample_value_chars) or "" for value in row]
        )
    return RowSample(columns=[column.name for column in columns], rows=rendered_rows)


def profile_table(
    engine: Engine,
    table: TableSchema,
    settings: IngestionSettings,
) -> tuple[TableSchema, RowSample]:
    profiled = table.model_copy(deep=True)
    with readonly_connection(engine) as connection:
        stats = _load_statistics(connection, profiled)
        for column in profiled.columns:
            column_stats = stats.get(column.name, {})
            raw_n_distinct = column_stats.get("n_distinct")
            column.distinct_count_estimate = _estimated_distinct(
                raw_n_distinct, profiled.row_count_estimate
            )
            column.null_fraction = column_stats.get("null_frac")

            if column.value_kind not in _PROFILEABLE_KINDS:
                continue

            low_cardinality = (
                column.distinct_count_estimate is not None
                and column.distinct_count_estimate <= settings.low_cardinality_threshold
            )
            if low_cardinality or column.distinct_count_estimate is None:
                allowed_values = _distinct_values(connection, profiled, column, settings)
                if allowed_values and (low_cardinality or column.distinct_count_estimate is None):
                    column.allowed_values = allowed_values
                    column.distinct_count_estimate = float(len(allowed_values))

            if not column.allowed_values:
                column.sample_values = _sample_values(connection, profiled, column, settings)
            column.min_value, column.max_value = _value_range(
                connection, profiled, column, settings
            )

        row_sample = _row_sample(connection, profiled, settings)
    return profiled, row_sample
