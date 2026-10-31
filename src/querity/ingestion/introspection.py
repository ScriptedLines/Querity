from __future__ import annotations

import fnmatch
from typing import Any

import sqlalchemy as sa
from sqlalchemy import types as sat
from sqlalchemy.engine import Connection, Engine

from querity.common.logging import get_logger
from querity.common.masking import is_sensitive_column
from querity.ingestion.db import qualified_name
from querity.ingestion.models import ColumnProfile, ForeignKeyRef, TableSchema
from querity.ingestion.settings import IngestionSettings

logger = get_logger(__name__)

_METADATA_TABLE_NAMES = {
    "ingestion_runs",
    "table_catalog",
    "semantic_glossary",
    "golden_sql_examples",
    "schema_graph",
}


def _matches_any(key: str, table_name: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(key, pattern) or fnmatch.fnmatchcase(table_name, pattern)
        for pattern in patterns
    )


def _is_selected(table_key: str, table_name: str, settings: IngestionSettings) -> bool:
    include = settings.include_patterns
    exclude = settings.exclude_patterns
    if exclude and _matches_any(table_key, table_name, exclude):
        return False
    return not (include and not _matches_any(table_key, table_name, include))


def _value_kind(column_type: sa.types.TypeEngine[Any]) -> str:
    if isinstance(column_type, (sat.ARRAY, sat.JSON, sat.LargeBinary, sat.PickleType)):
        return "other"
    if isinstance(column_type, sat.Enum):
        return "string"
    if isinstance(column_type, (sat.String, sat.Text, sat.Unicode, sat.Uuid)):
        return "string"
    if isinstance(column_type, (sat.DateTime, sat.Date, sat.Time, sat.Interval)):
        return "temporal"
    if isinstance(column_type, (sat.Integer, sat.Numeric, sat.Float, sat.Boolean)):
        return "boolean" if isinstance(column_type, sat.Boolean) else "numeric"
    rendered = str(column_type).upper()
    if any(token in rendered for token in ("CHAR", "TEXT", "UUID", "ENUM")):
        return "string"
    if any(token in rendered for token in ("TIMESTAMP", "DATE", "TIME", "INTERVAL")):
        return "temporal"
    if any(token in rendered for token in ("INT", "NUMERIC", "DECIMAL", "REAL", "DOUBLE", "FLOAT")):
        return "numeric"
    if "BOOL" in rendered:
        return "boolean"
    return "other"


def _to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _single_column_unique_columns(unique_sets: list[list[str]]) -> set[str]:
    return {columns[0] for columns in unique_sets if len(columns) == 1}


def _row_count_estimate(connection: Connection, schema: str, table: str) -> int | None:
    try:
        name = qualified_name(connection, schema, table)
        value = connection.execute(
            sa.text("SELECT reltuples::bigint FROM pg_class WHERE oid = to_regclass(:name)"),
            {"name": name},
        ).scalar_one_or_none()
        if value is None:
            return None
        estimate = int(value)
        return estimate if estimate >= 0 else None
    except sa.exc.SQLAlchemyError:
        logger.warning("Could not read row estimate for %s.%s", schema, table, exc_info=True)
        return None


def _unique_column_sets(inspector: sa.Inspector, schema: str, table: str) -> list[list[str]]:
    constraints: list[list[str]] = []
    try:
        for constraint in inspector.get_unique_constraints(table, schema=schema):
            columns = constraint.get("column_names") or []
            if columns:
                constraints.append(list(columns))
    except NotImplementedError:
        pass
    except sa.exc.SQLAlchemyError:
        logger.warning("Could not read unique constraints for %s.%s", schema, table)
    try:
        for index in inspector.get_indexes(table, schema=schema):
            columns = index.get("column_names") or []
            if index.get("unique") and columns and all(column is not None for column in columns):
                constraints.append(list(columns))
    except NotImplementedError:
        pass
    except sa.exc.SQLAlchemyError:
        logger.warning("Could not read indexes for %s.%s", schema, table)

    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for columns in constraints:
        key = tuple(columns)
        if key not in seen:
            seen.add(key)
            deduped.append(columns)
    return deduped


def _foreign_keys(inspector: sa.Inspector, schema: str, table: str) -> list[ForeignKeyRef]:
    try:
        raw_foreign_keys = inspector.get_foreign_keys(table, schema=schema)
    except NotImplementedError:
        return []
    except sa.exc.SQLAlchemyError:
        logger.warning("Could not read foreign keys for %s.%s", schema, table, exc_info=True)
        return []

    foreign_keys: list[ForeignKeyRef] = []
    for fk in raw_foreign_keys:
        constrained = list(fk.get("constrained_columns") or [])
        referred = list(fk.get("referred_columns") or [])
        referred_table = fk.get("referred_table")
        if not constrained or not referred or not referred_table:
            continue
        if len(constrained) != len(referred):
            logger.warning(
                "Skipping malformed foreign key on %s.%s: %d constrained columns, "
                "%d referred columns",
                schema,
                table,
                len(constrained),
                len(referred),
            )
            continue
        foreign_keys.append(
            ForeignKeyRef(
                name=fk.get("name"),
                constrained_schema=schema,
                constrained_table=table,
                constrained_columns=constrained,
                referred_schema=fk.get("referred_schema") or schema,
                referred_table=referred_table,
                referred_columns=referred,
            )
        )
    return foreign_keys


def _introspect_table(
    connection: Connection,
    inspector: sa.Inspector,
    schema: str,
    table: str,
    table_type: str,
    settings: IngestionSettings,
) -> TableSchema:
    raw_columns = inspector.get_columns(table, schema=schema)
    primary_key = list(
        (inspector.get_pk_constraint(table, schema=schema) or {}).get("constrained_columns") or []
    )
    foreign_keys = _foreign_keys(inspector, schema, table)
    unique_sets = _unique_column_sets(inspector, schema, table)
    if primary_key:
        unique_sets.append(list(primary_key))
    unique_sets = list({tuple(columns): list(columns) for columns in unique_sets}.values())
    unique_columns = {columns[0] for columns in unique_sets if len(columns) == 1}

    fk_by_column: dict[str, str] = {}
    for fk in foreign_keys:
        if len(fk.constrained_columns) == 1:
            target = f"{fk.referred_schema}.{fk.referred_table}.{fk.referred_columns[0]}"
            fk_by_column.setdefault(fk.constrained_columns[0], target)
        else:
            target = f"{fk.referred_schema}.{fk.referred_table}({', '.join(fk.referred_columns)})"
            for column in fk.constrained_columns:
                fk_by_column.setdefault(column, target)

    columns: list[ColumnProfile] = []
    for raw_column in raw_columns:
        name = raw_column["name"]
        column_type = raw_column["type"]
        comment = raw_column.get("comment")
        columns.append(
            ColumnProfile(
                name=name,
                data_type=str(column_type) or repr(column_type),
                value_kind=_value_kind(column_type),  # type: ignore[arg-type]
                nullable=bool(raw_column.get("nullable", True)),
                default=_to_string(raw_column.get("default")),
                comment=comment,
                primary_key=name in primary_key,
                unique=name in unique_columns,
                foreign_key=fk_by_column.get(name),
                is_sensitive=is_sensitive_column(
                    name,
                    comment=comment,
                    extra_patterns=settings.extra_sensitive_patterns,
                ),
            )
        )

    try:
        comment = (inspector.get_table_comment(table, schema=schema) or {}).get("text")
    except (NotImplementedError, sa.exc.SQLAlchemyError):
        comment = None

    return TableSchema(
        schema_name=schema,
        table_name=table,
        table_type=table_type,
        comment=comment,
        columns=columns,
        primary_key=primary_key,
        unique_constraints=unique_sets,
        foreign_keys=foreign_keys,
        row_count_estimate=_row_count_estimate(connection, schema, table),
    )


def introspect_tables(engine: Engine, settings: IngestionSettings) -> list[TableSchema]:
    tables: list[TableSchema] = []
    with engine.connect() as connection:
        inspector = sa.inspect(connection)
        available_schemas = set(inspector.get_schema_names())
        for schema in settings.schemas:
            if schema not in available_schemas:
                logger.warning("Schema %s does not exist or is not accessible; skipping", schema)
                continue
            candidates: list[tuple[str, str]] = [
                (table, "table")
                for table in inspector.get_table_names(schema=schema)
                if not (schema == settings.metadata_schema and table in _METADATA_TABLE_NAMES)
            ]
            if settings.include_views:
                candidates.extend(
                    (view, "view") for view in inspector.get_view_names(schema=schema)
                )
                materialized = getattr(inspector, "get_materialized_view_names", None)
                if callable(materialized):
                    try:
                        candidates.extend(
                            (view, "materialized_view") for view in materialized(schema=schema)
                        )
                    except (NotImplementedError, sa.exc.SQLAlchemyError):
                        logger.warning("Could not list materialized views in %s", schema)

            for table_name, table_type in sorted(set(candidates)):
                table_key = f"{schema}.{table_name}"
                if not _is_selected(table_key, table_name, settings):
                    logger.debug("Skipping table %s due to include/exclude filters", table_key)
                    continue
                try:
                    table = _introspect_table(
                        connection, inspector, schema, table_name, table_type, settings
                    )
                except (sa.exc.SQLAlchemyError, NotImplementedError):
                    logger.exception("Failed to introspect %s", table_key)
                    continue
                tables.append(table)
                logger.debug(
                    "Introspected table",
                    extra={"table": table.key, "columns": len(table.columns)},
                )

    tables.sort(key=lambda item: item.key)
    return tables
