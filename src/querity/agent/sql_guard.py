from __future__ import annotations

import hashlib
from typing import Any

import sqlglot
from sqlalchemy.engine import Engine
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from querity.agent.catalog import CatalogSnapshot
from querity.agent.settings import AgentSettings
from querity.agent.state import GuardResult, SQLIssue
from querity.common.cache import LocalCache, hash_key
from querity.ingestion.semantic_layer import (
    GoldenSQLValidationError,
    validate_golden_query_against_source,
)

_GUARD_NAMESPACE = "guard"
_GUARD_CACHE_VERSION = "guard-v1"
_GUARD_TTL_SECONDS = 7 * 24 * 60 * 60
_GUARD_EXPLAIN_TTL_SECONDS = 5 * 60

_DISALLOWED_NODES = (
    "Insert",
    "Update",
    "Delete",
    "Create",
    "Drop",
    "Alter",
    "Merge",
    "Into",
    "Command",
    "Copy",
    "Grant",
    "Revoke",
    "Transaction",
    "Commit",
    "Rollback",
    "TruncateTable",
    "Lock",
    "Set",
    "Reset",
    "Use",
    "Vacuum",
    "Analyze",
    "Call",
    "Do",
    "Listen",
    "Notify",
    "Unlisten",
)

_BLOCKED_FUNCTIONS = {
    "pg_sleep",
    "pg_sleep_for",
    "pg_sleep_until",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    "pg_rotate_logfile",
    "pg_switch_wal",
    "pg_create_restore_point",
    "pg_drop_replication_slot",
    "pg_stat_reset",
    "pg_stat_reset_shared",
    "pg_stat_reset_single_table_counters",
    "lo_import",
    "lo_export",
    "lo_unlink",
    "dblink",
    "dblink_exec",
    "dblink_connect",
    "set_config",
    "query_to_xml",
    "query_to_xmlschema",
    "query_to_xml_and_xmlschema",
    "current_setting",
    "nextval",
    "setval",
    "pg_notify",
    "pg_advisory_lock",
    "pg_advisory_lock_shared",
    "pg_advisory_unlock",
    "pg_advisory_unlock_all",
    "pg_advisory_unlock_shared",
    "pg_advisory_xact_lock",
    "pg_advisory_xact_lock_shared",
    "pg_try_advisory_lock",
    "pg_try_advisory_lock_shared",
    "pg_try_advisory_xact_lock",
    "pg_try_advisory_xact_lock_shared",
    "pg_logical_emit_message",
    "pg_create_logical_replication_slot",
    "pg_create_physical_replication_slot",
    "pg_replication_slot_advance",
    "pg_backup_start",
    "pg_backup_stop",
    "pg_start_backup",
    "pg_stop_backup",
    "pg_promote",
    "lo_create",
    "lo_creat",
    "lo_from_bytea",
    "lo_put",
    "lowrite",
    "lo_truncate",
    "lo_truncate64",
    "lo_open",
}


class GuardFailure(ValueError):
    def __init__(self, message: str, *, code: str = "guard_failure") -> None:
        super().__init__(message)
        self.code = code


def _hash(sql: str) -> str:
    return hashlib.sha256(" ".join(sql.split()).encode("utf-8")).hexdigest()


def _disallowed_node_types() -> tuple[type[Any], ...]:
    return tuple(getattr(exp, name) for name in _DISALLOWED_NODES if hasattr(exp, name))


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        raw = str(node.this)
    else:
        try:
            raw = str(node.sql_name())
        except Exception:
            raw = type(node).__name__
    return raw.split(".")[-1].strip('"').lower()


def _select_aliases(scope: Any) -> set[str]:
    expression = scope.expression
    names = getattr(expression, "named_selects", None)
    if names:
        return {str(name).lower() for name in names}
    aliases: set[str] = set()
    for select in getattr(expression, "selects", []):
        alias = select.alias_or_name
        if alias:
            aliases.add(alias.lower())
    return aliases


def _is_select_output(column: exp.Column) -> bool:
    parent = column.parent
    if isinstance(parent, exp.Alias):
        parent = parent.parent
    return isinstance(parent, exp.Select) and column in parent.expressions


def _table_reference(table: exp.Table) -> str:
    parts = [str(part) for part in (table.catalog, table.db, table.name) if part]
    return ".".join(parts)


def _limit_value(statement: exp.Expression) -> int | None:
    limit = statement.args.get("limit")
    if limit is None:
        return None
    expression = getattr(limit, "expression", None)
    if isinstance(expression, exp.Literal) and expression.is_int:
        try:
            return int(expression.this)
        except (TypeError, ValueError):
            return None
    return None


def _apply_limit(statement: exp.Expression, sql: str, max_rows: int) -> tuple[str, int]:
    current = _limit_value(statement)
    if current is not None and current <= max_rows:
        return sql, current
    return (
        f"SELECT * FROM ({sql}) AS _querity_result LIMIT {max_rows}",
        max_rows,
    )


def _validate_query(
    statement: exp.Expression, catalog: CatalogSnapshot
) -> tuple[list[SQLIssue], list[SQLIssue], list[str]]:
    errors: list[SQLIssue] = []
    warnings: list[SQLIssue] = []
    tables_used: set[str] = set()

    if not isinstance(statement, exp.Query):
        errors.append(
            SQLIssue(
                code="not_select",
                message=f"Only read-only SELECT queries are allowed; got {type(statement).__name__}.",
            )
        )
        return errors, warnings, []

    disallowed = _disallowed_node_types()
    for node in statement.walk():
        if isinstance(node, disallowed):
            errors.append(
                SQLIssue(
                    code="disallowed_statement",
                    message=f"SQL contains a disallowed construct: {type(node).__name__}.",
                )
            )
            break

    for node in statement.walk():
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if name in _BLOCKED_FUNCTIONS:
                errors.append(
                    SQLIssue(
                        code="blocked_function",
                        message=f"Function {name} is not allowed in generated SQL.",
                    )
                )

    cte_names = {
        cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE) if cte.alias_or_name
    }

    for table in statement.find_all(exp.Table):
        if not table.db and table.name.lower() in cte_names:
            continue
        reference = _table_reference(table)
        resolved = catalog.resolve_table(reference)
        if resolved is None:
            errors.append(
                SQLIssue(
                    code="unknown_table",
                    message=f"Table {reference!r} is not in the active catalog.",
                    suggestion="Use only fully qualified tables from the catalog.",
                )
            )
        else:
            tables_used.add(resolved)

    for star in statement.find_all(exp.Star):
        parent = star.parent
        if isinstance(parent, exp.Count):
            continue
        warnings.append(
            SQLIssue(
                code="select_star",
                message="SELECT * can return sensitive or unnecessary columns.",
                severity="warning",
                suggestion="Select only the columns needed to answer the question.",
            )
        )
        break

    for scope in traverse_scope(statement):
        aliases = {key.lower(): value for key, value in scope.sources.items()}
        table_sources: dict[str, Any] = {}
        for name, source in scope.sources.items():
            if isinstance(source, exp.Table):
                reference = _table_reference(source)
                resolved = catalog.resolve_table(reference)
                if resolved:
                    table_sources[name.lower()] = catalog.tables[resolved]
        select_aliases = _select_aliases(scope)
        for column in scope.columns:
            if column.name == "*":
                continue
            column_name = column.name.lower()
            if column.table:
                source = aliases.get(column.table.lower())
                if isinstance(source, exp.Table):
                    reference = _table_reference(source)
                    resolved = catalog.resolve_table(reference)
                    info = catalog.tables.get(resolved) if resolved else None
                    if info is not None and column_name not in info.columns:
                        errors.append(
                            SQLIssue(
                                code="unknown_column",
                                message=(f"Column {column.name!r} does not exist on {info.key}."),
                            )
                        )
                elif isinstance(source, Scope) or column.table.lower() in cte_names:
                    continue
                else:
                    errors.append(
                        SQLIssue(
                            code="unknown_alias",
                            message=f"Unknown table alias or CTE {column.table!r}.",
                        )
                    )
                continue

            if column_name in select_aliases and not _is_select_output(column):
                continue
            if not table_sources:
                continue
            if not any(column_name in info.columns for info in table_sources.values()):
                errors.append(
                    SQLIssue(
                        code="unknown_column",
                        message=f"Unqualified column {column.name!r} was not found in scope.",
                    )
                )

    if not tables_used:
        warnings.append(
            SQLIssue(
                code="no_tables",
                message="The query does not reference a catalog table.",
                severity="warning",
            )
        )

    return errors, warnings, sorted(tables_used)


class SqlGuard:
    def __init__(
        self,
        settings: AgentSettings,
        catalog: CatalogSnapshot,
        source_engine: Engine,
        *,
        cache: LocalCache | None = None,
        catalog_version: str = "",
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._source_engine = source_engine
        self._cache = cache
        self._catalog_version = catalog_version

    def validate(self, sql: str) -> GuardResult:
        cache_key = self._cache_key(sql)
        if self._cache is not None:
            cached = self._cache.get_json(_GUARD_NAMESPACE, cache_key)
            if cached is not None:
                try:
                    return GuardResult.model_validate(cached)
                except Exception:
                    self._cache.delete(_GUARD_NAMESPACE, cache_key)
        result = self._validate(sql)
        if self._cache is not None and result.ok:
            ttl = (
                _GUARD_EXPLAIN_TTL_SECONDS
                if self._settings.enable_explain_guard
                else _GUARD_TTL_SECONDS
            )
            self._cache.set_json(
                _GUARD_NAMESPACE,
                cache_key,
                result.model_dump(mode="json"),
                ttl=ttl,
            )
        return result

    def _cache_key(self, sql: str) -> str:
        return hash_key(
            _GUARD_CACHE_VERSION,
            getattr(self._settings, "db_fingerprint", ""),
            self._settings.source_id,
            self._settings.metadata_schema,
            self._catalog_version,
            self._settings.max_result_rows,
            self._settings.enable_explain_guard,
            sql,
        )

    def _validate(self, sql: str) -> GuardResult:
        cleaned = sql.strip()
        if not cleaned:
            return self._failure(sql, "The model returned an empty SQL query.", code="empty_sql")
        if "\x00" in cleaned:
            return self._failure(sql, "SQL contains a NUL byte.", code="invalid_sql")
        while cleaned.endswith(";"):
            cleaned = cleaned[:-1].rstrip()
        try:
            statements = sqlglot.parse(cleaned, read="postgres")
        except Exception as exc:
            return self._failure(sql, f"Could not parse SQL: {exc}", code="parse_error")
        if len(statements) != 1 or statements[0] is None:
            return self._failure(
                sql,
                "SQL must contain exactly one statement.",
                code="multiple_statements",
            )
        statement = statements[0]
        if any(
            comment
            for node in statement.walk()
            for comment in (getattr(node, "comments", None) or [])
        ):
            return self._failure(
                sql,
                "SQL comments are not allowed in generated queries.",
                code="sql_comments",
            )
        try:
            errors, warnings, tables_used = _validate_query(statement, self._catalog)
            if errors:
                return GuardResult(
                    ok=False,
                    sql=cleaned,
                    sql_hash=_hash(cleaned),
                    errors=errors,
                    warnings=warnings,
                    tables_used=tables_used,
                )
            normalized = statement.sql(dialect="postgres")
            normalized, limit_applied = _apply_limit(
                statement,
                normalized,
                self._settings.max_result_rows,
            )
        except Exception as exc:
            return self._failure(
                sql,
                f"Could not validate the SQL structure: {exc}",
                code="validation_error",
            )
        if self._settings.enable_explain_guard:
            try:
                validate_golden_query_against_source(self._source_engine, normalized)
            except GoldenSQLValidationError as exc:
                return GuardResult(
                    ok=False,
                    sql=normalized,
                    sql_hash=_hash(normalized),
                    errors=[
                        SQLIssue(
                            code="explain_failed",
                            message=f"PostgreSQL rejected the query during EXPLAIN: {exc}",
                        )
                    ],
                    warnings=warnings,
                    tables_used=tables_used,
                    limit_applied=limit_applied,
                )
            except Exception as exc:
                return GuardResult(
                    ok=False,
                    sql=normalized,
                    sql_hash=_hash(normalized),
                    errors=[
                        SQLIssue(
                            code="explain_error",
                            message=f"Could not validate the query with EXPLAIN: {exc}",
                        )
                    ],
                    warnings=warnings,
                    tables_used=tables_used,
                    limit_applied=limit_applied,
                )
        return GuardResult(
            ok=True,
            sql=normalized,
            sql_hash=_hash(normalized),
            errors=[],
            warnings=warnings,
            tables_used=tables_used,
            limit_applied=limit_applied,
        )

    def _failure(self, sql: str, message: str, *, code: str) -> GuardResult:
        return GuardResult(
            ok=False,
            sql=sql,
            sql_hash=_hash(sql),
            errors=[SQLIssue(code=code, message=message)],
        )
