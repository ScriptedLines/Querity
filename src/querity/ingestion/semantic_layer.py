from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import sqlalchemy as sa
import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from querity.common.logging import get_logger
from querity.ingestion.db import readonly_connection
from querity.ingestion.hashing import sha256_text, stable_json

try:
    import sqlglot
    from sqlglot import exp
except ImportError:
    sqlglot = None
    exp = None

logger = get_logger(__name__)

SEMANTIC_EMBEDDING_VERSION = "semantic-embedding-v2"
GOLDEN_EMBEDDING_VERSION = "golden-embedding-v2"


class MetricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Business name of the metric")
    synonyms: list[str] = Field(
        default_factory=list,
        description="Alternative names an analyst might use",
    )
    description: str = Field(default="", description="What the metric measures")
    required_tables: list[str] = Field(
        min_length=1,
        description="Fully qualified tables referenced by the formula",
    )
    sql_formula: str = Field(
        min_length=1,
        description="PostgreSQL aggregate expression without SELECT/FROM or a trailing semicolon",
    )
    notes: str | None = Field(default=None, description="Exclusions or interpretation notes")

    @field_validator("name", "sql_formula")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value


class FilterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Business name of the filter")
    synonyms: list[str] = Field(
        default_factory=list,
        description="Alternative names an analyst might use",
    )
    description: str = Field(default="", description="What the filter selects")
    table: str = Field(min_length=1, description="Fully qualified table the filter applies to")
    sql_condition: str = Field(
        min_length=1,
        description="PostgreSQL boolean expression without SELECT, semicolons or DDL",
    )
    notes: str | None = Field(default=None, description="Caveats about using the filter")

    @field_validator("name", "table", "sql_condition")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value


class SemanticLayerFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricDefinition] = Field(default_factory=list)
    filters: list[FilterDefinition] = Field(default_factory=list)


class SemanticEntry(BaseModel):
    kind: Literal["metric", "filter"]
    name: str
    synonyms: list[str]
    description: str
    required_tables: list[str]
    sql_formula: str | None = None
    sql_condition: str | None = None
    table_name: str | None = None
    notes: str | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}"

    @property
    def content(self) -> str:
        return stable_json(self.model_dump(mode="json"))

    @property
    def content_hash(self) -> str:
        return sha256_text(
            stable_json(
                {
                    "version": SEMANTIC_EMBEDDING_VERSION,
                    "content": self.model_dump(mode="json"),
                    "embedding_text": self.embedding_text,
                }
            )
        )

    @property
    def embedding_text(self) -> str:
        lines = [
            f"Type: {self.kind}",
            f"Name: {self.name}",
        ]
        if self.synonyms:
            lines.append(f"Synonyms: {', '.join(self.synonyms)}")
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.required_tables:
            lines.append(f"Required tables: {', '.join(self.required_tables)}")
        if self.table_name:
            lines.append(f"Table: {self.table_name}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


class GoldenQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    sql: str = Field(
        min_length=1,
        validation_alias=AliasChoices("sql", "sql_query"),
    )
    tables: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("tables", "tables_used"),
    )
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("question", "sql")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value


class GoldenEntry(BaseModel):
    example_key: str
    question: str
    sql: str
    tables: list[str]
    tags: list[str]
    notes: str | None = None

    @property
    def content(self) -> str:
        return stable_json(self.model_dump(mode="json"))

    @property
    def content_hash(self) -> str:
        return sha256_text(
            stable_json(
                {
                    "version": GOLDEN_EMBEDDING_VERSION,
                    "content": self.model_dump(mode="json"),
                    "embedding_text": self.embedding_text,
                }
            )
        )

    @property
    def embedding_text(self) -> str:
        lines = [f"Question: {self.question}"]
        if self.tables:
            lines.append(f"Tables: {', '.join(self.tables)}")
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


class GoldenSQLValidationError(ValueError):
    pass


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def semantic_entries_from_layer(parsed: SemanticLayerFile) -> list[SemanticEntry]:
    entries: list[SemanticEntry] = []
    seen: set[str] = set()
    for metric in parsed.metrics:
        entry = SemanticEntry(
            kind="metric",
            name=metric.name,
            synonyms=metric.synonyms,
            description=metric.description,
            required_tables=metric.required_tables,
            sql_formula=metric.sql_formula,
            notes=metric.notes,
        )
        if entry.key in seen:
            raise ValueError(f"Duplicate semantic metric: {metric.name}")
        seen.add(entry.key)
        entries.append(entry)
    for filter_definition in parsed.filters:
        entry = SemanticEntry(
            kind="filter",
            name=filter_definition.name,
            synonyms=filter_definition.synonyms,
            description=filter_definition.description,
            required_tables=[filter_definition.table],
            sql_condition=filter_definition.sql_condition,
            table_name=filter_definition.table,
            notes=filter_definition.notes,
        )
        if entry.key in seen:
            raise ValueError(f"Duplicate semantic filter: {filter_definition.name}")
        seen.add(entry.key)
        entries.append(entry)
    return entries


def load_semantic_layer(path: Path) -> list[SemanticEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ValueError(f"Semantic layer file must contain a YAML mapping: {path}")
    return semantic_entries_from_layer(SemanticLayerFile.model_validate(raw))


def _load_golden_payload(path: Path) -> list[dict[str, Any]]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("examples", "golden_queries", "queries"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Golden query file must be a JSON list or an object with an 'examples' list")


def load_golden_queries(path: Path) -> list[GoldenEntry]:
    specs = [GoldenQuerySpec.model_validate(item) for item in _load_golden_payload(path)]
    entries: list[GoldenEntry] = []
    seen: set[str] = set()
    for spec in specs:
        normalized = _normalize_question(spec.question)
        example_key = sha256_text(normalized)[:24]
        if example_key in seen:
            raise ValueError(f"Duplicate golden question: {spec.question}")
        seen.add(example_key)
        entries.append(
            GoldenEntry(
                example_key=example_key,
                question=spec.question,
                sql=spec.sql.strip().rstrip(";"),
                tables=spec.tables,
                tags=spec.tags,
                notes=spec.notes,
            )
        )
    return entries


_DISALLOWED_NODE_NAMES = (
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
)


def validate_read_only_sql(sql: str) -> str:
    if sqlglot is None or exp is None:
        raise GoldenSQLValidationError("sqlglot is required to validate golden SQL")
    if "\x00" in sql:
        raise GoldenSQLValidationError("SQL cannot contain NUL bytes")
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except Exception as exc:
        raise GoldenSQLValidationError(f"Could not parse SQL: {exc}") from exc
    if len(statements) != 1 or statements[0] is None:
        raise GoldenSQLValidationError("Golden SQL must contain exactly one statement")
    statement = statements[0]
    _ensure_read_only(statement)
    return statement.sql(dialect="postgres")


def validate_golden_query_against_source(engine: sa.Engine, sql: str) -> None:
    normalized = validate_read_only_sql(sql)
    with readonly_connection(engine) as connection:
        try:
            connection.execute(sa.text(f"EXPLAIN (FORMAT JSON, COSTS FALSE) {normalized}"))
        except sa.exc.SQLAlchemyError as exc:
            raise GoldenSQLValidationError(f"EXPLAIN failed: {exc}") from exc


def _disallowed_node_types() -> tuple[type[Any], ...]:
    if exp is None:
        return ()
    return tuple(getattr(exp, name) for name in _DISALLOWED_NODE_NAMES if hasattr(exp, name))


def _ensure_read_only(statement: Any) -> None:
    if not isinstance(statement, exp.Query):
        raise GoldenSQLValidationError(f"SQL must be read-only; found {type(statement).__name__}")
    disallowed = _disallowed_node_types()
    for node in statement.walk():
        if isinstance(node, disallowed):
            raise GoldenSQLValidationError(
                f"SQL contains a disallowed statement node: {type(node).__name__}"
            )


def validate_semantic_expression(
    expression: str,
    *,
    kind: Literal["metric", "filter"],
    tables: list[str],
) -> str:
    """Validate a generated SQL fragment before it enters the glossary.

    Metrics are aggregate expressions and filters are boolean predicates. Both are
    parsed inside a synthetic read-only SELECT so generated SQL can never contain a
    second statement or write operation, and table references must be in the catalog.
    """

    if sqlglot is None or exp is None:
        raise GoldenSQLValidationError("sqlglot is required to validate semantic SQL")
    cleaned = expression.strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if not cleaned:
        raise GoldenSQLValidationError("Semantic SQL expression cannot be empty")
    if "\x00" in cleaned:
        raise GoldenSQLValidationError("Semantic SQL expression cannot contain NUL bytes")
    if ";" in cleaned:
        raise GoldenSQLValidationError("Semantic SQL expression cannot contain multiple statements")
    if "--" in cleaned or "/*" in cleaned:
        raise GoldenSQLValidationError("Semantic SQL expression cannot contain comments")
    if not tables:
        raise GoldenSQLValidationError("Semantic SQL expression requires at least one table")

    if kind == "metric":
        wrapped = f"SELECT {cleaned} FROM {tables[0]}"
    else:
        wrapped = f"SELECT 1 FROM {tables[0]} WHERE {cleaned}"

    try:
        statements = sqlglot.parse(wrapped, read="postgres")
    except Exception as exc:
        raise GoldenSQLValidationError(f"Could not parse semantic SQL expression: {exc}") from exc
    if len(statements) != 1 or statements[0] is None:
        raise GoldenSQLValidationError("Semantic SQL expression must contain exactly one statement")
    statement = statements[0]
    _ensure_read_only(statement)

    allowed = set()
    for table in tables:
        normalized = ".".join(part.strip().strip('"') for part in table.split("."))
        allowed.add(normalized.lower())
        allowed.add(normalized.rsplit(".", 1)[-1].lower())
    for table_expression in statement.find_all(exp.Table):
        if table_expression.db:
            reference = f"{table_expression.db}.{table_expression.name}"
        else:
            reference = table_expression.name
        if reference.lower() not in allowed:
            raise GoldenSQLValidationError(
                f"Semantic SQL expression references unknown table: {reference}"
            )
    return cleaned


def _clean_text(value: str | None, limit: int) -> str:
    return (value or "").strip()[:limit]


def _clean_synonyms(values: list[str]) -> list[str]:
    synonyms: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value, 100)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            synonyms.append(cleaned)
        if len(synonyms) >= 12:
            break
    return synonyms


def _table_reference_maps(
    table_keys: list[str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact: dict[str, str] = {}
    bare: dict[str, list[str]] = {}
    for key in table_keys:
        exact[key.lower()] = key
        bare.setdefault(key.rsplit(".", 1)[-1].lower(), []).append(key)
    return exact, bare


def _resolve_table_reference(
    reference: str,
    exact: dict[str, str],
    bare: dict[str, list[str]],
) -> str | None:
    normalized = ".".join(part.strip().strip('"') for part in reference.strip().split("."))
    if not normalized:
        return None
    matched = exact.get(normalized.lower())
    if matched is not None:
        return matched
    candidates = bare.get(normalized.lower(), [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def sanitize_generated_semantic_layer(
    parsed: SemanticLayerFile,
    table_keys: list[str],
) -> list[SemanticEntry]:
    """Drop generated entries that reference unknown tables or invalid SQL."""

    exact, bare = _table_reference_maps(table_keys)
    entries: list[SemanticEntry] = []
    seen: set[str] = set()

    for metric in parsed.metrics:
        name = _clean_text(metric.name, 200)
        key = f"metric:{name}"
        if not name or key in seen:
            continue
        required_tables: list[str] = []
        missing: str | None = None
        for reference in metric.required_tables:
            resolved = _resolve_table_reference(reference, exact, bare)
            if resolved is None:
                missing = reference
                break
            if resolved not in required_tables:
                required_tables.append(resolved)
        if missing is not None or not required_tables:
            logger.warning(
                "Discarding generated metric %r because it references an unknown table: %s",
                name,
                missing,
            )
            continue
        try:
            formula = validate_semantic_expression(
                metric.sql_formula,
                kind="metric",
                tables=required_tables,
            )
        except GoldenSQLValidationError as exc:
            logger.warning("Discarding generated metric %r: %s", name, exc)
            continue
        seen.add(key)
        entries.append(
            SemanticEntry(
                kind="metric",
                name=name,
                synonyms=_clean_synonyms(metric.synonyms),
                description=_clean_text(metric.description, 1_000),
                required_tables=required_tables,
                sql_formula=formula,
                notes=_clean_text(metric.notes, 1_000) or None,
            )
        )

    for filter_definition in parsed.filters:
        name = _clean_text(filter_definition.name, 200)
        key = f"filter:{name}"
        if not name or key in seen:
            continue
        table = _resolve_table_reference(filter_definition.table, exact, bare)
        if table is None:
            logger.warning(
                "Discarding generated filter %r because it references an unknown table: %s",
                name,
                filter_definition.table,
            )
            continue
        try:
            condition = validate_semantic_expression(
                filter_definition.sql_condition,
                kind="filter",
                tables=[table],
            )
        except GoldenSQLValidationError as exc:
            logger.warning("Discarding generated filter %r: %s", name, exc)
            continue
        seen.add(key)
        entries.append(
            SemanticEntry(
                kind="filter",
                name=name,
                synonyms=_clean_synonyms(filter_definition.synonyms),
                description=_clean_text(filter_definition.description, 1_000),
                required_tables=[table],
                sql_condition=condition,
                table_name=table,
                notes=_clean_text(filter_definition.notes, 1_000) or None,
            )
        )

    return entries
