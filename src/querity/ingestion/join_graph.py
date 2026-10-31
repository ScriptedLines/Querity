from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import networkx as nx
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from querity.common.logging import get_logger
from querity.ingestion.db import quote_ident, readonly_connection
from querity.ingestion.models import ForeignKeyRef, JoinEdge, TableSchema
from querity.ingestion.settings import IngestionSettings

try:
    import inflect

    _INFLECT_ENGINE: inflect.engine | None = inflect.engine()
except ImportError:
    _INFLECT_ENGINE = None

try:
    import sqlglot
    from sqlglot import exp
except ImportError:
    sqlglot = None
    exp = None

logger = get_logger(__name__)

_TABLE_PREFIXES = (
    "dwh_fact_",
    "dwh_dim_",
    "matview_",
    "summary_",
    "bronze_",
    "silver_",
    "lookup_",
    "marts_",
    "table_",
    "core_",
    "fact_",
    "gold_",
    "mart_",
    "link_",
    "xref_",
    "base_",
    "fct_",
    "dim_",
    "stg_",
    "int_",
    "raw_",
    "agg_",
    "ref_",
    "lkp_",
    "rel_",
    "map_",
    "tbl_",
    "app_",
    "vw_",
    "mv_",
    "v_",
    "t_",
)

_INVARIANT_NOUNS = {
    "series": "series",
    "species": "species",
    "movies": "movie",
    "cookies": "cookie",
    "zombies": "zombie",
    "data": "data",
    "media": "media",
    "news": "news",
}


def _strip_prefix(name: str) -> str:
    normalized = name.lower()
    for prefix in _TABLE_PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _singularize(word: str) -> str:
    if _INFLECT_ENGINE is not None:
        try:
            singular = _INFLECT_ENGINE.singular_noun(word)
            if isinstance(singular, str) and singular:
                return singular.lower()
            return word
        except Exception:
            pass

    if word in _INVARIANT_NOUNS:
        return _INVARIANT_NOUNS[word]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _normalize_name(name: str) -> str:
    stripped = _strip_prefix(name)
    return _singularize(stripped)


def _column_ref(engine: Engine, schema: str, table: str, column: str) -> str:
    return (
        f"{quote_ident(engine, schema)}.{quote_ident(engine, table)}.{quote_ident(engine, column)}"
    )


def _fk_condition(engine: Engine, fk: ForeignKeyRef) -> str:
    pairs = [
        f"{_column_ref(engine, fk.constrained_schema, fk.constrained_table, source)} = "
        f"{_column_ref(engine, fk.referred_schema, fk.referred_table, target)}"
        for source, target in zip(fk.constrained_columns, fk.referred_columns, strict=True)
    ]
    return " AND ".join(pairs)


def _edge_key(kind: str, condition: str) -> str:
    return f"{kind}:{condition}"


def _unique_columns(table: TableSchema) -> set[str]:
    unique: set[str] = set(table.primary_key)
    for columns in table.unique_constraints:
        if len(columns) == 1:
            unique.add(columns[0])
    return unique


def _target_column(
    source_column: str,
    source_table: TableSchema,
    target_table: TableSchema,
) -> str | None:
    unique_columns = _unique_columns(target_table)
    if not unique_columns:
        return None

    target_columns = {column.name for column in target_table.columns}
    candidates = [
        source_column,
        f"{_normalize_name(target_table.table_name)}_id",
        "id",
    ]
    for candidate in candidates:
        if candidate in target_columns and candidate in unique_columns:
            return candidate

    if len(target_table.primary_key) == 1 and target_table.primary_key[0] in target_columns:
        return target_table.primary_key[0]
    return None


def infer_join_edges(tables: list[TableSchema], engine: Engine) -> list[JoinEdge]:
    table_by_normalized_name: dict[str, list[TableSchema]] = defaultdict(list)
    for table in tables:
        normalized = _normalize_name(table.table_name)
        table_by_normalized_name[normalized].append(table)
        bare = _strip_prefix(table.table_name)
        if bare != normalized and table not in table_by_normalized_name[bare]:
            table_by_normalized_name[bare].append(table)

    edges: list[JoinEdge] = []
    for source_table in tables:
        already_linked = {
            column for fk in source_table.foreign_keys for column in fk.constrained_columns
        }
        for source_column in source_table.columns:
            lowered = source_column.name.lower()
            if lowered in already_linked or source_column.primary_key:
                continue
            if not lowered.endswith(("_id", "_key")):
                continue
            base = lowered.rsplit("_", 1)[0]
            if not base:
                continue

            candidates = table_by_normalized_name.get(base, [])
            if not candidates:
                continue

            resolved: list[tuple[TableSchema, str]] = []
            for target_table in candidates:
                if target_table.key == source_table.key:
                    continue
                target_column = _target_column(source_column.name, source_table, target_table)
                if target_column:
                    resolved.append((target_table, target_column))

            if len(resolved) != 1:
                if len(resolved) > 1:
                    logger.debug(
                        "Ambiguous inferred join for %s.%s; skipping",
                        source_table.key,
                        source_column.name,
                    )
                continue

            target_table, target_column = resolved[0]
            condition = (
                f"{_column_ref(engine, source_table.schema_name, source_table.table_name, source_column.name)}"
                f" = {_column_ref(engine, target_table.schema_name, target_table.table_name, target_column)}"
            )
            confidence = 0.7
            if source_column.name == target_column:
                confidence = 0.75
            edges.append(
                JoinEdge(
                    source_key=source_table.key,
                    target_key=target_table.key,
                    source_columns=[source_column.name],
                    target_columns=[target_column],
                    condition=condition,
                    kind="inferred",
                    confidence=confidence,
                    evidence="naming convention and unique target column",
                )
            )
    return edges


def _resolve_table_key(
    schema: str | None,
    name: str,
    by_key: set[str],
    by_bare_name: dict[str, list[str]],
) -> str | None:
    if schema:
        candidate = f"{schema}.{name}"
        return candidate if candidate in by_key else None
    matches = by_bare_name.get(name, [])
    if len(matches) == 1:
        return matches[0]
    return None


def _extract_query_log_edges(
    query: str,
    calls: float,
    engine: Engine,
    by_key: set[str],
    by_bare_name: dict[str, list[str]],
) -> list[JoinEdge]:
    if sqlglot is None or exp is None or not query.strip():
        return []
    try:
        statements = sqlglot.parse(query, read="postgres")
    except Exception:
        return []

    edges: list[JoinEdge] = []
    for statement in statements:
        if statement is None or not isinstance(statement, exp.Query):
            continue
        cte_names = {
            cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE) if cte.alias_or_name
        }
        alias_map: dict[str, str] = {}
        for table_expression in statement.find_all(exp.Table):
            if table_expression.db is None and table_expression.name.lower() in cte_names:
                continue
            key = _resolve_table_key(
                table_expression.db or None,
                table_expression.name,
                by_key,
                by_bare_name,
            )
            if key is None:
                continue
            alias_map[table_expression.alias_or_name.lower()] = key
            alias_map[table_expression.name.lower()] = key

        for join in statement.find_all(exp.Join):
            on_clause = join.args.get("on")
            if on_clause is None:
                continue
            for equality in on_clause.find_all(exp.EQ):
                left, right = equality.left, equality.right
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue
                if not left.table or not right.table:
                    continue
                if left.table.lower() in cte_names or right.table.lower() in cte_names:
                    continue
                left_key = alias_map.get(left.table.lower())
                right_key = alias_map.get(right.table.lower())
                if left_key is None or right_key is None or left_key == right_key:
                    continue
                left_schema, left_table = left_key.split(".", 1)
                right_schema, right_table = right_key.split(".", 1)
                condition = (
                    f"{_column_ref(engine, left_schema, left_table, left.name)} = "
                    f"{_column_ref(engine, right_schema, right_table, right.name)}"
                )
                confidence = min(0.9, 0.55 + 0.1 * math.log10(max(calls, 1) + 1))
                edges.append(
                    JoinEdge(
                        source_key=left_key,
                        target_key=right_key,
                        source_columns=[left.name],
                        target_columns=[right.name],
                        condition=condition,
                        kind="query_log",
                        confidence=round(confidence, 3),
                        evidence=f"pg_stat_statements: {int(calls)} calls",
                    )
                )
    return edges


def mine_query_log_edges(
    engine: Engine,
    tables: list[TableSchema],
    settings: IngestionSettings,
) -> list[JoinEdge]:
    if not settings.mine_query_logs:
        return []
    if sqlglot is None:
        logger.warning("sqlglot is not installed; skipping query-log mining")
        return []

    by_key = {table.key for table in tables}
    by_bare_name: dict[str, list[str]] = defaultdict(list)
    for table in tables:
        by_bare_name[table.table_name].append(table.key)

    aggregated: dict[tuple[str, str, str], JoinEdge] = {}
    with readonly_connection(engine) as connection:
        try:
            rows = connection.execute(
                sa.text(
                    "SELECT query, calls FROM pg_stat_statements "
                    "WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database()) "
                    "AND query ILIKE '% join %' "
                    "ORDER BY calls DESC LIMIT :limit"
                ),
                {"limit": settings.query_log_limit},
            ).fetchall()
        except sa.exc.SQLAlchemyError as exc:
            logger.warning("Could not mine pg_stat_statements: %s", exc)
            return []

    for row in rows:
        for edge in _extract_query_log_edges(
            row.query, float(row.calls), engine, by_key, by_bare_name
        ):
            key = (edge.source_key, edge.target_key, edge.condition)
            existing = aggregated.get(key)
            if existing is None:
                aggregated[key] = edge
            else:
                existing.confidence = max(existing.confidence, edge.confidence)

    logger.info("Mined %d query-log join edges", len(aggregated))
    return list(aggregated.values())


def build_join_graph(
    tables: list[TableSchema],
    engine: Engine,
    query_log_edges: list[JoinEdge] | None = None,
) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    table_by_key = {table.key: table for table in tables}
    for table in tables:
        graph.add_node(
            table.key,
            schema=table.schema_name,
            table=table.table_name,
            table_type=table.table_type,
            row_count_estimate=table.row_count_estimate,
        )

    seen_edges: set[tuple[str, str, str, str]] = set()

    def add_edge(edge: JoinEdge) -> None:
        if edge.source_key not in table_by_key or edge.target_key not in table_by_key:
            return
        signature = (
            edge.source_key,
            edge.target_key,
            edge.condition,
            edge.kind,
        )
        reverse_signature = (
            edge.target_key,
            edge.source_key,
            edge.condition,
            edge.kind,
        )
        if signature in seen_edges or reverse_signature in seen_edges:
            return
        seen_edges.add(signature)
        graph.add_edge(
            edge.source_key,
            edge.target_key,
            key=_edge_key(edge.kind, edge.condition),
            condition=edge.condition,
            kind=edge.kind,
            confidence=edge.confidence,
            evidence=edge.evidence,
            source_columns=edge.source_columns,
            target_columns=edge.target_columns,
        )

    for table in tables:
        for fk in table.foreign_keys:
            if fk.referred_key not in table_by_key:
                logger.debug(
                    "Skipping FK %s -> %s because target is outside the indexed catalog",
                    table.key,
                    fk.referred_key,
                )
                continue
            add_edge(
                JoinEdge(
                    source_key=fk.constrained_key,
                    target_key=fk.referred_key,
                    source_columns=fk.constrained_columns,
                    target_columns=fk.referred_columns,
                    condition=_fk_condition(engine, fk),
                    kind="foreign_key",
                    confidence=1.0,
                    evidence=fk.name or "declared foreign key",
                )
            )

    for edge in infer_join_edges(tables, engine):
        add_edge(edge)

    for edge in query_log_edges or []:
        add_edge(edge)

    logger.info(
        "Built join graph",
        extra={
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
    )
    return graph


def graph_to_json(graph: nx.MultiGraph) -> str:
    data = nx.node_link_data(graph, edges="edges", nodes="nodes")
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def load_join_graph(data: str | dict[str, Any]) -> nx.MultiGraph:
    payload = json.loads(data) if isinstance(data, str) else data
    return nx.node_link_graph(
        payload,
        directed=False,
        multigraph=True,
        edges="edges",
        nodes="nodes",
    )


def shortest_table_path(graph: nx.MultiGraph, source_key: str, target_key: str) -> list[str] | None:
    if source_key not in graph or target_key not in graph:
        return None
    try:
        return list(nx.shortest_path(graph, source_key, target_key))
    except nx.NetworkXNoPath:
        return None


def join_keys_for_table(graph: nx.MultiGraph, table_key: str) -> list[dict[str, Any]]:
    if table_key not in graph:
        return []
    join_keys: list[dict[str, Any]] = []
    for source, target, data in graph.edges(table_key, data=True):
        other = target if source == table_key else source
        join_keys.append(
            {
                "table": other,
                "condition": data.get("condition"),
                "kind": data.get("kind"),
                "confidence": data.get("confidence"),
            }
        )
    join_keys.sort(key=lambda item: (str(item.get("table")), str(item.get("condition"))))
    return join_keys
