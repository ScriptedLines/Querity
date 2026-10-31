from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import sqlalchemy as sa

from querity.ingestion.db import quote_ident
from querity.ingestion.join_graph import load_join_graph
from querity.ingestion.store import MetadataStore


class CatalogEmptyError(RuntimeError):
    pass


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_sensitive: bool = False


@dataclass
class TableInfo:
    key: str
    schema_name: str
    table_name: str
    business_description: str
    m_schema: str
    columns: dict[str, ColumnInfo] = field(default_factory=dict)
    join_keys: list[dict[str, Any]] = field(default_factory=list)
    row_count_estimate: int | None = None

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns.values()]


@dataclass
class CatalogSnapshot:
    source_id: str
    tables: dict[str, TableInfo]
    graph: nx.MultiGraph

    def __post_init__(self) -> None:
        self._lower_keys = {key.lower(): key for key in self.tables}
        self._bare_names: dict[str, list[str]] = {}
        for key in self.tables:
            bare = key.rsplit(".", 1)[-1].lower()
            self._bare_names.setdefault(bare, []).append(key)

    @property
    def empty(self) -> bool:
        return not self.tables

    @property
    def table_keys(self) -> list[str]:
        return sorted(self.tables)

    def resolve_table(self, reference: str) -> str | None:
        parts = [part.strip().strip('"') for part in reference.strip().split(".")]
        if not parts or any(not part for part in parts) or len(parts) > 2:
            return None
        normalized = ".".join(parts)
        exact = self._lower_keys.get(normalized.lower())
        if exact is not None:
            return exact
        candidates = self._bare_names.get(normalized.lower(), [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    def table(self, reference: str) -> TableInfo | None:
        key = self.resolve_table(reference)
        return self.tables.get(key) if key else None

    def column(self, table: str, column: str) -> ColumnInfo | None:
        info = self.table(table)
        if info is None:
            return None
        return info.columns.get(column.lower())

    def are_connected(self, left: str, right: str) -> bool:
        if left not in self.graph or right not in self.graph:
            return False
        return nx.has_path(self.graph, left, right)

    def join_edges(self, keys: set[str]) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        if self.graph.number_of_edges() == 0:
            return edges
        for source, target, data in self.graph.edges(data=True):
            if source not in keys or target not in keys:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "condition": data.get("condition"),
                    "kind": data.get("kind"),
                    "confidence": data.get("confidence"),
                }
            )
        return edges


def load_catalog(store: MetadataStore, source_id: str) -> CatalogSnapshot:
    schema_sql = quote_ident(store.engine, store.schema)
    try:
        with store.engine.connect() as connection:
            rows = connection.execute(
                sa.text(
                    f"SELECT schema_name, table_name, business_description, m_schema, columns, "
                    f"join_keys, row_count_estimate "
                    f"FROM {schema_sql}.table_catalog "
                    "WHERE source_id = :source_id AND is_active = true"
                ),
                {"source_id": source_id},
            ).fetchall()
            graph_row = connection.execute(
                sa.text(
                    f"SELECT graph FROM {schema_sql}.schema_graph WHERE source_id = :source_id"
                ),
                {"source_id": source_id},
            ).first()
    except sa.exc.ProgrammingError as exc:
        raise CatalogEmptyError(
            "The metadata catalog does not exist yet. Run ingestion first."
        ) from exc

    tables: dict[str, TableInfo] = {}
    for row in rows:
        key = f"{row.schema_name}.{row.table_name}"
        columns: dict[str, ColumnInfo] = {}
        for item in _json_value(row.columns, []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            columns[name.lower()] = ColumnInfo(
                name=name,
                data_type=str(item.get("data_type") or item.get("type") or ""),
                is_sensitive=bool(item.get("is_sensitive") or item.get("pii")),
            )
        tables[key] = TableInfo(
            key=key,
            schema_name=row.schema_name,
            table_name=row.table_name,
            business_description=row.business_description or "",
            m_schema=row.m_schema or "",
            columns=columns,
            join_keys=_json_value(row.join_keys, []),
            row_count_estimate=row.row_count_estimate,
        )

    graph = nx.MultiGraph()
    if graph_row is not None and graph_row.graph:
        try:
            graph = load_join_graph(_json_value(graph_row.graph, graph_row.graph))
        except Exception:
            graph = nx.MultiGraph()
    for key in tables:
        if key not in graph:
            graph.add_node(key)

    return CatalogSnapshot(source_id=source_id, tables=tables, graph=graph)


class CatalogCache:
    """In-process snapshot cache keyed by the latest ingestion version."""

    def __init__(self, store: MetadataStore, source_id: str) -> None:
        self._store = store
        self._source_id = source_id
        self._lock = threading.RLock()
        self._snapshot: CatalogSnapshot | None = None
        self._version: str | None = None

    def version(self) -> str:
        try:
            return self._store.catalog_version(self._source_id)
        except sa.exc.ProgrammingError:
            return ""

    def get(self, version: str | None = None) -> CatalogSnapshot:
        current = self.version() if version is None else version
        with self._lock:
            if self._snapshot is not None and self._version == current:
                return self._snapshot
            snapshot = load_catalog(self._store, self._source_id)
            self._snapshot = snapshot
            self._version = current
            return snapshot

    def invalidate(self) -> None:
        with self._lock:
            self._snapshot = None
            self._version = None
