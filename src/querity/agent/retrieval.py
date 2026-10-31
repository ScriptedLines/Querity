from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import sqlalchemy as sa

from querity.agent.catalog import CatalogCache, CatalogEmptyError, CatalogSnapshot
from querity.agent.chat_store import ChatStore
from querity.agent.settings import AgentSettings
from querity.agent.state import (
    GlossaryHit,
    GoldenHit,
    JoinPath,
    QueryPlan,
    RetrievalBundle,
    TableHit,
)
from querity.common.cache import LocalCache, hash_key
from querity.common.logging import get_logger
from querity.ingestion.db import quote_ident
from querity.ingestion.embeddings import EmbeddingClient
from querity.ingestion.join_graph import shortest_table_path
from querity.ingestion.store import MetadataStore, vector_literal

logger = get_logger(__name__)

_RETRIEVAL_NAMESPACE = "retrieval"
_RETRIEVAL_CACHE_VERSION = "retrieval-v1"
_RETRIEVAL_TTL_SECONDS = 7 * 24 * 60 * 60


class Retriever:
    def __init__(
        self,
        settings: AgentSettings,
        store: MetadataStore,
        embedder: EmbeddingClient,
        chat_store: ChatStore,
        *,
        cache: LocalCache | None = None,
        catalog_cache: CatalogCache | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._chat_store = chat_store
        self._cache = cache
        self._catalog_cache = catalog_cache or CatalogCache(store, settings.source_id)

    def retrieve(
        self,
        *,
        question: str,
        plan: QueryPlan,
        conversation_id: str,
        recent_context: str,
        exclude_message_id: str | None = None,
    ) -> RetrievalBundle:
        del exclude_message_id
        warnings: list[str] = []
        search_query = self._search_query(question, plan)
        catalog_version = self._catalog_cache.version()
        catalog = self._catalog_cache.get(catalog_version)
        if catalog.empty:
            raise CatalogEmptyError(
                "No active tables are indexed for this database. Run ingestion first."
            )

        cache_key = self._retrieval_cache_key(search_query, catalog_version)
        cached = self._load_cached_search(cache_key)
        embedding: list[float] | None = None
        if cached is not None:
            tables, glossary, golden, found_tables = cached
            try:
                embedding = self._embedder.embed_query(search_query[:12_000])
            except Exception as exc:
                warnings.append(f"Semantic retrieval is unavailable: {exc}")
        else:
            try:
                embedding = self._embedder.embed_query(search_query[:12_000])
            except Exception as exc:
                warnings.append(f"Semantic retrieval is unavailable: {exc}")

            table_scores = self._search_tables(search_query, embedding, warnings)
            glossary = self._search_glossary(search_query, embedding, warnings)
            golden = self._search_golden(search_query, embedding, warnings)
            selected = self._expand_tables(table_scores, glossary, catalog, warnings)
            tables = []
            for key in selected:
                info = catalog.tables.get(key)
                if info is None:
                    continue
                tables.append(
                    TableHit(
                        key=key,
                        schema_name=info.schema_name,
                        table_name=info.table_name,
                        business_description=info.business_description,
                        m_schema=info.m_schema,
                        score=table_scores.get(key, 0.0),
                    )
                )
            found_tables = bool(table_scores)
            if embedding is not None and not warnings:
                self._store_cached_search(cache_key, tables, glossary, golden, found_tables)

        history = self._chat_store.search_verified_turns(
            conversation_id=conversation_id,
            query=search_query,
            query_embedding=embedding,
            limit=5,
        )
        if not history:
            history = self._chat_store.recent_verified_turns(conversation_id, limit=3)

        join_paths = [
            JoinPath(
                source=str(edge["source"]),
                target=str(edge["target"]),
                condition=str(edge.get("condition") or ""),
                kind=str(edge.get("kind") or ""),
                confidence=float(edge.get("confidence") or 0.0),
            )
            for edge in catalog.join_edges({table.key for table in tables})
        ]
        join_paths.sort(key=lambda path: (-path.confidence, path.source, path.target))

        context_text = self._build_context(
            tables=tables,
            glossary=glossary,
            golden=golden,
            history=history,
            join_paths=join_paths,
            recent_context=recent_context,
        )
        if not found_tables:
            warnings.append("No table was found semantically; using catalog fallback.")
        return RetrievalBundle(
            tables=tables,
            glossary=glossary,
            golden=golden,
            history=history,
            join_paths=join_paths,
            context_text=context_text,
            warnings=warnings,
        )

    @property
    def _catalog(self) -> CatalogSnapshot:
        return self._catalog_cache.get()

    def search_query_text(self, question: str, plan: QueryPlan) -> str:
        return self._search_query(question, plan)

    def _retrieval_cache_key(self, search_query: str, catalog_version: str) -> str:
        return hash_key(
            _RETRIEVAL_CACHE_VERSION,
            getattr(self._settings, "db_fingerprint", ""),
            self._settings.source_id,
            getattr(self._settings, "metadata_schema", ""),
            catalog_version,
            getattr(self._embedder, "model_id", ""),
            self._settings.max_tables,
            search_query,
        )

    def _load_cached_search(
        self, cache_key: str
    ) -> tuple[list[TableHit], list[GlossaryHit], list[GoldenHit], bool] | None:
        if self._cache is None:
            return None
        payload = self._cache.get_json(_RETRIEVAL_NAMESPACE, cache_key)
        if not isinstance(payload, dict):
            return None
        try:
            return (
                [TableHit.model_validate(item) for item in payload.get("tables", [])],
                [GlossaryHit.model_validate(item) for item in payload.get("glossary", [])],
                [GoldenHit.model_validate(item) for item in payload.get("golden", [])],
                bool(payload.get("found_tables", True)),
            )
        except Exception:
            self._cache.delete(_RETRIEVAL_NAMESPACE, cache_key)
            return None

    def _store_cached_search(
        self,
        cache_key: str,
        tables: list[TableHit],
        glossary: list[GlossaryHit],
        golden: list[GoldenHit],
        found_tables: bool,
    ) -> None:
        if self._cache is None:
            return
        self._cache.set_json(
            _RETRIEVAL_NAMESPACE,
            cache_key,
            {
                "tables": [table.model_dump(mode="json") for table in tables],
                "glossary": [entry.model_dump(mode="json") for entry in glossary],
                "golden": [entry.model_dump(mode="json") for entry in golden],
                "found_tables": found_tables,
            },
            ttl=_RETRIEVAL_TTL_SECONDS,
        )

    def _search_query(self, question: str, plan: QueryPlan) -> str:
        pieces = [question, plan.intent, " ".join(plan.search_terms)]
        pieces.extend(plan.metrics)
        pieces.extend(plan.filters)
        pieces.extend(plan.dimensions)
        pieces.extend(plan.tables)
        if plan.time_range:
            pieces.append(plan.time_range)
        return " ".join(piece.strip() for piece in pieces if piece and piece.strip())

    def _search_tables(
        self,
        query: str,
        embedding: list[float] | None,
        warnings: list[str],
    ) -> dict[str, float]:
        rows: list[dict[str, Any]] = []
        if embedding:
            rows.extend(self._vector_search("table_catalog", embedding, 10, warnings))
        rows.extend(self._lexical_search("table_catalog", query, 10, warnings))
        merged: dict[str, float] = {}
        for row in rows:
            key = f"{row['schema_name']}.{row['table_name']}"
            merged[key] = max(merged.get(key, 0.0), float(row.get("score") or 0.0))
        if not merged:
            try:
                with self._store.engine.connect() as connection:
                    fallback = connection.execute(
                        sa.text(
                            f"SELECT schema_name, table_name FROM "
                            f"{quote_ident(connection, self._store.schema)}.table_catalog "
                            "WHERE source_id = :source_id AND is_active = true "
                            "ORDER BY row_count_estimate DESC NULLS LAST LIMIT 10"
                        ),
                        {"source_id": self._settings.source_id},
                    ).fetchall()
                for row in fallback:
                    merged[f"{row.schema_name}.{row.table_name}"] = 0.0
            except Exception as exc:
                warnings.append(f"Catalog fallback search failed: {exc}")
        return merged

    def _search_glossary(
        self,
        query: str,
        embedding: list[float] | None,
        warnings: list[str],
    ) -> list[GlossaryHit]:
        rows: list[dict[str, Any]] = []
        if embedding:
            rows.extend(self._vector_search("semantic_glossary", embedding, 8, warnings))
        rows.extend(self._lexical_search("semantic_glossary", query, 8, warnings))
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = f"{row['kind']}:{row['name']}"
            current = merged.get(key)
            if current is None or float(row.get("score") or 0.0) > float(
                current.get("score") or 0.0
            ):
                merged[key] = row
        return [
            GlossaryHit(
                kind=row["kind"],
                name=row["name"],
                description=row.get("description") or "",
                synonyms=_json_list(row.get("synonyms")),
                required_tables=_json_list(row.get("required_tables")),
                sql_formula=row.get("sql_formula"),
                sql_condition=row.get("sql_condition"),
                table_name=row.get("table_name"),
                notes=row.get("notes"),
                score=float(row.get("score") or 0.0),
            )
            for row in sorted(
                merged.values(),
                key=lambda item: float(item.get("score") or 0.0),
                reverse=True,
            )
        ]

    def _search_golden(
        self,
        query: str,
        embedding: list[float] | None,
        warnings: list[str],
    ) -> list[GoldenHit]:
        rows: list[dict[str, Any]] = []
        if embedding:
            rows.extend(self._vector_search("golden_sql_examples", embedding, 5, warnings))
        rows.extend(self._lexical_search("golden_sql_examples", query, 5, warnings))
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row["example_key"])
            current = merged.get(key)
            if current is None or float(row.get("score") or 0.0) > float(
                current.get("score") or 0.0
            ):
                merged[key] = row
        return [
            GoldenHit(
                example_key=row["example_key"],
                question=row["question"],
                sql_query=row["sql_query"],
                tables_used=_json_list(row.get("tables_used")),
                tags=_json_list(row.get("tags")),
                notes=row.get("notes"),
                score=float(row.get("score") or 0.0),
            )
            for row in sorted(
                merged.values(),
                key=lambda item: float(item.get("score") or 0.0),
                reverse=True,
            )
        ]

    def _vector_search(
        self,
        table: str,
        embedding: list[float],
        limit: int,
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        schema = quote_ident(self._store.engine, self._store.schema)
        table_sql = f"{schema}.{table}"
        vector_type = self._store.vector_cast()
        vector = vector_literal(embedding, self._settings.embedding_dimensions)
        columns = self._columns_for(table)
        try:
            with self._store.engine.connect() as connection:
                result = connection.execute(
                    sa.text(
                        f"SELECT {columns}, "
                        f"1 - (embedding <=> CAST(:vector AS {vector_type})) AS score "
                        f"FROM {table_sql} "
                        "WHERE source_id = :source_id AND is_active = true "
                        "AND embedding IS NOT NULL "
                        f"ORDER BY embedding <=> CAST(:vector AS {vector_type}) "
                        "LIMIT :limit"
                    ),
                    {
                        "vector": vector,
                        "source_id": self._settings.source_id,
                        "limit": limit,
                    },
                ).fetchall()
            return [dict(row._mapping) for row in result]
        except Exception as exc:
            warnings.append(
                f"Vector search over {table} failed ({exc}); "
                "check EMBEDDING_DIMENSIONS against the ingestion config."
            )
            return []

    def _lexical_search(
        self,
        table: str,
        query: str,
        limit: int,
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        schema = quote_ident(self._store.engine, self._store.schema)
        table_sql = f"{schema}.{table}"
        columns = self._columns_for(table)
        try:
            with self._store.engine.connect() as connection:
                result = connection.execute(
                    sa.text(
                        f"SELECT {columns}, "
                        "ts_rank_cd(to_tsvector('english', content), "
                        "websearch_to_tsquery('english', :query)) AS score "
                        f"FROM {table_sql} "
                        "WHERE source_id = :source_id AND is_active = true "
                        "AND to_tsvector('english', content) @@ "
                        "websearch_to_tsquery('english', :query) "
                        "ORDER BY score DESC LIMIT :limit"
                    ),
                    {"query": query, "source_id": self._settings.source_id, "limit": limit},
                ).fetchall()
            if not result:
                return []
            max_score = max(float(row.score or 0.0) for row in result) or 1.0
            return [
                {**dict(row._mapping), "score": float(row.score or 0.0) / max_score}
                for row in result
            ]
        except Exception as exc:
            warnings.append(f"Lexical search over {table} failed: {exc}")
            return []

    @staticmethod
    def _columns_for(table: str) -> str:
        if table == "table_catalog":
            return "schema_name, table_name, business_description, m_schema"
        if table == "semantic_glossary":
            return (
                "kind, name, synonyms, description, required_tables, sql_formula, "
                "sql_condition, table_name, notes"
            )
        return "example_key, question, sql_query, tables_used, tags, notes"

    def _expand_tables(
        self,
        scores: dict[str, float],
        glossary: Iterable[GlossaryHit],
        catalog: CatalogSnapshot,
        warnings: list[str],
    ) -> list[str]:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        selected: list[str] = []
        for key, _ in ordered:
            if key in catalog.tables and key not in selected:
                selected.append(key)
            if len(selected) >= self._settings.max_tables:
                break
        for entry in glossary:
            for reference in entry.required_tables:
                key = catalog.resolve_table(reference)
                if key and key not in selected and len(selected) < self._settings.max_tables:
                    selected.append(key)

        expanded = list(selected)
        cap = self._settings.max_tables + 4
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                path = shortest_table_path(catalog.graph, left, right)
                if not path:
                    continue
                for key in path:
                    if key not in expanded and key in catalog.tables and len(expanded) < cap:
                        expanded.append(key)
        if not expanded:
            warnings.append("The catalog has no usable tables.")
        return expanded

    def _build_context(
        self,
        *,
        tables: list[TableHit],
        glossary: list[GlossaryHit],
        golden: list[GoldenHit],
        history: list[Any],
        join_paths: list[JoinPath],
        recent_context: str,
    ) -> str:
        sections: list[tuple[str, str]] = []
        table_text = "\n\n".join(
            f"### {table.key}\n{table.m_schema or table.business_description}" for table in tables
        )
        if table_text:
            sections.append(("RELEVANT TABLES", table_text))
        if recent_context.strip():
            sections.append(("RECENT CONVERSATION", recent_context.strip()))
        if join_paths:
            sections.append(
                (
                    "JOIN PATHS",
                    "\n".join(
                        f"- {path.source} = {path.target} "
                        f"({path.kind}, confidence {path.confidence:.2f}): {path.condition}"
                        for path in join_paths
                    ),
                )
            )
        if glossary:
            sections.append(
                (
                    "SEMANTIC GLOSSARY",
                    "\n".join(self._render_glossary(entry) for entry in glossary),
                )
            )
        if golden:
            sections.append(
                (
                    "VERIFIED EXAMPLES",
                    "\n\n".join(self._render_golden(entry) for entry in golden),
                )
            )
        if history:
            rendered_history: list[str] = []
            for hit in history:
                rendered_history.append(f"Q: {hit.question}\nSQL: {hit.sql}")
            sections.append(("PREVIOUS VERIFIED TURNS", "\n\n".join(rendered_history)))

        remaining = self._settings.max_context_chars
        output: list[str] = []
        for title, body in sections:
            if remaining <= 0:
                break
            block = f"## {title}\n{body.strip()}\n"
            if len(block) > remaining:
                block = block[:remaining] + "\n[context truncated]\n"
            output.append(block)
            remaining -= len(block)
        return "\n".join(output).strip()

    @staticmethod
    def _render_glossary(entry: GlossaryHit) -> str:
        lines = [f"### {entry.kind}: {entry.name}"]
        if entry.synonyms:
            lines.append(f"Synonyms: {', '.join(entry.synonyms)}")
        if entry.description:
            lines.append(f"Description: {entry.description}")
        if entry.sql_formula:
            lines.append(f"SQL formula: {entry.sql_formula}")
        if entry.sql_condition:
            lines.append(f"SQL condition: {entry.sql_condition}")
        if entry.required_tables:
            lines.append(f"Required tables: {', '.join(entry.required_tables)}")
        if entry.notes:
            lines.append(f"Notes: {entry.notes}")
        return "\n".join(lines)

    @staticmethod
    def _render_golden(entry: GoldenHit) -> str:
        lines = [f"Q: {entry.question}", f"SQL: {entry.sql_query}"]
        if entry.tables_used:
            lines.append(f"Tables: {', '.join(entry.tables_used)}")
        if entry.tags:
            lines.append(f"Tags: {', '.join(entry.tags)}")
        if entry.notes:
            lines.append(f"Notes: {entry.notes}")
        return "\n".join(lines)


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []
