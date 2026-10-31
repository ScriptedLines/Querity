from __future__ import annotations

import fnmatch
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
from sqlalchemy.engine import Engine

from querity.common.logging import get_logger
from querity.ingestion.db import create_source_engine
from querity.ingestion.embeddings import EmbeddingClient, embedding_model_id
from querity.ingestion.enrichment import (
    PROMPT_VERSION,
    Enricher,
    apply_column_descriptions,
    build_embedding_text,
    build_m_schema,
    sanitize_enrichment,
)
from querity.ingestion.hashing import (
    content_hash,
    enrichment_fingerprint,
    profile_fingerprint,
    structural_fingerprint,
)
from querity.ingestion.introspection import introspect_tables
from querity.ingestion.join_graph import (
    build_join_graph,
    graph_to_json,
    join_keys_for_table,
    mine_query_log_edges,
)
from querity.ingestion.models import (
    ExistingTableState,
    RowSample,
    TableCatalogRecord,
    TableEnrichment,
    TableSchema,
)
from querity.ingestion.profiling import profile_table
from querity.ingestion.semantic_generation import SemanticLayerGenerator
from querity.ingestion.semantic_layer import (
    GoldenEntry,
    GoldenSQLValidationError,
    SemanticEntry,
    load_golden_queries,
    load_semantic_layer,
    validate_golden_query_against_source,
    validate_read_only_sql,
)
from querity.ingestion.settings import IngestionSettings
from querity.ingestion.store import MetadataStore

logger = get_logger(__name__)


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineOptions:
    force: bool = False
    table_patterns: list[str] = field(default_factory=list)
    semantic_layer_path: Path | None = None
    generate_semantic_layer: bool | None = None
    golden_queries_path: Path | None = None
    validate_golden_sql: bool | None = None
    dry_run: bool = False
    enable_llm: bool | None = None
    manage_schema: bool = True
    refresh_profiles: bool = False
    deactivate_missing: bool = True


@dataclass
class RunSummary:
    run_id: str | None
    status: str
    stats: dict[str, Any]
    errors: list[dict[str, str]]
    duration_seconds: float


@dataclass
class TableContext:
    table: TableSchema
    sample: RowSample
    structural_hash: str
    profile_hash: str
    enrichment_hash: str
    existing: ExistingTableState | None
    enrichment: TableEnrichment | None = None
    record: TableCatalogRecord | None = None
    needs_embedding: bool = False
    embedded: bool = False
    reused_from_catalog: bool = False
    skipped_unchanged: bool = False


class IngestionPipeline:
    def __init__(
        self,
        settings: IngestionSettings,
        *,
        source_engine: Engine | None = None,
        store: MetadataStore | None = None,
        enricher: Enricher | None = None,
        embedder: EmbeddingClient | None = None,
        semantic_generator: SemanticLayerGenerator | None = None,
    ) -> None:
        self.settings = settings
        self._source_engine = source_engine or create_source_engine(settings)
        self._owns_source = source_engine is None
        self._store = store
        self._owns_store = store is None
        self._store_created_here = False
        self._enricher = enricher
        self._enricher_mode: bool | None = None
        self._embedder = embedder
        self._semantic_generator = semantic_generator
        self._semantic_generator_mode: bool | None = None

    def __enter__(self) -> IngestionPipeline:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_source:
            self._source_engine.dispose()
        if self._owns_store and self._store_created_here and self._store is not None:
            self._store.close()

    def run(self, options: PipelineOptions | None = None) -> RunSummary:
        options = options or PipelineOptions()
        started = time.monotonic()
        source_id = self.settings.source_id
        errors: list[dict[str, str]] = []
        stats: dict[str, Any] = {"source_id": source_id, "dry_run": options.dry_run}
        run_id: uuid.UUID | None = None
        status = "failed"
        store = self._store

        try:
            if not options.dry_run:
                if store is None:
                    store = MetadataStore(self.settings)
                    self._store = store
                    self._store_created_here = True
                if options.manage_schema:
                    store.ensure_schema()
                run_id = store.start_run(source_id)

            tables = introspect_tables(self._source_engine, self.settings)
            tables = self._filter_tables(tables, options.table_patterns)
            if not tables:
                raise PipelineError(
                    "No tables discovered. Check SOURCE_SCHEMAS/TABLE_INCLUDE/TABLE_EXCLUDE "
                    "and the read-only role's privileges."
                )
            stats["tables_discovered"] = len(tables)

            query_log_edges = mine_query_log_edges(self._source_engine, tables, self.settings)
            graph = build_join_graph(tables, self._source_engine, query_log_edges)
            stats["graph_nodes"] = graph.number_of_nodes()
            stats["graph_edges"] = graph.number_of_edges()

            semantic_path = options.semantic_layer_path or self.settings.semantic_layer_path
            golden_path = options.golden_queries_path or self.settings.golden_queries_path
            semantic_provided = semantic_path is not None
            golden_provided = golden_path is not None
            semantic_entries = self._load_semantics(semantic_path)
            golden_entries = self._load_goldens(golden_path)

            validate_explain = (
                self.settings.validate_golden_sql
                if options.validate_golden_sql is None
                else options.validate_golden_sql
            )
            valid_golden_entries = self._validate_golden_queries(
                golden_entries, validate_explain, errors
            )
            stats["semantic_entries"] = len(semantic_entries)
            stats["golden_queries"] = len(golden_entries)
            stats["golden_queries_valid"] = len(valid_golden_entries)

            effective_llm = (
                self.settings.enable_llm if options.enable_llm is None else options.enable_llm
            )
            assert store is not None or options.dry_run
            existing_tables = store.get_table_states(source_id) if store else {}
            contexts = self._profile_tables(
                tables,
                existing_tables,
                errors,
                force=options.force,
                refresh_profiles=options.refresh_profiles or self.settings.refresh_unchanged_tables,
                embedding_model=embedding_model_id(self.settings),
                effective_llm=effective_llm,
            )
            stats["tables_profiled"] = sum(
                1 for context in contexts if not context.skipped_unchanged
            )
            stats["tables_skipped"] = sum(1 for context in contexts if context.skipped_unchanged)

            if options.dry_run:
                status = "partial" if errors else "succeeded"
                stats["tables_failed"] = len(tables) - len(contexts)
                return self._finish_summary(run_id, status, stats, errors, started)

            assert store is not None
            existing_semantics = store.get_semantic_states(source_id)
            existing_golden = store.get_golden_states(source_id)

            self._mark_reusable(contexts, existing_tables, options.force, effective_llm)
            self._enrich_contexts(contexts, graph, effective_llm, errors)
            self._build_records(
                contexts,
                graph,
                embedding_model_id(self.settings),
                errors,
            )

            generate_semantics = (
                self.settings.generate_semantic_layer
                if options.generate_semantic_layer is None
                else options.generate_semantic_layer
            )
            stats["semantic_generated"] = 0
            generation_succeeded = False
            generated_entries: list[SemanticEntry] = []
            if generate_semantics:
                if not effective_llm:
                    logger.warning(
                        "Skipping semantic glossary generation because LLM enrichment is disabled"
                    )
                else:
                    generated_entries, generation_succeeded = self._generate_semantics(
                        contexts, graph, effective_llm, errors
                    )
                    stats["semantic_generated"] = len(generated_entries)
            semantic_entries = self._merge_semantics(semantic_entries, generated_entries)
            if generation_succeeded:
                semantic_provided = True
            stats["semantic_entries"] = len(semantic_entries)

            stats["tables_reused"] = sum(1 for context in contexts if context.reused_from_catalog)
            stats["tables_enriched"] = sum(
                1
                for context in contexts
                if context.record is not None and not context.reused_from_catalog
            )

            table_failures = self._sync_tables(store, run_id, contexts, tables, errors, stats)
            stats["tables_failed"] = table_failures

            if table_failures == 0 and not options.table_patterns and options.deactivate_missing:
                deactivated = store.deactivate_missing_tables(
                    source_id,
                    [(table.schema_name, table.table_name) for table in tables],
                )
                stats["tables_deactivated"] = deactivated
            else:
                stats["tables_deactivated"] = 0
                if options.table_patterns:
                    logger.info("Skipping table deactivation because a --table filter is active")
                else:
                    logger.warning(
                        "Skipping table deactivation because %d table(s) failed",
                        table_failures,
                    )

            self._sync_semantics(
                store,
                run_id,
                semantic_entries,
                semantic_provided,
                existing_semantics,
                errors,
                stats,
            )
            self._sync_golden_queries(
                store,
                run_id,
                golden_entries,
                valid_golden_entries,
                golden_provided,
                existing_golden,
                errors,
                stats,
            )

            store.save_join_graph(
                source_id,
                run_id,
                graph_to_json(graph),
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )

            success_count = (
                len(tables)
                - table_failures
                + int(stats.get("semantic_embedded", 0))
                + int(stats.get("golden_embedded", 0))
            )
            if (errors or table_failures) and success_count == 0:
                status = "failed"
            elif errors or table_failures:
                status = "partial"
            else:
                status = "succeeded"
        except Exception as exc:
            logger.exception("Ingestion run failed")
            errors.append(
                {
                    "scope": "pipeline",
                    "stage": "fatal",
                    "key": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            status = "failed"
        finally:
            stats["errors"] = errors[:100]
            stats["error_count"] = len(errors)
            if store is not None and run_id is not None:
                try:
                    store.finish_run(
                        run_id,
                        status,
                        stats,
                        error=errors[-1]["error"] if errors else None,
                    )
                except Exception:
                    logger.exception("Could not persist ingestion run status")

        return self._finish_summary(run_id, status, stats, errors, started)

    def _finish_summary(
        self,
        run_id: uuid.UUID | None,
        status: str,
        stats: dict[str, Any],
        errors: list[dict[str, str]],
        started: float,
    ) -> RunSummary:
        duration = round(time.monotonic() - started, 2)
        stats["duration_seconds"] = duration
        logger.info(
            "Ingestion finished",
            extra={
                "status": status,
                "run_id": str(run_id) if run_id else None,
                "duration_seconds": duration,
                "error_count": len(errors),
            },
        )
        return RunSummary(
            run_id=str(run_id) if run_id else None,
            status=status,
            stats=stats,
            errors=errors,
            duration_seconds=duration,
        )

    def _filter_tables(self, tables: list[TableSchema], patterns: list[str]) -> list[TableSchema]:
        if not patterns:
            return tables
        return [
            table
            for table in tables
            if any(
                fnmatch.fnmatchcase(table.key, pattern)
                or fnmatch.fnmatchcase(table.table_name, pattern)
                for pattern in patterns
            )
        ]

    def _load_semantics(self, path: Path | None) -> list[SemanticEntry]:
        if path is None:
            return []
        if not path.exists():
            raise PipelineError(f"Semantic layer file does not exist: {path}")
        try:
            return load_semantic_layer(path)
        except Exception as exc:
            raise PipelineError(f"Invalid semantic layer file {path}: {exc}") from exc

    def _load_goldens(self, path: Path | None) -> list[GoldenEntry]:
        if path is None:
            return []
        if not path.exists():
            raise PipelineError(f"Golden query file does not exist: {path}")
        try:
            return load_golden_queries(path)
        except Exception as exc:
            raise PipelineError(f"Invalid golden query file {path}: {exc}") from exc

    def _validate_golden_queries(
        self,
        entries: list[GoldenEntry],
        validate_explain: bool,
        errors: list[dict[str, str]],
    ) -> list[GoldenEntry]:
        valid: list[GoldenEntry] = []
        for entry in entries:
            try:
                validate_read_only_sql(entry.sql)
                if validate_explain:
                    validate_golden_query_against_source(self._source_engine, entry.sql)
                valid.append(entry)
            except GoldenSQLValidationError as exc:
                errors.append(
                    {
                        "scope": "golden_sql",
                        "stage": "validate",
                        "key": entry.question,
                        "error": str(exc),
                    }
                )
                logger.error("Golden query validation failed for %r: %s", entry.question, exc)
        return valid

    def _profile_tables(
        self,
        tables: list[TableSchema],
        existing_tables: dict[str, ExistingTableState],
        errors: list[dict[str, str]],
        *,
        force: bool,
        refresh_profiles: bool,
        embedding_model: str,
        effective_llm: bool,
    ) -> list[TableContext]:
        contexts: list[TableContext] = []
        enrichment_model = (
            f"{self.settings.llm_provider}:{self.settings.llm_model}"
            if effective_llm
            else "heuristic"
        )
        for table in tables:
            structural = structural_fingerprint(table)
            existing = existing_tables.get(table.key)
            expected_enrichment_hash = (
                enrichment_fingerprint(
                    structural,
                    existing.profile_hash,
                    llm_model=enrichment_model,
                    prompt_version=PROMPT_VERSION,
                )
                if existing is not None
                else None
            )
            if (
                not force
                and not refresh_profiles
                and existing is not None
                and existing.structural_hash == structural
                and existing.enrichment_hash == expected_enrichment_hash
                and existing.enrichment
            ):
                try:
                    enrichment = sanitize_enrichment(
                        table, TableEnrichment.model_validate(existing.enrichment)
                    )
                except Exception as exc:
                    logger.warning("Stored enrichment is not reusable for %s: %s", table.key, exc)
                else:
                    reused_content = build_embedding_text(table, enrichment, self.settings)
                    skip_unchanged = bool(
                        existing.content
                        and existing.m_schema
                        and existing.has_embedding
                        and existing.embedding_model == embedding_model
                        and existing.content_hash == content_hash(reused_content)
                    )
                    contexts.append(
                        TableContext(
                            table=table,
                            sample=RowSample(),
                            structural_hash=structural,
                            profile_hash=existing.profile_hash,
                            enrichment_hash=existing.enrichment_hash,
                            existing=existing,
                            enrichment=enrichment,
                            reused_from_catalog=True,
                            skipped_unchanged=skip_unchanged,
                        )
                    )
                    continue

            try:
                profiled, sample = profile_table(self._source_engine, table, self.settings)
            except Exception as exc:
                errors.append(
                    {
                        "scope": "table",
                        "stage": "profile",
                        "key": table.key,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                logger.exception("Profiling failed for %s", table.key)
                continue
            structural = structural_fingerprint(profiled)
            profile = profile_fingerprint(profiled.columns)
            enrichment_hash = enrichment_fingerprint(
                structural,
                profile,
                llm_model=enrichment_model,
                prompt_version=PROMPT_VERSION,
            )
            contexts.append(
                TableContext(
                    table=profiled,
                    sample=sample,
                    structural_hash=structural,
                    profile_hash=profile,
                    enrichment_hash=enrichment_hash,
                    existing=existing,
                )
            )
        return contexts

    def _mark_reusable(
        self,
        contexts: list[TableContext],
        existing_tables: dict[str, ExistingTableState],
        force: bool,
        effective_llm: bool,
    ) -> None:
        enrichment_model = (
            f"{self.settings.llm_provider}:{self.settings.llm_model}"
            if effective_llm
            else "heuristic"
        )
        for context in contexts:
            if context.skipped_unchanged:
                continue
            existing = existing_tables.get(context.table.key)
            context.existing = existing
            hash_with_mode = enrichment_fingerprint(
                context.structural_hash,
                context.profile_hash,
                llm_model=enrichment_model,
                prompt_version=PROMPT_VERSION,
            )
            context.enrichment_hash = hash_with_mode
            if force or existing is None:
                context.enrichment = None
                continue
            if existing.enrichment_hash != hash_with_mode or not existing.enrichment:
                context.enrichment = None
                continue
            try:
                enrichment = TableEnrichment.model_validate(existing.enrichment)
                context.enrichment = sanitize_enrichment(context.table, enrichment)
                context.reused_from_catalog = True
            except Exception as exc:
                logger.warning(
                    "Stored enrichment is not reusable for %s: %s", context.table.key, exc
                )
                context.enrichment = None

    def _enrich_contexts(
        self,
        contexts: list[TableContext],
        graph: nx.MultiGraph,
        effective_llm: bool,
        errors: list[dict[str, str]],
    ) -> None:
        pending = [context for context in contexts if context.enrichment is None]
        if not pending:
            return
        try:
            enricher = self._get_enricher(effective_llm)
        except Exception as exc:
            for context in pending:
                errors.append(
                    {
                        "scope": "table",
                        "stage": "enrich",
                        "key": context.table.key,
                        "error": f"could not initialize LLM: {exc}",
                    }
                )
            logger.exception("Could not initialize the enrichment model")
            return

        join_keys = {
            context.table.key: self._limited_join_keys(graph, context.table.key)
            for context in pending
        }
        if effective_llm and self.settings.llm_concurrency > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=self.settings.llm_concurrency) as executor:
                futures = {
                    executor.submit(
                        enricher.enrich,
                        context.table,
                        context.sample,
                        join_keys[context.table.key],
                    ): context
                    for context in pending
                }
                for future in as_completed(futures):
                    context = futures[future]
                    try:
                        context.enrichment = future.result()
                    except Exception as exc:
                        errors.append(
                            {
                                "scope": "table",
                                "stage": "enrich",
                                "key": context.table.key,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        logger.error("Enrichment failed for %s: %s", context.table.key, exc)
        else:
            for context in pending:
                try:
                    context.enrichment = enricher.enrich(
                        context.table,
                        context.sample,
                        join_keys[context.table.key],
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "scope": "table",
                            "stage": "enrich",
                            "key": context.table.key,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    logger.error("Enrichment failed for %s: %s", context.table.key, exc)

    def _get_enricher(self, effective_llm: bool) -> Enricher:
        if self._enricher is not None and (
            self._enricher_mode is None or self._enricher_mode == effective_llm
        ):
            return self._enricher
        settings = self.settings
        if settings.enable_llm != effective_llm:
            settings = settings.model_copy(update={"enable_llm": effective_llm})
        self._enricher = Enricher(settings)
        self._enricher_mode = effective_llm
        return self._enricher

    def _get_semantic_generator(self, effective_llm: bool) -> SemanticLayerGenerator:
        if self._semantic_generator is not None and (
            self._semantic_generator_mode is None or self._semantic_generator_mode == effective_llm
        ):
            return self._semantic_generator
        settings = self.settings
        if settings.enable_llm != effective_llm:
            settings = settings.model_copy(update={"enable_llm": effective_llm})
        self._semantic_generator = SemanticLayerGenerator(settings)
        self._semantic_generator_mode = effective_llm
        return self._semantic_generator

    def _generate_semantics(
        self,
        contexts: list[TableContext],
        graph: nx.MultiGraph,
        effective_llm: bool,
        errors: list[dict[str, str]],
    ) -> tuple[list[SemanticEntry], bool]:
        enrichments: dict[str, TableEnrichment] = {}
        tables: list[TableSchema] = []
        for context in contexts:
            if context.enrichment is None:
                continue
            enrichments[context.table.key] = context.enrichment
            tables.append(context.table)
        if not tables:
            logger.warning(
                "Skipping semantic glossary generation because no table enrichment is available"
            )
            return [], False
        join_keys = {table.key: self._limited_join_keys(graph, table.key) for table in tables}
        try:
            generator = self._get_semantic_generator(effective_llm)
            if not generator.enabled:
                logger.warning("Semantic glossary generator is disabled")
                return [], False
            entries = generator.generate(tables, enrichments, join_keys)
        except Exception as exc:
            errors.append(
                {
                    "scope": "semantic_layer",
                    "stage": "generate",
                    "key": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            logger.exception("Semantic glossary generation failed")
            return [], False
        if not entries:
            logger.info("LLM did not propose any valid semantic entries; keeping existing glossary")
            return [], False
        return entries, True

    @staticmethod
    def _merge_semantics(
        curated: list[SemanticEntry],
        generated: list[SemanticEntry],
    ) -> list[SemanticEntry]:
        merged = list(curated)
        seen = {entry.key for entry in merged}
        for entry in generated:
            if entry.key in seen:
                logger.info(
                    "Keeping curated semantic entry over generated duplicate: %s",
                    entry.key,
                )
                continue
            seen.add(entry.key)
            merged.append(entry)
        return merged

    def _limited_join_keys(self, graph: nx.MultiGraph, table_key: str) -> list[dict[str, Any]]:
        keys = join_keys_for_table(graph, table_key)
        rank = {"foreign_key": 0, "inferred": 1, "query_log": 2}
        keys.sort(
            key=lambda item: (
                rank.get(str(item.get("kind")), 9),
                -float(item.get("confidence") or 0),
                str(item.get("table")),
            )
        )
        return keys[:25]

    def _build_records(
        self,
        contexts: list[TableContext],
        graph: nx.MultiGraph,
        embedding_model: str,
        errors: list[dict[str, str]],
    ) -> None:
        for context in contexts:
            if context.enrichment is None or context.skipped_unchanged:
                continue
            try:
                join_keys = self._limited_join_keys(graph, context.table.key)
                columns = apply_column_descriptions(context.table, context.enrichment)
                m_schema = build_m_schema(context.table, context.enrichment, join_keys)
                content = build_embedding_text(context.table, context.enrichment, self.settings)
                new_content_hash = content_hash(content)
                record = TableCatalogRecord(
                    source_id=self.settings.source_id,
                    schema_name=context.table.schema_name,
                    table_name=context.table.table_name,
                    table_type=context.table.table_type,
                    business_description=context.enrichment.business_description,
                    m_schema=m_schema,
                    columns=columns,
                    enrichment=context.enrichment.model_dump(mode="json"),
                    sample_questions=context.enrichment.sample_questions,
                    join_keys=join_keys,
                    row_count_estimate=context.table.row_count_estimate,
                    structural_hash=context.structural_hash,
                    profile_hash=context.profile_hash,
                    enrichment_hash=context.enrichment_hash,
                    content=content,
                    content_hash=new_content_hash,
                )
                existing = context.existing
                context.record = record
                context.needs_embedding = (
                    existing is None
                    or existing.content_hash != new_content_hash
                    or not existing.has_embedding
                    or existing.embedding_model != embedding_model
                )
                context.embedded = False
            except Exception as exc:
                errors.append(
                    {
                        "scope": "table",
                        "stage": "artifact",
                        "key": context.table.key,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                logger.exception("Could not build catalog artifact for %s", context.table.key)

    def _sync_tables(
        self,
        store: MetadataStore,
        run_id: uuid.UUID,
        contexts: list[TableContext],
        tables: list[TableSchema],
        errors: list[dict[str, str]],
        stats: dict[str, Any],
    ) -> int:
        to_embed = [
            context
            for context in contexts
            if context.record is not None and context.needs_embedding
        ]
        if to_embed:
            try:
                embedder = self._get_embedder()
                vectors, embedding_errors = embedder.embed_documents_tolerant(
                    [context.record.content for context in to_embed if context.record]
                )
            except Exception as exc:
                vectors = [None] * len(to_embed)
                embedding_errors = [f"{type(exc).__name__}: {exc}"] * len(to_embed)
                logger.exception("Could not initialize the embedding model")
            for context, vector, error in zip(to_embed, vectors, embedding_errors, strict=False):
                if vector is None:
                    errors.append(
                        {
                            "scope": "table",
                            "stage": "embed",
                            "key": context.table.key,
                            "error": error or "embedding failed",
                        }
                    )
                    continue
                assert context.record is not None
                context.record.embedding = vector
                context.record.embedding_model = embedding_model_id(self.settings)
                context.embedded = True
                stats["embeddings_created"] = int(stats.get("embeddings_created", 0)) + 1

        ready = [
            context.record
            for context in contexts
            if context.record is not None and context.embedded
        ]
        written_keys: set[str] = set()
        if ready:
            try:
                store.upsert_tables(self.settings.source_id, run_id, ready)
                written_keys = {record.key for record in ready}
            except Exception as batch_error:
                logger.warning("Batch table upsert failed; retrying individually: %s", batch_error)
                for record in ready:
                    try:
                        store.upsert_tables(self.settings.source_id, run_id, [record])
                        written_keys.add(record.key)
                    except Exception as exc:
                        errors.append(
                            {
                                "scope": "table",
                                "stage": "store",
                                "key": record.key,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        logger.exception("Could not store table %s", record.key)

        touch_targets: list[tuple[str, str]] = []
        for context in contexts:
            if context.table.key in written_keys:
                continue
            if context.existing is not None:
                touch_targets.append((context.table.schema_name, context.table.table_name))
        try:
            store.touch_tables(self.settings.source_id, run_id, touch_targets)
        except Exception as exc:
            logger.exception("Could not refresh last_seen_at for unchanged tables")
            errors.append(
                {
                    "scope": "table",
                    "stage": "touch",
                    "key": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        failed_tables = {
            context.table.key
            for context in contexts
            if context.record is None and not context.skipped_unchanged
        }
        failed_tables.update(
            context.table.key
            for context in contexts
            if context.record is not None and not context.embedded and context.needs_embedding
        )
        failed_tables.update(
            context.table.key
            for context in contexts
            if context.record is not None
            and context.needs_embedding
            and context.embedded
            and context.table.key not in written_keys
        )
        failed_tables.update(
            context.table.key
            for context in contexts
            if context.table.key not in written_keys
            and context.existing is None
            and context.record is not None
        )
        known_keys = {context.table.key for context in contexts}
        for table in tables:
            if table.key not in known_keys:
                failed_tables.add(table.key)
        stats["tables_written"] = len(written_keys)
        stats["tables_embedded"] = sum(1 for context in contexts if context.embedded)
        return len(failed_tables)

    def _get_embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            self._embedder = EmbeddingClient(self.settings)
        return self._embedder

    def _sync_semantics(
        self,
        store: MetadataStore,
        run_id: uuid.UUID,
        entries: list[SemanticEntry],
        provided: bool,
        existing: dict[str, Any],
        errors: list[dict[str, str]],
        stats: dict[str, Any],
    ) -> None:
        if not provided:
            return
        model_id = embedding_model_id(self.settings)
        pending: list[SemanticEntry] = []
        unchanged: list[SemanticEntry] = []
        for entry in entries:
            state = existing.get(entry.key)
            if (
                state is not None
                and state.content_hash == entry.content_hash
                and state.has_embedding
                and state.embedding_model == model_id
            ):
                unchanged.append(entry)
            else:
                pending.append(entry)

        embedded_entries: list[tuple[SemanticEntry, list[float], str]] = []
        if pending:
            try:
                embedder = self._get_embedder()
                vectors, embedding_errors = embedder.embed_documents_tolerant(
                    [entry.embedding_text[: self.settings.embedding_max_chars] for entry in pending]
                )
            except Exception as exc:
                vectors = [None] * len(pending)
                embedding_errors = [f"{type(exc).__name__}: {exc}"] * len(pending)
                logger.exception("Could not initialize the embedding model for semantics")
            for entry, vector, error in zip(pending, vectors, embedding_errors, strict=False):
                if vector is None:
                    errors.append(
                        {
                            "scope": "semantic_layer",
                            "stage": "embed",
                            "key": entry.key,
                            "error": error or "embedding failed",
                        }
                    )
                    if entry.key in existing:
                        unchanged.append(entry)
                    continue
                embedded_entries.append((entry, vector, model_id))

        stored_semantic_keys: set[str] = set()
        if embedded_entries:
            try:
                store.upsert_semantics(self.settings.source_id, run_id, embedded_entries)
                stored_semantic_keys.update(entry.key for entry, _, _ in embedded_entries)
            except Exception as batch_error:
                logger.warning(
                    "Batch semantic upsert failed; retrying individually: %s", batch_error
                )
                for item in embedded_entries:
                    try:
                        store.upsert_semantics(self.settings.source_id, run_id, [item])
                        stored_semantic_keys.add(item[0].key)
                    except Exception as exc:
                        errors.append(
                            {
                                "scope": "semantic_layer",
                                "stage": "store",
                                "key": item[0].key,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        logger.exception("Could not store semantic entry %s", item[0].key)
        try:
            store.touch_semantics(self.settings.source_id, run_id, unchanged)
        except Exception as exc:
            logger.exception("Could not touch unchanged semantic entries")
            errors.append(
                {
                    "scope": "semantic_layer",
                    "stage": "touch",
                    "key": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        try:
            deleted = store.deactivate_missing_semantics(self.settings.source_id, entries)
        except Exception as exc:
            logger.exception("Could not deactivate stale semantic entries")
            errors.append(
                {
                    "scope": "semantic_layer",
                    "stage": "deactivate",
                    "key": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            deleted = 0

        stats["semantic_embedded"] = len(stored_semantic_keys)
        stats["semantic_unchanged"] = len(unchanged)
        stats["semantic_deactivated"] = deleted

    def _sync_golden_queries(
        self,
        store: MetadataStore,
        run_id: uuid.UUID,
        entries: list[GoldenEntry],
        valid_entries: list[GoldenEntry],
        provided: bool,
        existing: dict[str, Any],
        errors: list[dict[str, str]],
        stats: dict[str, Any],
    ) -> None:
        if not provided:
            return
        model_id = embedding_model_id(self.settings)
        pending: list[GoldenEntry] = []
        unchanged: list[GoldenEntry] = []
        for entry in valid_entries:
            state = existing.get(entry.example_key)
            if (
                state is not None
                and state.content_hash == entry.content_hash
                and state.has_embedding
                and state.embedding_model == model_id
            ):
                unchanged.append(entry)
            else:
                pending.append(entry)

        embedded_entries: list[tuple[GoldenEntry, list[float], str]] = []
        if pending:
            try:
                embedder = self._get_embedder()
                vectors, embedding_errors = embedder.embed_documents_tolerant(
                    [entry.embedding_text[: self.settings.embedding_max_chars] for entry in pending]
                )
            except Exception as exc:
                vectors = [None] * len(pending)
                embedding_errors = [f"{type(exc).__name__}: {exc}"] * len(pending)
                logger.exception("Could not initialize the embedding model for golden SQL")
            for entry, vector, error in zip(pending, vectors, embedding_errors, strict=False):
                if vector is None:
                    errors.append(
                        {
                            "scope": "golden_sql",
                            "stage": "embed",
                            "key": entry.question,
                            "error": error or "embedding failed",
                        }
                    )
                    if entry.example_key in existing:
                        unchanged.append(entry)
                    continue
                embedded_entries.append((entry, vector, model_id))

        stored_golden_keys: set[str] = set()
        if embedded_entries:
            try:
                store.upsert_golden_queries(self.settings.source_id, run_id, embedded_entries)
                stored_golden_keys.update(entry.example_key for entry, _, _ in embedded_entries)
            except Exception as batch_error:
                logger.warning("Batch golden upsert failed; retrying individually: %s", batch_error)
                for item in embedded_entries:
                    try:
                        store.upsert_golden_queries(self.settings.source_id, run_id, [item])
                        stored_golden_keys.add(item[0].example_key)
                    except Exception as exc:
                        errors.append(
                            {
                                "scope": "golden_sql",
                                "stage": "store",
                                "key": item[0].question,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        logger.exception("Could not store golden query %r", item[0].question)
        try:
            store.touch_golden_queries(self.settings.source_id, run_id, unchanged)
        except Exception as exc:
            logger.exception("Could not touch unchanged golden queries")
            errors.append(
                {
                    "scope": "golden_sql",
                    "stage": "touch",
                    "key": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        try:
            deleted = store.deactivate_missing_golden_queries(
                self.settings.source_id, valid_entries
            )
        except Exception as exc:
            logger.exception("Could not deactivate stale golden queries")
            errors.append(
                {
                    "scope": "golden_sql",
                    "stage": "deactivate",
                    "key": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            deleted = 0

        stats["golden_embedded"] = len(stored_golden_keys)
        stats["golden_unchanged"] = len(unchanged)
        stats["golden_deactivated"] = deleted


def ingest(
    settings: IngestionSettings | None = None,
    options: PipelineOptions | None = None,
) -> RunSummary:
    """Run one ingestion pass and return its summary.

    This is the programmatic entry point for the ingestion feature. Callers that
    need lifecycle control can instantiate ``IngestionPipeline`` directly.
    """
    with IngestionPipeline(settings if settings is not None else IngestionSettings()) as pipeline:
        return pipeline.run(options)
