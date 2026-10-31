from __future__ import annotations

import uuid

import networkx as nx
import pytest
import sqlalchemy as sa

import querity.ingestion.pipeline as pipeline_module
from querity.ingestion.models import (
    ColumnDescription,
    ColumnProfile,
    RowSample,
    TableEnrichment,
    TableSchema,
)
from querity.ingestion.pipeline import IngestionPipeline, PipelineOptions, TableContext
from querity.ingestion.semantic_generation import SemanticLayerGenerator
from querity.ingestion.semantic_layer import (
    FilterDefinition,
    GoldenSQLValidationError,
    MetricDefinition,
    SemanticEntry,
    SemanticLayerFile,
    sanitize_generated_semantic_layer,
    validate_semantic_expression,
)


def _table() -> TableSchema:
    return TableSchema(
        schema_name="public",
        table_name="orders",
        columns=[
            ColumnProfile(
                name="order_id",
                data_type="bigint",
                value_kind="numeric",
                primary_key=True,
            ),
            ColumnProfile(
                name="amount_usd",
                data_type="numeric",
                value_kind="numeric",
            ),
            ColumnProfile(
                name="status",
                data_type="text",
                value_kind="string",
                allowed_values=["new", "paid", "cancelled"],
            ),
        ],
        primary_key=["order_id"],
    )


def _enrichment() -> TableEnrichment:
    return TableEnrichment(
        business_description="One row per order.",
        grain="one row per order",
        column_descriptions=[
            ColumnDescription(name="order_id", description="Order identifier."),
            ColumnDescription(name="amount_usd", description="Order revenue in USD."),
            ColumnDescription(name="status", description="Order lifecycle status."),
        ],
        sample_questions=["How much revenue did we collect?"],
    )


def _layer() -> SemanticLayerFile:
    return SemanticLayerFile(
        metrics=[
            MetricDefinition(
                name="Order Revenue",
                synonyms=["revenue"],
                description="Total order revenue.",
                required_tables=["public.orders"],
                sql_formula="SUM(amount_usd)",
            )
        ],
        filters=[
            FilterDefinition(
                name="Paid Orders",
                synonyms=["paid"],
                description="Only paid orders.",
                table="public.orders",
                sql_condition="status = 'paid'",
            )
        ],
    )


def test_validate_semantic_expression_accepts_metric_and_filter() -> None:
    assert (
        validate_semantic_expression(
            "SUM(amount_usd) / NULLIF(COUNT(*), 0)",
            kind="metric",
            tables=["public.orders"],
        )
        == "SUM(amount_usd) / NULLIF(COUNT(*), 0)"
    )
    assert (
        validate_semantic_expression(
            "status IN ('paid', 'new')",
            kind="filter",
            tables=["public.orders"],
        )
        == "status IN ('paid', 'new')"
    )


def test_validate_semantic_expression_rejects_multiple_statements() -> None:
    with pytest.raises(GoldenSQLValidationError, match="multiple"):
        validate_semantic_expression(
            "SUM(amount_usd); DROP TABLE public.orders",
            kind="metric",
            tables=["public.orders"],
        )


def test_validate_semantic_expression_rejects_unknown_table() -> None:
    with pytest.raises(GoldenSQLValidationError, match="unknown table"):
        validate_semantic_expression(
            "SUM((SELECT amount FROM secret.orders))",
            kind="metric",
            tables=["public.orders"],
        )


def test_sanitize_generated_semantic_layer_keeps_valid_entries() -> None:
    entries = sanitize_generated_semantic_layer(_layer(), ["public.orders"])
    assert [entry.key for entry in entries] == ["metric:Order Revenue", "filter:Paid Orders"]
    assert entries[0].sql_formula == "SUM(amount_usd)"


def test_sanitize_generated_semantic_layer_resolves_bare_table_names() -> None:
    layer = _layer()
    layer.metrics[0].required_tables = ["orders"]
    layer.filters[0].table = "orders"
    entries = sanitize_generated_semantic_layer(layer, ["public.orders"])
    assert entries[0].required_tables == ["public.orders"]
    assert entries[1].table_name == "public.orders"


def test_sanitize_generated_semantic_layer_drops_unknown_tables() -> None:
    layer = _layer()
    layer.metrics[0].required_tables = ["public.missing"]
    layer.filters[0].table = "public.missing"
    entries = sanitize_generated_semantic_layer(layer, ["public.orders"])
    assert entries == []


class _FakeStructuredLLM:
    def __init__(self, layer: SemanticLayerFile) -> None:
        self._layer = layer
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self._layer


class _FakeChatModel:
    def __init__(self, layer: SemanticLayerFile) -> None:
        self.structured = _FakeStructuredLLM(layer)

    def with_structured_output(self, schema):
        return self.structured


def test_generator_builds_entries_from_catalog(settings_factory) -> None:
    table = _table()
    llm = _FakeChatModel(_layer())
    generator = SemanticLayerGenerator(settings_factory(enable_llm=True), llm=llm)
    entries = generator.generate(
        [table],
        {table.key: _enrichment()},
        {},
    )
    assert [entry.key for entry in entries] == ["metric:Order Revenue", "filter:Paid Orders"]
    assert llm.structured.calls == 1


def test_generator_rejects_payloads_that_only_contain_invalid_entries(settings_factory) -> None:
    table = _table()
    layer = _layer()
    layer.metrics[0].required_tables = ["public.missing"]
    layer.filters = []
    generator = SemanticLayerGenerator(settings_factory(enable_llm=True), llm=_FakeChatModel(layer))
    with pytest.raises(ValueError, match="none passed validation"):
        generator.generate([table], {table.key: _enrichment()}, {})


def test_merge_prefers_curated_entries(settings_factory) -> None:
    curated = SemanticEntry(
        kind="metric",
        name="Order Revenue",
        synonyms=[],
        description="Curated.",
        required_tables=["public.orders"],
        sql_formula="SUM(amount_usd)",
    )
    generated = SemanticEntry(
        kind="metric",
        name="Order Revenue",
        synonyms=[],
        description="Generated.",
        required_tables=["public.orders"],
        sql_formula="SUM(amount_usd)",
    )
    merged = IngestionPipeline._merge_semantics([curated], [generated])
    assert merged == [curated]


class _FakeEnricher:
    def enrich(self, table, row_sample, join_keys):
        return _enrichment()


class _FakeSemanticGenerator:
    enabled = True

    def __init__(self, entries: list[SemanticEntry]) -> None:
        self.entries = entries
        self.calls = 0

    def generate(self, tables, enrichments, join_keys):
        self.calls += 1
        assert tables[0].key == "public.orders"
        assert "public.orders" in enrichments
        return self.entries


class _FakeEmbedder:
    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def embed_documents_tolerant(self, texts):
        return [[0.1] * self._dimensions for _ in texts], [""] * len(texts)


class _FakeStore:
    def __init__(self) -> None:
        self.run_status: str | None = None
        self.semantic_entries: list[SemanticEntry] = []
        self.finished = False

    def ensure_schema(self) -> None:
        return None

    def start_run(self, source_id: str) -> uuid.UUID:
        return uuid.uuid4()

    def get_table_states(self, source_id: str) -> dict[str, object]:
        return {}

    def get_semantic_states(self, source_id: str) -> dict[str, object]:
        return {}

    def get_golden_states(self, source_id: str) -> dict[str, object]:
        return {}

    def upsert_tables(self, source_id, run_id, records) -> None:
        return None

    def touch_tables(self, source_id, run_id, tables) -> None:
        return None

    def deactivate_missing_tables(self, source_id, active_tables) -> int:
        return 0

    def upsert_semantics(self, source_id, run_id, entries) -> None:
        self.semantic_entries = [entry for entry, _, _ in entries]

    def touch_semantics(self, source_id, run_id, entries) -> None:
        return None

    def deactivate_missing_semantics(self, source_id, entries) -> int:
        return 0

    def save_join_graph(self, source_id, run_id, graph_json, node_count, edge_count) -> None:
        return None

    def finish_run(self, run_id, status, stats, error=None) -> None:
        self.finished = True
        self.run_status = status


def test_pipeline_generates_and_stores_semantic_glossary(monkeypatch, settings_factory) -> None:
    table = _table()
    graph = nx.MultiGraph()
    graph.add_node(table.key)
    settings = settings_factory(
        enable_llm=True,
        generate_semantic_layer=True,
        embedding_dimensions=64,
    )
    generated = sanitize_generated_semantic_layer(_layer(), [table.key])
    generator = _FakeSemanticGenerator(generated)
    store = _FakeStore()

    monkeypatch.setattr(pipeline_module, "introspect_tables", lambda *args, **kwargs: [table])
    monkeypatch.setattr(pipeline_module, "mine_query_log_edges", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline_module, "build_join_graph", lambda *args, **kwargs: graph)
    monkeypatch.setattr(
        pipeline_module,
        "profile_table",
        lambda *args, **kwargs: (table, RowSample()),
    )

    engine = sa.create_engine("sqlite://")
    try:
        with IngestionPipeline(
            settings,
            source_engine=engine,
            store=store,
            enricher=_FakeEnricher(),
            embedder=_FakeEmbedder(settings.embedding_dimensions),
            semantic_generator=generator,
        ) as pipeline:
            summary = pipeline.run(
                PipelineOptions(generate_semantic_layer=True),
            )
    finally:
        engine.dispose()

    assert summary.status == "succeeded"
    assert summary.stats["semantic_generated"] == 2
    assert generator.calls == 1
    assert [entry.key for entry in store.semantic_entries] == [
        "metric:Order Revenue",
        "filter:Paid Orders",
    ]
    assert store.run_status == "succeeded"


def test_generate_semantics_returns_failure_without_enrichment(settings_factory) -> None:
    table = _table()
    engine = sa.create_engine("sqlite://")
    errors: list[dict[str, str]] = []
    try:
        with IngestionPipeline(settings_factory(enable_llm=True), source_engine=engine) as pipeline:
            entries, succeeded = pipeline._generate_semantics(
                [
                    TableContext(
                        table=table,
                        sample=RowSample(),
                        structural_hash="",
                        profile_hash="",
                        enrichment_hash="",
                        existing=None,
                        enrichment=None,
                    )
                ],
                nx.MultiGraph(),
                True,
                errors,
            )
    finally:
        engine.dispose()

    assert entries == []
    assert succeeded is False
    assert errors == []
