from __future__ import annotations

import uuid

import networkx as nx
import sqlalchemy as sa

import querity.ingestion.pipeline as pipeline_module
from querity.ingestion.models import ColumnProfile, RowSample, TableSchema


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
                name="status",
                data_type="text",
                value_kind="string",
                allowed_values=["new", "paid"],
            ),
        ],
        primary_key=["order_id"],
    )


class FakeStore:
    def __init__(self) -> None:
        self.tables: dict[str, object] = {}
        self.run_status: str | None = None
        self.finished = False
        self.saved_graph: str | None = None

    def ensure_schema(self) -> None:
        self.ensured = True

    def start_run(self, source_id: str) -> uuid.UUID:
        run_id = uuid.uuid4()
        self.run_id = run_id
        return run_id

    def get_table_states(self, source_id: str) -> dict[str, object]:
        return {}

    def get_semantic_states(self, source_id: str) -> dict[str, object]:
        return {}

    def get_golden_states(self, source_id: str) -> dict[str, object]:
        return {}

    def upsert_tables(self, source_id, run_id, records) -> None:
        self.tables.update({record.key: record for record in records})

    def touch_tables(self, source_id, run_id, tables) -> None:
        return None

    def deactivate_missing_tables(self, source_id, active_tables) -> int:
        return 0

    def save_join_graph(self, source_id, run_id, graph_json, node_count, edge_count) -> None:
        self.saved_graph = graph_json

    def finish_run(self, run_id, status, stats, error=None) -> None:
        self.finished = True
        self.run_status = status


class FakeEmbedder:
    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def embed_documents_tolerant(self, texts):
        return [[0.1] * self._dimensions for _ in texts], [""] * len(texts)


def test_run_ingests_tables_end_to_end(monkeypatch, settings_factory) -> None:
    settings = settings_factory(enable_llm=False, embedding_dimensions=64)
    table = _table()
    graph = nx.MultiGraph()
    graph.add_node(table.key)

    monkeypatch.setattr(pipeline_module, "introspect_tables", lambda *args, **kwargs: [table])
    monkeypatch.setattr(pipeline_module, "mine_query_log_edges", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline_module, "build_join_graph", lambda *args, **kwargs: graph)
    monkeypatch.setattr(
        pipeline_module,
        "profile_table",
        lambda *args, **kwargs: (table, RowSample()),
    )

    engine = sa.create_engine("sqlite://")
    store = FakeStore()
    try:
        with pipeline_module.IngestionPipeline(
            settings, source_engine=engine, store=store
        ) as pipeline:
            pipeline._embedder = FakeEmbedder(settings.embedding_dimensions)  # type: ignore[assignment]
            summary = pipeline.run()
    finally:
        engine.dispose()

    assert summary.status == "succeeded"
    assert summary.errors == []
    assert summary.run_id is not None
    assert summary.stats["tables_discovered"] == 1
    assert summary.stats["tables_written"] == 1
    assert table.key in store.tables
    assert store.tables[table.key].embedding is not None  # type: ignore[union-attr]
    assert store.saved_graph is not None
    assert store.finished
    assert store.run_status == "succeeded"


def test_run_dry_run_does_not_touch_metadata(monkeypatch, settings_factory) -> None:
    settings = settings_factory(enable_llm=False, embedding_dimensions=64)
    table = _table()
    graph = nx.MultiGraph()
    graph.add_node(table.key)

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
        with pipeline_module.IngestionPipeline(settings, source_engine=engine) as pipeline:
            summary = pipeline.run(pipeline_module.PipelineOptions(dry_run=True))
    finally:
        engine.dispose()

    assert summary.status == "succeeded"
    assert summary.run_id is None
    assert summary.stats["tables_profiled"] == 1
    assert "tables_written" not in summary.stats
