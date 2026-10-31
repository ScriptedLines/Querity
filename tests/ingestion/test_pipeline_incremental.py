from __future__ import annotations

import sqlalchemy as sa

import querity.ingestion.pipeline as pipeline_module
from querity.ingestion.embeddings import embedding_model_id
from querity.ingestion.enrichment import PROMPT_VERSION, build_embedding_text
from querity.ingestion.hashing import (
    content_hash,
    enrichment_fingerprint,
    profile_fingerprint,
    structural_fingerprint,
)
from querity.ingestion.models import (
    ColumnDescription,
    ColumnProfile,
    ExistingTableState,
    TableEnrichment,
    TableSchema,
)
from querity.ingestion.pipeline import IngestionPipeline


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


def _enrichment() -> TableEnrichment:
    return TableEnrichment(
        business_description="One row per order.",
        grain="one row per order",
        column_descriptions=[
            ColumnDescription(name="order_id", description="Order identifier."),
            ColumnDescription(name="status", description="Order status."),
        ],
        sample_questions=["How many orders are paid?"],
    )


def _existing(
    settings,
    table: TableSchema,
    *,
    mode: str,
) -> ExistingTableState:
    structural = structural_fingerprint(table)
    profile = profile_fingerprint(table.columns)
    content = build_embedding_text(table, _enrichment(), settings)
    return ExistingTableState(
        schema_name=table.schema_name,
        table_name=table.table_name,
        business_description="One row per order.",
        m_schema="table_name: public.orders",
        columns=[column.model_dump(mode="json") for column in table.columns],
        enrichment=_enrichment().model_dump(mode="json"),
        sample_questions=["How many orders are paid?"],
        content=content,
        content_hash=content_hash(content),
        structural_hash=structural,
        profile_hash=profile,
        enrichment_hash=enrichment_fingerprint(
            structural,
            profile,
            llm_model=mode,
            prompt_version=PROMPT_VERSION,
        ),
        embedding_model=embedding_model_id(settings),
        has_embedding=True,
    )


def test_unchanged_table_is_skipped_without_profiling(monkeypatch, settings_factory) -> None:
    settings = settings_factory(enable_llm=True)
    table = _table()
    state = _existing(settings, table, mode=f"{settings.llm_provider}:{settings.llm_model}")
    engine = sa.create_engine("sqlite://")
    monkeypatch.setattr(
        pipeline_module,
        "profile_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("profiled")),
    )

    try:
        with IngestionPipeline(settings, source_engine=engine) as pipeline:
            contexts = pipeline._profile_tables(
                [table],
                {table.key: state},
                [],
                force=False,
                refresh_profiles=False,
                embedding_model=embedding_model_id(settings),
                effective_llm=True,
            )
    finally:
        engine.dispose()

    assert len(contexts) == 1
    context = contexts[0]
    assert context.skipped_unchanged
    assert context.reused_from_catalog
    assert context.record is None
    assert context.enrichment is not None


def test_content_change_reuses_enrichment_without_profiling(monkeypatch, settings_factory) -> None:
    settings = settings_factory(enable_llm=True)
    table = _table()
    state = _existing(settings, table, mode=f"{settings.llm_provider}:{settings.llm_model}")
    state.content_hash = content_hash("stale content")
    engine = sa.create_engine("sqlite://")
    monkeypatch.setattr(
        pipeline_module,
        "profile_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("profiled")),
    )

    try:
        with IngestionPipeline(settings, source_engine=engine) as pipeline:
            contexts = pipeline._profile_tables(
                [table],
                {table.key: state},
                [],
                force=False,
                refresh_profiles=False,
                embedding_model=embedding_model_id(settings),
                effective_llm=True,
            )
    finally:
        engine.dispose()

    assert len(contexts) == 1
    context = contexts[0]
    assert not context.skipped_unchanged
    assert context.reused_from_catalog
    assert context.enrichment is not None


def test_llm_mode_change_reprofiles_unchanged_table(monkeypatch, settings_factory) -> None:
    settings = settings_factory(enable_llm=True)
    table = _table()
    state = _existing(settings, table, mode="heuristic")
    calls = 0

    def fake_profile(*args, **kwargs):
        nonlocal calls
        calls += 1
        from querity.ingestion.models import RowSample

        return table, RowSample()

    engine = sa.create_engine("sqlite://")
    monkeypatch.setattr(pipeline_module, "profile_table", fake_profile)

    try:
        with IngestionPipeline(settings, source_engine=engine) as pipeline:
            contexts = pipeline._profile_tables(
                [table],
                {table.key: state},
                [],
                force=False,
                refresh_profiles=False,
                embedding_model=embedding_model_id(settings),
                effective_llm=True,
            )
    finally:
        engine.dispose()

    assert calls == 1
    assert len(contexts) == 1
    assert not contexts[0].skipped_unchanged


def test_force_reprofiles_unchanged_table(monkeypatch, settings_factory) -> None:
    settings = settings_factory(enable_llm=True)
    table = _table()
    state = _existing(settings, table, mode=f"{settings.llm_provider}:{settings.llm_model}")
    calls = 0

    def fake_profile(*args, **kwargs):
        nonlocal calls
        calls += 1
        from querity.ingestion.models import RowSample

        return table, RowSample()

    engine = sa.create_engine("sqlite://")
    monkeypatch.setattr(pipeline_module, "profile_table", fake_profile)

    try:
        with IngestionPipeline(settings, source_engine=engine) as pipeline:
            contexts = pipeline._profile_tables(
                [table],
                {table.key: state},
                [],
                force=True,
                refresh_profiles=False,
                embedding_model=embedding_model_id(settings),
                effective_llm=True,
            )
    finally:
        engine.dispose()

    assert calls == 1
    assert len(contexts) == 1
    assert not contexts[0].skipped_unchanged


def test_enricher_cache_respects_llm_mode(monkeypatch, settings_factory) -> None:
    settings = settings_factory(enable_llm=True)
    engine = sa.create_engine("sqlite://")
    created: list[bool] = []

    class FakeEnricher:
        def __init__(self, settings) -> None:
            created.append(settings.enable_llm)

    monkeypatch.setattr(pipeline_module, "Enricher", FakeEnricher)
    try:
        with IngestionPipeline(settings, source_engine=engine) as pipeline:
            first = pipeline._get_enricher(False)
            again = pipeline._get_enricher(False)
            second = pipeline._get_enricher(True)
    finally:
        engine.dispose()

    assert first is again
    assert first is not second
    assert created == [False, True]
