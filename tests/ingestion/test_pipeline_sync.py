from __future__ import annotations

import uuid

import sqlalchemy as sa

from querity.ingestion.embeddings import embedding_model_id
from querity.ingestion.models import ExistingGoldenState
from querity.ingestion.pipeline import IngestionPipeline
from querity.ingestion.semantic_layer import GoldenEntry


def _entry(key: str, question: str) -> GoldenEntry:
    return GoldenEntry(
        example_key=key,
        question=question,
        sql="SELECT 1",
        tables=[],
        tags=[],
        notes=None,
    )


def test_golden_deactivation_uses_only_valid_entries(settings_factory) -> None:
    settings = settings_factory(enable_llm=False)
    valid = _entry("valid", "Valid question")
    invalid = _entry("invalid", "Invalid question")
    existing = {
        valid.example_key: ExistingGoldenState(
            example_key=valid.example_key,
            content_hash=valid.content_hash,
            embedding_model=embedding_model_id(settings),
            has_embedding=True,
            is_active=True,
        )
    }

    class FakeStore:
        deactivated = None

        def touch_golden_queries(self, source_id, run_id, entries):
            assert entries == [valid]

        def deactivate_missing_golden_queries(self, source_id, entries):
            self.deactivated = entries
            return 0

    store = FakeStore()
    engine = sa.create_engine("sqlite://")
    try:
        with IngestionPipeline(settings, source_engine=engine) as pipeline:
            errors: list[dict[str, str]] = []
            stats: dict[str, object] = {}
            pipeline._sync_golden_queries(
                store,  # type: ignore[arg-type]
                uuid.uuid4(),
                [valid, invalid],
                [valid],
                True,
                existing,
                errors,
                stats,
            )
    finally:
        engine.dispose()

    assert store.deactivated == [valid]
    assert errors == []
    assert stats["golden_unchanged"] == 1


def test_golden_store_failure_is_not_counted_as_embedded(settings_factory) -> None:
    settings = settings_factory(enable_llm=False)
    entry = _entry("pending", "Pending question")

    class FakeStore:
        def upsert_golden_queries(self, source_id, run_id, entries):
            raise RuntimeError("database unavailable")

        def touch_golden_queries(self, source_id, run_id, entries):
            return None

        def deactivate_missing_golden_queries(self, source_id, entries):
            return 0

    class FakeEmbedder:
        def embed_documents_tolerant(self, texts):
            return [[0.1] * 8 for _ in texts], [None] * len(texts)

    engine = sa.create_engine("sqlite://")
    try:
        with IngestionPipeline(settings, source_engine=engine) as pipeline:
            pipeline._embedder = FakeEmbedder()  # type: ignore[assignment]
            errors: list[dict[str, str]] = []
            stats: dict[str, object] = {}
            pipeline._sync_golden_queries(
                FakeStore(),  # type: ignore[arg-type]
                uuid.uuid4(),
                [entry],
                [entry],
                True,
                {},
                errors,
                stats,
            )
    finally:
        engine.dispose()

    assert stats["golden_embedded"] == 0
    assert any(error["scope"] == "golden_sql" and error["stage"] == "store" for error in errors)
