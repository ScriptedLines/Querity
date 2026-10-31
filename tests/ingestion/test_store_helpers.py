from __future__ import annotations

from types import SimpleNamespace

import pytest
import sqlalchemy as sa

import querity.ingestion.embeddings as embeddings_module
from querity.ingestion.embeddings import EmbeddingClient, embedding_model_id
from querity.ingestion.store import MetadataStore, vector_literal


def test_vector_literal() -> None:
    rendered = vector_literal([0.1, 1.0, -0.5], 3)
    assert rendered == "[0.1,1,-0.5]"


def test_vector_literal_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        vector_literal([0.1, 0.2], 3)


def test_vector_literal_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        vector_literal([0.1, float("nan")], 2)


def test_embedding_model_id_includes_dimensions(settings_factory) -> None:
    first = settings_factory(embedding_dimensions=768)
    second = settings_factory(embedding_dimensions=1_536)
    assert embedding_model_id(first) != embedding_model_id(second)
    assert embedding_model_id(first).endswith(":768")


def test_vector_cast_detects_pgvector_without_ensure_schema(settings_factory) -> None:
    dialect = sa.create_engine("sqlite://").dialect

    class FakeResult:
        def __init__(self, value):
            self.value = value

        def scalar_one_or_none(self):
            return self.value

    class FakeConnection:
        def execute(self, statement, params=None):
            if "pg_extension" in str(statement):
                return FakeResult("extensions")
            return FakeResult(None)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeEngine:
        pass

    connection = FakeConnection()
    connection.dialect = dialect
    engine = FakeEngine()
    engine.connect = lambda: connection

    store = MetadataStore(settings_factory(), engine=engine)
    assert store._vector_cast() == "extensions.vector"


def test_embed_query_uses_the_query_endpoint(monkeypatch, settings_factory) -> None:
    class FakeClient:
        def embed_query(self, text):
            return [0.1] * 64

        def embed_documents(self, texts):
            raise AssertionError("queries must not use the document endpoint")

    monkeypatch.setattr(embeddings_module, "_build_embedding_client", lambda settings: FakeClient())
    client = EmbeddingClient(settings_factory(embedding_dimensions=64))
    assert client.embed_query("question") == [0.1] * 64


def test_embedding_dimension_mismatch_does_not_recurse(monkeypatch, settings_factory) -> None:
    calls = 0

    class FakeClient:
        def embed_documents(self, texts):
            nonlocal calls
            calls += 1
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(embeddings_module, "_build_embedding_client", lambda settings: FakeClient())
    settings = settings_factory(embedding_dimensions=64)
    client = EmbeddingClient(settings)
    vectors, errors = client.embed_documents_tolerant(["a", "b", "c", "d"])
    assert vectors == [None, None, None, None]
    assert len(errors) == 4
    assert calls == 1


def test_catalog_version_uses_join_graph_run(settings_factory) -> None:
    from datetime import UTC, datetime

    test_dialect = sa.create_engine("sqlite://").dialect

    class FakeResult:
        def first(self):
            return SimpleNamespace(
                last_run_id="run-1",
                updated_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
            )

    class FakeConnection:
        dialect = test_dialect

        def execute(self, statement, params=None):
            assert "schema_graph" in str(statement)
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeEngine:
        pass

    engine = FakeEngine()
    engine.connect = lambda: FakeConnection()
    store = MetadataStore(settings_factory(), engine=engine)
    assert store.catalog_version("default") == "run-1:2026-01-02T03:04:05+00:00"


def test_catalog_version_is_empty_without_graph(settings_factory) -> None:
    test_dialect = sa.create_engine("sqlite://").dialect

    class FakeResult:
        def first(self):
            return None

    class FakeConnection:
        dialect = test_dialect

        def execute(self, statement, params=None):
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeEngine:
        pass

    engine = FakeEngine()
    engine.connect = lambda: FakeConnection()
    store = MetadataStore(settings_factory(), engine=engine)
    assert store.catalog_version("default") == ""
