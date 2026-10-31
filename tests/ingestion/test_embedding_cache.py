from __future__ import annotations

import querity.ingestion.embeddings as embeddings_module
from querity.common.cache import LocalCache
from querity.ingestion.embeddings import EmbeddingClient


def test_embedding_cache_reuses_query_and_document_vectors(
    monkeypatch, settings_factory, tmp_path
) -> None:
    class FakeClient:
        def __init__(self, dimensions: int) -> None:
            self.dimensions = dimensions
            self.query_calls: list[str] = []
            self.document_calls: list[list[str]] = []

        def embed_query(self, text: str) -> list[float]:
            self.query_calls.append(text)
            return [0.1] * self.dimensions

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.document_calls.append(list(texts))
            return [[float(len(text))] * self.dimensions for text in texts]

    fake = FakeClient(64)
    monkeypatch.setattr(embeddings_module, "_build_embedding_client", lambda settings: fake)
    cache = LocalCache(tmp_path / "cache.sqlite3")
    client = EmbeddingClient(
        settings_factory(embedding_dimensions=64, embedding_batch_size=16),
        cache=cache,
    )

    first = client.embed_documents(["alpha", "beta", "alpha"])
    assert len(first) == 3
    assert first[0] == first[2]
    assert fake.document_calls == [["alpha", "beta"]]

    second = client.embed_documents(["alpha", "beta", "alpha"])
    assert second == first
    assert fake.document_calls == [["alpha", "beta"]]

    assert client.embed_query("alpha") == [0.1] * 64
    assert client.embed_query("alpha") == [0.1] * 64
    assert fake.query_calls == ["alpha"]


def test_embedding_cache_distinguishes_query_and_document_tasks(
    monkeypatch, settings_factory, tmp_path
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.query_calls = 0
            self.document_calls = 0

        def embed_query(self, text: str) -> list[float]:
            self.query_calls += 1
            return [0.1] * 64

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.document_calls += 1
            return [[0.2] * 64 for _ in texts]

    fake = FakeClient()
    monkeypatch.setattr(embeddings_module, "_build_embedding_client", lambda settings: fake)
    cache = LocalCache(tmp_path / "cache.sqlite3")
    client = EmbeddingClient(settings_factory(embedding_dimensions=64), cache=cache)

    assert client.embed_query("same text") == [0.1] * 64
    assert client.embed_documents(["same text"]) == [[0.2] * 64]
    assert fake.query_calls == 1
    assert fake.document_calls == 1
