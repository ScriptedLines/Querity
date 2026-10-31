from __future__ import annotations

from typing import Any

import networkx as nx

from querity.agent.catalog import CatalogSnapshot, ColumnInfo, TableInfo
from querity.agent.retrieval import Retriever
from querity.agent.settings import AgentSettings
from querity.agent.state import QueryPlan
from querity.common.cache import LocalCache


class FakeEmbedder:
    model_id = "fake:model:64"

    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1] * 64


class FakeCatalogCache:
    def __init__(self, snapshot: CatalogSnapshot, version: str = "run-1") -> None:
        self.snapshot = snapshot
        self.current_version = version

    def version(self) -> str:
        return self.current_version

    def get(self, version: str | None = None) -> CatalogSnapshot:
        return self.snapshot


class FakeChatStore:
    def search_verified_turns(self, **_kwargs: Any) -> list[Any]:
        return []

    def recent_verified_turns(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []


def _settings() -> AgentSettings:
    return AgentSettings(
        _env_file=None,
        source_database_url="postgresql://user:pass@localhost:5432/analytics",
        metadata_database_url="postgresql://user:pass@localhost:5432/analytics",
        embedding_dimensions=64,
    )


def _snapshot() -> CatalogSnapshot:
    users = TableInfo(
        key="public.users",
        schema_name="public",
        table_name="users",
        business_description="Users",
        m_schema="users(id)",
        columns={"id": ColumnInfo("id", "integer")},
    )
    return CatalogSnapshot(
        source_id="default",
        tables={users.key: users},
        graph=nx.MultiGraph(),
    )


def test_retrieval_search_cache_reuses_results_until_catalog_version_changes(tmp_path):
    calls = {"tables": 0, "glossary": 0, "golden": 0}
    embedder = FakeEmbedder()
    catalog_cache = FakeCatalogCache(_snapshot())
    cache = LocalCache(tmp_path / "cache.sqlite3")
    retriever = Retriever(
        _settings(),
        store=None,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        chat_store=FakeChatStore(),  # type: ignore[arg-type]
        cache=cache,
        catalog_cache=catalog_cache,  # type: ignore[arg-type]
    )

    def search_tables(_query, _embedding, _warnings):
        calls["tables"] += 1
        return {"public.users": 0.9}

    def search_glossary(_query, _embedding, _warnings):
        calls["glossary"] += 1
        return []

    def search_golden(_query, _embedding, _warnings):
        calls["golden"] += 1
        return []

    retriever._search_tables = search_tables  # type: ignore[method-assign]
    retriever._search_glossary = search_glossary  # type: ignore[method-assign]
    retriever._search_golden = search_golden  # type: ignore[method-assign]

    plan = QueryPlan(intent="count users")
    first = retriever.retrieve(
        question="How many users?",
        plan=plan,
        conversation_id="conversation-1",
        recent_context="",
    )
    assert [table.key for table in first.tables] == ["public.users"]
    assert calls == {"tables": 1, "glossary": 1, "golden": 1}
    assert cache.stats()["entries"] == 1

    second = retriever.retrieve(
        question="How many users?",
        plan=plan,
        conversation_id="conversation-1",
        recent_context="",
    )
    assert [table.key for table in second.tables] == ["public.users"]
    assert calls == {"tables": 1, "glossary": 1, "golden": 1}
    assert cache.stats()["entries"] == 1

    catalog_cache.current_version = "run-2"
    third = retriever.retrieve(
        question="How many users?",
        plan=plan,
        conversation_id="conversation-1",
        recent_context="",
    )
    assert [table.key for table in third.tables] == ["public.users"]
    assert calls == {"tables": 2, "glossary": 2, "golden": 2}
    assert cache.stats()["entries"] == 2
