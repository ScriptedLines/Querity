from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import networkx as nx

from querity.agent.catalog import CatalogSnapshot, ColumnInfo, TableInfo
from querity.agent.chat_store import ChatStore
from querity.agent.nodes import AgentNodes
from querity.agent.settings import AgentSettings
from querity.agent.state import GuardResult, HumanDecision


class FakeResult:
    def __init__(self, columns: list[str], rows: list[tuple[Any, ...]]) -> None:
        self._columns = columns
        self._rows = rows

    def keys(self) -> list[str]:
        return self._columns

    def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        return self._rows[:size]


class FakeConnection:
    def __init__(self, result: FakeResult) -> None:
        self._result = result

    def execute(self, statement: Any) -> FakeResult:
        return self._result


def _nodes(tmp_path, monkeypatch, result: FakeResult) -> AgentNodes:
    settings = AgentSettings(
        _env_file=None,
        source_database_url="postgresql://user:pass@localhost:5432/analytics",
        metadata_database_url="postgresql://user:pass@localhost:5432/analytics",
        enable_explain_guard=False,
    )
    source = SimpleNamespace(connect=lambda: None)
    nodes = AgentNodes(
        settings=settings,
        store=SimpleNamespace(),  # type: ignore[arg-type]
        chat_store=ChatStore(tmp_path / "history.sqlite3"),
        source_engine=source,  # type: ignore[arg-type]
        embedder=SimpleNamespace(),
        models=SimpleNamespace(),  # type: ignore[arg-type]
        retriever=SimpleNamespace(),  # type: ignore[arg-type]
    )
    connection = FakeConnection(result)

    @contextmanager
    def fake_readonly(_engine: Any):
        yield connection

    monkeypatch.setattr("querity.agent.nodes.readonly_connection", fake_readonly)
    users = TableInfo(
        key="public.users",
        schema_name="public",
        table_name="users",
        business_description="Users",
        m_schema="users(id, email)",
        columns={
            "id": ColumnInfo("id", "integer"),
            "email": ColumnInfo("email", "text", is_sensitive=True),
        },
    )
    catalog = CatalogSnapshot(
        source_id="default",
        tables={users.key: users},
        graph=nx.MultiGraph(),
    )
    nodes.catalog = lambda: catalog  # type: ignore[method-assign]
    return nodes


def test_execute_uses_result_column_names_and_masks_catalog_sensitive_columns(
    tmp_path, monkeypatch
):
    nodes = _nodes(
        tmp_path,
        monkeypatch,
        FakeResult(["id", "email"], [(1, "person@example.com")]),
    )
    guard = GuardResult(
        ok=True,
        sql="SELECT id, email FROM public.users",
        sql_hash="hash-1",
        tables_used=["public.users"],
    )
    state = {
        "conversation_id": "conversation-1",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "question": "Show users",
        "sql": guard.sql,
        "sql_hash": guard.sql_hash,
        "guard": guard,
        "human_decision": HumanDecision(kind="approve", sql_hash=guard.sql_hash),
        "status": "running",
    }
    nodes.chat_store.create_query_run(
        run_id="run-1",
        conversation_id="conversation-1",
        question="Show users",
    )

    update = nodes.execute(state)  # type: ignore[arg-type]
    execution = update["execution"]
    assert execution.ok
    assert execution.columns == ["id", "email"]
    assert execution.rows == [["1", "***@example.com"]]
