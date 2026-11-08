from __future__ import annotations

import sqlite3
import stat

from querity.agent.chat_store import ChatStore


def test_history_database_permissions(tmp_path):
    store = ChatStore(tmp_path / "history.sqlite3")
    try:
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    finally:
        store.close()


def test_one_conversation_per_fingerprint(tmp_path):
    store = ChatStore(tmp_path / "history.sqlite3")
    try:
        first = store.get_or_create_conversation("fingerprint-a", "analytics")
        second = store.get_or_create_conversation("fingerprint-a", "analytics")
        other = store.get_or_create_conversation("fingerprint-b", "analytics")
        assert first["conversation_id"] == second["conversation_id"]
        assert first["conversation_id"] != other["conversation_id"]
    finally:
        store.close()


def test_create_conversation_starts_a_new_saved_chat(tmp_path):
    store = ChatStore(tmp_path / "history.sqlite3")
    try:
        first = store.get_or_create_conversation("fingerprint-a", "analytics")
        second = store.create_conversation("fingerprint-a", "analytics")
        assert second["conversation_id"] != first["conversation_id"]
        assert (
            store.get_or_create_conversation("fingerprint-a", "analytics")["conversation_id"]
            == second["conversation_id"]
        )
    finally:
        store.close()


def test_conversations_store_database_url_and_list_across_databases(tmp_path):
    store = ChatStore(tmp_path / "history.sqlite3")
    try:
        first = store.create_conversation(
            "fingerprint-a",
            "shop",
            database_url="postgresql+psycopg://user:secret@localhost:5432/shop",
        )
        second = store.create_conversation(
            "fingerprint-b",
            "analytics",
            database_url="postgresql+psycopg://user:secret@localhost:5432/analytics",
        )
        assert first["database_url"].endswith("/shop")
        assert second["database_url"].endswith("/analytics")
        conversations = store.list_all_conversations()
        assert {item["conversation_id"] for item in conversations} == {
            first["conversation_id"],
            second["conversation_id"],
        }
        assert {item["database_url"] for item in conversations} == {
            first["database_url"],
            second["database_url"],
        }
    finally:
        store.close()


def test_pending_and_messages(tmp_path):
    store = ChatStore(tmp_path / "history.sqlite3")
    try:
        conversation = store.get_or_create_conversation("fingerprint", "analytics")
        conversation_id = conversation["conversation_id"]
        store.set_pending(conversation_id, thread_id="thread-1", kind="approval")
        assert store.get_pending(conversation_id) == {
            "thread_id": "thread-1",
            "kind": "approval",
        }
        store.clear_pending(conversation_id)
        assert store.get_pending(conversation_id) is None

        message_id = store.add_message(
            conversation_id=conversation_id,
            role="user",
            content="How many customers?",
            thread_id="thread-1",
        )
        assert message_id
        store.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content="There are 42 customers.",
            thread_id="thread-1",
            status="completed",
        )
        messages = store.recent_messages(conversation_id)
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert store.stats(conversation_id)["messages"] == 2
        assert "How many customers?" in store.conversation_context(conversation_id)
    finally:
        store.close()


def test_query_run_and_verified_search(tmp_path):
    store = ChatStore(tmp_path / "history.sqlite3")
    try:
        conversation = store.get_or_create_conversation("fingerprint", "analytics")
        conversation_id = conversation["conversation_id"]
        store.create_query_run(
            run_id="run-1",
            conversation_id=conversation_id,
            question="Total revenue last month",
        )
        store.update_query_run(
            "run-1",
            status="completed",
            sql="SELECT sum(amount) FROM public.orders",
            row_count=1,
        )
        run = store.get_query_run("run-1")
        assert run is not None
        assert run["status"] == "completed"
        assert run["row_count"] == 1
        assert store.recent_runs(conversation_id)[0]["run_id"] == "run-1"

        store.save_verified_turn(
            turn_id="turn-1",
            conversation_id=conversation_id,
            question="Total revenue last month",
            sql="SELECT sum(amount) FROM public.orders",
            embedding=[1.0, 0.0, 0.0],
        )
        hits = store.search_verified_turns(
            conversation_id=conversation_id,
            query="revenue",
            query_embedding=[1.0, 0.0, 0.0],
            limit=5,
        )
        assert hits
        assert hits[0].turn_id == "turn-1"
        assert store.stats(conversation_id)["verified_turns"] == 1
    finally:
        store.close()


def test_get_latest_conversation_and_none_safety(tmp_path):
    store = ChatStore(tmp_path / "history.sqlite3")
    try:
        assert store.get_latest_conversation("nonexistent") is None
        assert store.get_latest_pending_conversation("nonexistent") is None
        assert store.recent_messages(None) == []
        assert store.get_messages(None) == []
        assert store.stats(None) == {"messages": 0, "query_runs": 0, "verified_turns": 0}
        assert store.recent_runs(None) == []
        assert store.get_query_runs(None) == {}
        assert store.get_pending(None) is None
        store.clear_pending(None)

        created = store.create_conversation("fingerprint-a", "analytics", title="First chat")
        assert store.get_latest_pending_conversation("fingerprint-a") is None
        latest = store.get_latest_conversation("fingerprint-a")
        assert latest is not None
        assert latest["conversation_id"] == created["conversation_id"]
        assert latest["title"] == "First chat"

        store.set_pending(created["conversation_id"], thread_id="t1", kind="clarification")
        pending_conv = store.get_latest_pending_conversation("fingerprint-a")
        assert pending_conv is not None
        assert pending_conv["conversation_id"] == created["conversation_id"]

        store.clear_pending(created["conversation_id"])
        assert store.get_latest_pending_conversation("fingerprint-a") is None
    finally:
        store.close()

