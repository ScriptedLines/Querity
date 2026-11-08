from __future__ import annotations

import stat
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from querity.agent.chat_store import ChatStore
from querity.agent.checkpoint import create_checkpoint_store
from querity.agent.graph import (
    build_graph,
    route_after_analyze,
    route_after_execute,
    route_after_guard,
    route_after_review,
)
from querity.agent.service import AgentService
from querity.agent.settings import AgentSettings
from querity.agent.state import (
    ExecutionResult,
    GuardResult,
    HumanDecision,
    QueryPlan,
    RetrievalBundle,
    ReviewVerdict,
    SQLCandidate,
)


def _settings(**overrides: object) -> AgentSettings:
    values = {
        "_env_file": None,
        "source_database_url": "postgresql://user:pass@localhost:5432/analytics",
        "metadata_database_url": "postgresql://user:pass@localhost:5432/analytics",
        "enable_explain_guard": False,
    }
    values.update(overrides)
    return AgentSettings(**values)


class FakeNodes:
    def ingest_turn(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"status": "running", "recent_context": ""}

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"plan": QueryPlan(intent="count users"), "status": "running"}

    def clarify(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"status": "running"}

    def retrieve(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"retrieval": RetrievalBundle(), "status": "running"}

    def generate(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate": SQLCandidate(sql="SELECT count(*) FROM public.users"),
            "sql": "SELECT count(*) FROM public.users",
            "sql_hash": "hash-1",
            "status": "running",
        }

    def guard(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "guard": GuardResult(
                ok=True,
                sql=state["sql"],
                sql_hash="hash-1",
            ),
            "status": "running",
        }

    def review(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "review": ReviewVerdict(approved=True, summary="looks right"),
            "status": "running",
        }

    def revise(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"status": "running", "revision_count": state.get("revision_count", 0) + 1}

    def human_review(self, state: dict[str, Any]) -> dict[str, Any]:
        decision = interrupt(
            {
                "kind": "approval",
                "thread_id": state["thread_id"],
                "run_id": state["run_id"],
                "question": state["question"],
                "sql": state["sql"],
                "sql_hash": state["sql_hash"],
                "can_approve": True,
            }
        )
        return {
            "human_decision": HumanDecision.model_validate(decision),
            "status": "running",
        }

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "execution": ExecutionResult(
                ok=True,
                columns=["count"],
                rows=[["3"]],
                row_count=1,
            ),
            "status": "running",
        }

    def persist(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"status": "completed"}

    def finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"status": state.get("status", "completed")}


def test_graph_interrupts_for_approval_and_resumes_to_execution():
    settings = _settings()
    graph = build_graph(FakeNodes(), settings, checkpointer=InMemorySaver())  # type: ignore[arg-type]
    config = {"configurable": {"thread_id": "thread-1"}}
    initial = {
        "conversation_id": "conversation-1",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "question": "How many users?",
        "status": "running",
        "feedback": [],
        "attempts": [],
    }
    first = graph.invoke(initial, config)
    snapshot = graph.get_state(config)
    assert snapshot.interrupts
    assert first.get("sql") == "SELECT count(*) FROM public.users"
    payload = snapshot.interrupts[0].value
    assert payload["kind"] == "approval"
    assert payload["sql_hash"] == "hash-1"

    final = graph.invoke(
        Command(resume={"kind": "approve", "sql_hash": "hash-1"}),
        config,
    )
    assert final["status"] == "completed"
    assert final["execution"].row_count == 1
    assert final["human_decision"].kind == "approve"


def test_service_streaming_and_resume(tmp_path):
    settings = _settings()
    graph = build_graph(FakeNodes(), settings, checkpointer=InMemorySaver())  # type: ignore[arg-type]
    service = AgentService.__new__(AgentService)
    service.settings = settings
    service.graph = graph
    service.chat_store = ChatStore(tmp_path / "history.sqlite3")
    service.conversation_id = "conversation-1"
    initial = {
        "conversation_id": "conversation-1",
        "thread_id": "thread-2",
        "run_id": "run-2",
        "question": "How many users?",
        "status": "running",
        "feedback": [],
        "attempts": [],
    }
    events: list[str] = []
    result = service._run(
        initial,
        thread_id="thread-2",
        conversation_id="conversation-1",
        run_id="run-2",
        on_event=lambda label, update: events.append(label),
    )
    assert result.status == "awaiting_approval"
    assert result.interrupt is not None
    assert result.interrupt["sql_hash"] == "hash-1"
    assert "review" in events
    assert "interrupt" in events

    resumed = service._run(
        Command(resume={"kind": "approve", "sql_hash": "hash-1"}),
        thread_id="thread-2",
        conversation_id="conversation-1",
        run_id="run-2",
        on_event=lambda label, update: events.append(label),
    )
    assert resumed.status == "completed"
    assert resumed.execution is not None
    assert resumed.execution.row_count == 1
    service.chat_store.close()


def test_sqlite_checkpoint_survives_process_restart(tmp_path):
    settings = _settings()
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    first_store = create_checkpoint_store(checkpoint_path)
    assert stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o600
    first_graph = build_graph(
        FakeNodes(),  # type: ignore[arg-type]
        settings,
        checkpointer=first_store.saver,
    )
    first = AgentService.__new__(AgentService)
    first.settings = settings
    first.graph = first_graph
    first.chat_store = ChatStore(tmp_path / "history.sqlite3")
    first.chat_store.get_or_create_conversation("conversation-1", "default")
    first.conversation_id = "conversation-1"
    initial = {
        "conversation_id": "conversation-1",
        "thread_id": "thread-restart",
        "run_id": "run-restart",
        "question": "How many users?",
        "status": "running",
        "feedback": [],
        "attempts": [],
    }
    result = first._run(
        initial,
        thread_id="thread-restart",
        conversation_id="conversation-1",
        run_id="run-restart",
        on_event=None,
    )
    assert result.status == "awaiting_approval"
    first.chat_store.set_pending(
        "conversation-1",
        thread_id="thread-restart",
        kind="approval",
    )
    first_store.close()

    second_store = create_checkpoint_store(checkpoint_path)
    second_graph = build_graph(
        FakeNodes(),  # type: ignore[arg-type]
        settings,
        checkpointer=second_store.saver,
    )
    second = AgentService.__new__(AgentService)
    second.settings = settings
    second.graph = second_graph
    second.chat_store = first.chat_store
    second.conversation_id = "conversation-1"
    recovered = second.pending_interrupt()
    assert recovered is not None
    assert recovered.status == "awaiting_approval"
    assert recovered.interrupt is not None
    assert recovered.interrupt["sql_hash"] == "hash-1"

    resumed = second._run(
        Command(resume={"kind": "approve", "sql_hash": "hash-1"}),
        thread_id="thread-restart",
        conversation_id="conversation-1",
        run_id="run-restart",
        on_event=None,
    )
    assert resumed.status == "completed"
    assert resumed.execution is not None
    second.chat_store.close()
    second_store.close()


def test_service_lazy_conversation_lifecycle(tmp_path):
    settings = _settings()
    store = ChatStore(tmp_path / "history.sqlite3")
    service = AgentService.__new__(AgentService)
    service.settings = settings
    service.chat_store = store
    service.conversation = store.get_latest_pending_conversation(settings.db_fingerprint)
    service.conversation_id = (
        str(service.conversation["conversation_id"]) if service.conversation else None
    )

    # Initial state: no conversation created yet
    assert service.conversation_id is None
    assert service.conversation is None
    assert store.list_all_conversations() == []

    # Calling new_conversation does not insert rows
    service.new_conversation(title="Custom Title")
    assert service.conversation_id is None
    assert store.list_all_conversations() == []

    # Calling start_turn lazily creates the conversation with the first question as title
    service.graph = build_graph(
        FakeNodes(),  # type: ignore[arg-type]
        settings,
        checkpointer=InMemorySaver(),
    )
    turn = service.start_turn("How many users are active?")
    assert service.conversation_id is not None
    assert service.conversation is not None
    assert service.conversation["title"] == "How many users are active?"
    assert len(store.list_all_conversations()) == 1

    # On new service startup without any pending interrupt, it opens as blank (None)
    fresh_service = AgentService.__new__(AgentService)
    fresh_service.settings = settings
    fresh_service.chat_store = store
    fresh_service.conversation = store.get_latest_pending_conversation(settings.db_fingerprint)
    fresh_service.conversation_id = (
        str(fresh_service.conversation["conversation_id"]) if fresh_service.conversation else None
    )
    assert fresh_service.conversation_id is None
    assert fresh_service.conversation is None

    # But if that conversation has a pending interrupt, new service startup recovers it
    store.set_pending(service.conversation_id, thread_id="t-pending", kind="clarification")
    recovered_service = AgentService.__new__(AgentService)
    recovered_service.settings = settings
    recovered_service.chat_store = store
    recovered_service.conversation = store.get_latest_pending_conversation(settings.db_fingerprint)
    recovered_service.conversation_id = (
        str(recovered_service.conversation["conversation_id"]) if recovered_service.conversation else None
    )
    assert recovered_service.conversation_id == service.conversation_id

    # Calling new_conversation resets back to unpersisted
    service.new_conversation()
    assert service.conversation_id is None
    assert service.conversation is None
    # Still only 1 conversation in DB
    assert len(store.list_all_conversations()) == 1
    store.close()



def test_routing_helpers():
    settings = _settings()
    assert route_after_analyze({"status": "failed"}, settings) == "finalize"
    assert (
        route_after_analyze(
            {"status": "running", "plan": QueryPlan(intent="x")},
            settings,
        )
        == "retrieve"
    )
    assert (
        route_after_analyze(
            {
                "status": "running",
                "plan": QueryPlan(intent="x", answerable=False),
            },
            settings,
        )
        == "clarify"
    )
    assert (
        route_after_guard(
            {
                "status": "running",
                "guard": GuardResult(ok=True, sql="SELECT 1", sql_hash="h"),
            },
            settings,
        )
        == "review"
    )
    assert (
        route_after_guard(
            {
                "status": "running",
                "guard": GuardResult(ok=False, sql="SELECT 1", sql_hash="h"),
                "revision_count": 0,
            },
            settings,
        )
        == "revise"
    )
    assert (
        route_after_review(
            {
                "status": "running",
                "review": ReviewVerdict(approved=True),
            },
            settings,
        )
        == "human_review"
    )
    assert (
        route_after_execute(
            {
                "status": "running",
                "execution": ExecutionResult(ok=True),
            },
            settings,
        )
        == "persist"
    )
