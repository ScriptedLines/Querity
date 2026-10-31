from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

from querity.agent.catalog import CatalogCache
from querity.agent.chat_store import ChatStore
from querity.agent.checkpoint import CheckpointStore, create_checkpoint_store
from querity.agent.graph import build_graph
from querity.agent.llm import AgentModels
from querity.agent.nodes import AgentNodes
from querity.agent.retrieval import Retriever
from querity.agent.settings import AgentSettings
from querity.agent.state import AgentState, HumanDecision
from querity.common.cache import LocalCache, build_local_cache
from querity.common.logging import get_logger
from querity.ingestion.db import create_agent_engine
from querity.ingestion.embeddings import EmbeddingClient
from querity.ingestion.store import MetadataStore

logger = get_logger(__name__)

EventCallback = Callable[[str, Any], None]


@dataclass
class AgentTurnResult:
    thread_id: str
    conversation_id: str
    status: str
    state: dict[str, Any]
    interrupt: dict[str, Any] | None = None
    error: str | None = None
    events: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str | None:
        value = self.state.get("run_id")
        return str(value) if value else None

    @property
    def question(self) -> str:
        return str(self.state.get("question") or "")

    @property
    def sql(self) -> str | None:
        value = self.state.get("sql")
        return str(value) if value else None

    @property
    def execution(self) -> Any:
        return self.state.get("execution")

    @property
    def plan(self) -> Any:
        return self.state.get("plan")

    @property
    def review(self) -> Any:
        return self.state.get("review")

    @property
    def guard(self) -> Any:
        return self.state.get("guard")

    @property
    def candidate(self) -> Any:
        return self.state.get("candidate")


class AgentService:
    def __init__(
        self,
        settings: AgentSettings,
        *,
        graph: Any | None = None,
        chat_store: ChatStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        cache: LocalCache | None = None,
    ) -> None:
        self.settings = settings
        settings.ensure_local_dirs()
        self.chat_store = chat_store or ChatStore(settings.resolved_history_db_path)
        self.checkpoint_store = checkpoint_store or create_checkpoint_store(
            settings.resolved_checkpoint_db_path
        )
        self.metadata_store = MetadataStore(settings)
        self.source_engine = create_agent_engine(settings)
        self.cache = cache if cache is not None else build_local_cache(settings)
        self.catalog_cache = CatalogCache(self.metadata_store, settings.source_id)
        self.embedder = EmbeddingClient(settings, cache=self.cache)
        self.models = AgentModels(settings)
        self.retriever = Retriever(
            settings,
            self.metadata_store,
            self.embedder,
            self.chat_store,
            cache=self.cache,
            catalog_cache=self.catalog_cache,
        )
        self.nodes = AgentNodes(
            settings=settings,
            store=self.metadata_store,
            chat_store=self.chat_store,
            source_engine=self.source_engine,
            embedder=self.embedder,
            models=self.models,
            retriever=self.retriever,
            cache=self.cache,
            catalog_cache=self.catalog_cache,
        )
        self.graph = graph or build_graph(
            self.nodes,
            settings,
            checkpointer=self.checkpoint_store.saver,
        )
        self.conversation = self.chat_store.get_or_create_conversation(
            settings.db_fingerprint,
            settings.source_id,
            database_url=settings.source_url,
        )
        self.conversation_id = str(self.conversation["conversation_id"])

    def start_turn(
        self,
        question: str,
        *,
        on_event: EventCallback | None = None,
    ) -> AgentTurnResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        pending = self.pending_interrupt()
        if pending is not None:
            raise ValueError(
                "A previous query is waiting for clarification or approval; "
                "resolve it before starting another turn."
            )
        conversation = self.chat_store.get_conversation(self.conversation_id)
        if (
            conversation is None
            or conversation.get("db_fingerprint") != self.settings.db_fingerprint
        ):
            conversation = self.chat_store.get_or_create_conversation(
                self.settings.db_fingerprint,
                self.settings.source_id,
                database_url=self.settings.source_url,
            )
        self.conversation = conversation
        self.conversation_id = str(conversation["conversation_id"])
        if str(self.conversation.get("title") or "") in {"Data chat", "New chat"}:
            title = " ".join(question.split())[:80]
            self.chat_store.set_conversation_title(self.conversation_id, title)
            self.conversation["title"] = title
        request_id = uuid.uuid4().hex
        run_id = uuid.uuid4().hex
        thread_id = f"{self.conversation_id}:{request_id}"
        state: AgentState = {
            "conversation_id": self.conversation_id,
            "db_fingerprint": self.settings.db_fingerprint,
            "thread_id": thread_id,
            "request_id": request_id,
            "run_id": run_id,
            "question": question,
            "status": "running",
            "revision_count": 0,
            "human_revision_count": 0,
            "clarification_count": 0,
            "feedback": [],
            "attempts": [],
        }
        return self._run(
            state,
            thread_id=thread_id,
            conversation_id=self.conversation_id,
            run_id=run_id,
            on_event=on_event,
        )

    def resume(
        self,
        thread_id: str,
        decision: HumanDecision | dict[str, Any] | str,
        *,
        conversation_id: str | None = None,
        on_event: EventCallback | None = None,
    ) -> AgentTurnResult:
        if isinstance(decision, HumanDecision):
            resume_value: Any = decision.model_dump(mode="json")
        else:
            resume_value = decision
        values = self._values(thread_id)
        conv_id = conversation_id or str(values.get("conversation_id") or self.conversation_id)
        run_id = str(values.get("run_id") or "")
        return self._run(
            Command(resume=resume_value),
            thread_id=thread_id,
            conversation_id=conv_id,
            run_id=run_id,
            on_event=on_event,
        )

    def pending_interrupt(self) -> AgentTurnResult | None:
        pending = self.chat_store.get_pending(self.conversation_id)
        if pending is None:
            return None
        thread_id = pending["thread_id"]
        values = self._values(thread_id)
        interrupts = self._interrupts(thread_id, values)
        if not interrupts:
            self.chat_store.clear_pending(self.conversation_id)
            return None
        payload = _payload(interrupts[0])
        status = (
            "awaiting_clarification"
            if payload.get("kind") == "clarification"
            else "awaiting_approval"
        )
        return AgentTurnResult(
            thread_id=thread_id,
            conversation_id=self.conversation_id,
            status=status,
            state=values,
            interrupt=payload,
        )

    def refresh(self) -> None:
        self.catalog_cache.invalidate()
        current_id = self.conversation_id
        conversation = self.chat_store.get_conversation(current_id)
        if (
            conversation is None
            or conversation.get("db_fingerprint") != self.settings.db_fingerprint
        ):
            conversation = self.chat_store.get_or_create_conversation(
                self.settings.db_fingerprint,
                self.settings.source_id,
                database_url=self.settings.source_url,
            )
        self.conversation = conversation
        self.conversation_id = str(conversation["conversation_id"])

    def new_conversation(self, *, title: str = "New chat") -> dict[str, Any]:
        self.conversation = self.chat_store.create_conversation(
            self.settings.db_fingerprint,
            self.settings.source_id,
            title=title,
            database_url=self.settings.source_url,
        )
        self.conversation_id = str(self.conversation["conversation_id"])
        return self.conversation

    def conversations(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.chat_store.list_all_conversations(limit=limit)

    def open_conversation(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.chat_store.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found.")
        if conversation.get("db_fingerprint") != self.settings.db_fingerprint:
            raise ValueError("Conversation belongs to a different database.")
        if not conversation.get("database_url"):
            self.chat_store.set_conversation_database_url(
                conversation_id,
                self.settings.source_url,
            )
            conversation["database_url"] = self.settings.source_url
        self.conversation = conversation
        self.conversation_id = conversation_id
        return conversation

    def check_connection(self) -> None:
        import sqlalchemy as sa

        with self.metadata_store.engine.connect() as connection:
            connection.execute(sa.text("SELECT 1"))

    def ingestion_status(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ingested": False,
            "tables": 0,
            "last_run": None,
            "error": None,
        }
        try:
            result["tables"] = self.metadata_store.catalog_table_count(self.settings.source_id)
            result["last_run"] = self.metadata_store.latest_run(self.settings.source_id)
            result["ingested"] = bool(result["tables"])
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def status_report(self) -> dict[str, Any]:
        health: dict[str, Any]
        try:
            health = self.metadata_store.health()
        except Exception as exc:
            health = {"error": str(exc), "counts": {}}
        catalog_tables = 0
        try:
            import sqlalchemy as sa

            from querity.ingestion.db import quote_ident

            with self.metadata_store.engine.connect() as connection:
                catalog_tables = int(
                    connection.execute(
                        sa.text(
                            f"SELECT count(*) FROM "
                            f"{quote_ident(connection, self.metadata_store.schema)}."
                            "table_catalog WHERE source_id = :source_id AND is_active = true"
                        ),
                        {"source_id": self.settings.source_id},
                    ).scalar_one()
                )
        except Exception:
            catalog_tables = 0
        return {
            "database": _safe_database_label(self.settings.source_url),
            "source_id": self.settings.source_id,
            "source_schemas": self.settings.source_schemas,
            "metadata_schema": self.settings.metadata_schema,
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model,
            "fallback_models": self.settings.llm_fallback_models,
            "review_model": self.settings.effective_review_model,
            "embedding": f"{self.settings.embedding_provider}:{self.settings.embedding_model}",
            "catalog_tables": catalog_tables,
            "health": health,
            "chat": self.chat_store.stats(self.conversation_id),
            "pending": self.chat_store.get_pending(self.conversation_id),
            "history_db": str(self.chat_store.path),
            "checkpoint_db": str(self.checkpoint_store.path),
        }

    def close(self) -> None:
        try:
            self.source_engine.dispose()
        finally:
            try:
                self.metadata_store.close()
            finally:
                self.checkpoint_store.close()
                self.chat_store.close()

    def _run(
        self,
        graph_input: Any,
        *,
        thread_id: str,
        conversation_id: str,
        run_id: str,
        on_event: EventCallback | None,
    ) -> AgentTurnResult:
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(
                100,
                20 + 8 * self.settings.max_human_revisions + 6 * self.settings.max_revisions,
            ),
        }
        events: list[str] = []
        try:
            if on_event is None:
                self.graph.invoke(graph_input, config)
            else:
                for chunk in self.graph.stream(graph_input, config, stream_mode="updates"):
                    for node, update in chunk.items():
                        label = "interrupt" if node == "__interrupt__" else node
                        events.append(label)
                        on_event(label, update)
            values = self._values(thread_id)
            interrupts = self._interrupts(thread_id, values)
            if interrupts:
                payload = _payload(interrupts[0])
                status = (
                    "awaiting_clarification"
                    if payload.get("kind") == "clarification"
                    else "awaiting_approval"
                )
                return AgentTurnResult(
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    status=status,
                    state=values,
                    interrupt=payload,
                    events=events,
                )
            status = str(values.get("status") or "failed")
            error = values.get("error")
            if status in {"failed", "rejected"} and not error:
                error = values.get("final_message")
            return AgentTurnResult(
                thread_id=thread_id,
                conversation_id=conversation_id,
                status=status,
                state=values,
                error=str(error) if error else None,
                events=events,
            )
        except Exception as exc:
            logger.exception("Agent turn failed")
            message = f"{type(exc).__name__}: {exc}"
            values = self._values(thread_id)
            values.setdefault("status", "failed")
            values.setdefault("conversation_id", conversation_id)
            values.setdefault("run_id", run_id)
            self.chat_store.clear_pending(conversation_id)
            self.chat_store.add_message(
                conversation_id=conversation_id,
                thread_id=thread_id,
                role="assistant",
                content=f"The agent failed before completing the request.\n\n{message}",
                message_id=hashlib.sha256(f"{run_id}:assistant".encode()).hexdigest()[:32],
                status="failed",
                query_run_id=run_id or None,
                metadata={"error": message},
            )
            if run_id:
                existing_run = self.chat_store.get_query_run(run_id)
                if existing_run is None or existing_run.get("status") != "completed":
                    self.chat_store.update_query_run(
                        run_id,
                        status="failed",
                        error=message,
                    )
            return AgentTurnResult(
                thread_id=thread_id,
                conversation_id=conversation_id,
                status="failed",
                state=values,
                error=message,
                events=events,
            )

    def _values(self, thread_id: str) -> dict[str, Any]:
        try:
            snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
            return dict(snapshot.values)
        except Exception:
            return {}

    def _interrupts(self, thread_id: str, values: dict[str, Any]) -> tuple[Any, ...]:
        try:
            snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
            if snapshot.interrupts:
                return tuple(snapshot.interrupts)
        except Exception:
            pass
        raw = values.get("__interrupt__")
        if raw:
            return tuple(raw)
        return ()


def _payload(interrupt: Any) -> dict[str, Any]:
    value = getattr(interrupt, "value", interrupt)
    if isinstance(value, dict):
        return value
    return {"kind": "unknown", "value": value}


def _safe_database_label(url: str) -> str:
    from sqlalchemy.engine import make_url

    try:
        parsed = make_url(url)
    except Exception:
        return "(unparseable database url)"
    return (
        f"{parsed.host or 'local'}:{parsed.port or ''}/"
        f"{parsed.database or ''} ({parsed.username or 'unknown'})"
    )
