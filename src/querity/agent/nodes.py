from __future__ import annotations

import hashlib
import time
from typing import Any

import sqlalchemy as sa
from langgraph.types import interrupt
from sqlalchemy.engine import Engine

from querity.agent.catalog import CatalogCache, CatalogEmptyError
from querity.agent.chat_store import ChatStore
from querity.agent.llm import AgentModels, build_messages
from querity.agent.prompts import (
    generator_messages,
    planner_messages,
    reviewer_messages,
)
from querity.agent.retrieval import Retriever
from querity.agent.settings import AgentSettings
from querity.agent.sql_guard import SqlGuard
from querity.agent.state import (
    AgentState,
    AttemptRecord,
    ExecutionResult,
    HumanDecision,
    QueryPlan,
    ReviewVerdict,
    SQLCandidate,
    SQLIssue,
)
from querity.common.cache import LocalCache
from querity.common.logging import get_logger
from querity.common.masking import is_sensitive_column, mask_value, truncate_value
from querity.ingestion.db import readonly_connection
from querity.ingestion.store import MetadataStore

logger = get_logger(__name__)


def _hash(sql: str) -> str:
    return hashlib.sha256(" ".join(sql.split()).encode("utf-8")).hexdigest()


def _message_id(run_id: str, role: str) -> str:
    return hashlib.sha256(f"{run_id}:{role}".encode()).hexdigest()[:32]


def _issue_list(value: Any) -> list[SQLIssue]:
    if not value:
        return []
    issues: list[SQLIssue] = []
    for item in value:
        if isinstance(item, SQLIssue):
            issues.append(item)
        elif isinstance(item, dict):
            issues.append(SQLIssue.model_validate(item))
    return issues


class AgentNodes:
    def __init__(
        self,
        *,
        settings: AgentSettings,
        store: MetadataStore,
        chat_store: ChatStore,
        source_engine: Engine,
        embedder: Any,
        models: AgentModels,
        retriever: Retriever,
        cache: LocalCache | None = None,
        catalog_cache: CatalogCache | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.chat_store = chat_store
        self.source_engine = source_engine
        self.embedder = embedder
        self.models = models
        self.retriever = retriever
        self.cache = cache
        self.catalog_cache = catalog_cache or CatalogCache(store, settings.source_id)

    def catalog(self) -> Any:
        return self.catalog_cache.get()

    def ingest_turn(self, state: AgentState) -> dict[str, Any]:
        conversation_id = state["conversation_id"]
        run_id = state["run_id"]
        thread_id = state["thread_id"]
        question = state["question"].strip()
        user_message_id = _message_id(run_id, "user")
        self.chat_store.create_query_run(
            run_id=run_id,
            conversation_id=conversation_id,
            request_id=state["request_id"],
            question=question,
        )
        self.chat_store.add_message(
            conversation_id=conversation_id,
            thread_id=thread_id,
            role="user",
            content=question,
            message_id=user_message_id,
            query_run_id=run_id,
        )
        recent = self.chat_store.conversation_context(
            conversation_id,
            limit=12,
            exclude_message_id=user_message_id,
        )
        return {
            "question": question,
            "user_message_id": user_message_id,
            "recent_context": recent,
            "status": "running",
            "revision_count": int(state.get("revision_count", 0)),
            "human_revision_count": int(state.get("human_revision_count", 0)),
            "clarification_count": int(state.get("clarification_count", 0)),
        }

    def analyze(self, state: AgentState) -> dict[str, Any]:
        try:
            system, human = planner_messages(
                state["question"],
                state.get("recent_context", ""),
                state.get("clarification"),
            )
            plan = self.models.plan(build_messages(system, human), QueryPlan)
        except Exception as exc:
            logger.exception("Agent planning failed")
            return self._failure(exc)
        self.chat_store.update_query_run(
            state["run_id"],
            plan=plan.model_dump(mode="json"),
            status="running",
        )
        can_clarify = (
            self.settings.enable_clarification
            and int(state.get("clarification_count", 0)) < 1
            and not state.get("clarification")
        )
        if not plan.answerable and not can_clarify:
            message = (
                plan.clarification_question or "The question cannot be answered from this catalog."
            )
            return {
                "plan": plan,
                "status": "failed",
                "error": message,
                "final_message": message,
            }
        return {
            "plan": plan,
            "status": "running",
            "error": None,
        }

    def clarify(self, state: AgentState) -> dict[str, Any]:
        plan = state.get("plan") or QueryPlan(intent=state["question"])
        question = (
            plan.clarification_question
            or "Could you clarify the entity, metric, or time range you want?"
        )
        self.chat_store.update_query_run(state["run_id"], status="awaiting_clarification")
        self.chat_store.set_pending(
            state["conversation_id"],
            thread_id=state["thread_id"],
            kind="clarification",
        )
        payload = {
            "kind": "clarification",
            "thread_id": state["thread_id"],
            "run_id": state["run_id"],
            "question": question,
            "plan": plan.model_dump(mode="json"),
        }
        response = interrupt(payload)
        self.chat_store.clear_pending(state["conversation_id"])
        answer = _clarification_answer(response)
        self.chat_store.update_query_run(state["run_id"], status="running")
        if not answer:
            return {
                "status": "failed",
                "error": "No clarification answer was provided.",
                "final_message": "I need a clarification before I can write this query.",
            }
        clarification = f"Question: {question}\nAnswer: {answer}"
        return {
            "clarification": clarification,
            "clarification_count": int(state.get("clarification_count", 0)) + 1,
            "plan": None,
            "status": "running",
            "error": None,
        }

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        plan = state.get("plan") or QueryPlan(intent=state["question"])
        try:
            bundle = self.retriever.retrieve(
                question=state["question"],
                plan=plan,
                conversation_id=state["conversation_id"],
                recent_context=state.get("recent_context", ""),
            )
        except CatalogEmptyError as exc:
            return {
                "status": "failed",
                "error": str(exc),
                "final_message": (
                    "This database has no indexed catalog yet. Run `querity ingest` "
                    "or use `/ingest` in the chat first."
                ),
            }
        except Exception as exc:
            logger.exception("Agent retrieval failed")
            return self._failure(exc)
        return {"retrieval": bundle, "status": "running", "error": None}

    def generate(self, state: AgentState) -> dict[str, Any]:
        plan = state.get("plan") or QueryPlan(intent=state["question"])
        retrieval = state.get("retrieval")
        if retrieval is None:
            return self._failure_message("No retrieval context is available.")
        attempt = int(state.get("revision_count", 0)) + 1
        try:
            system, human = generator_messages(
                question=state["question"],
                plan=plan,
                retrieval=retrieval,
                feedback=_issue_list(state.get("feedback")),
                previous_sql=state.get("sql"),
                attempt=attempt,
            )
            candidate = self.models.generate_sql(build_messages(system, human), SQLCandidate)
        except Exception as exc:
            logger.exception("Agent SQL generation failed")
            return self._failure(exc)
        if not candidate.sql.strip():
            return self._failure_message("The model returned an empty SQL query.")
        sql_hash = _hash(candidate.sql)
        self.chat_store.record_attempt(
            run_id=state["run_id"],
            stage="generate",
            revision=int(state.get("revision_count", 0)),
            payload=candidate.model_dump(mode="json"),
        )
        return {
            "candidate": candidate,
            "sql": candidate.sql.strip(),
            "sql_hash": sql_hash,
            "status": "running",
            "error": None,
            "attempts": [
                AttemptRecord(
                    attempt=attempt,
                    sql=candidate.sql,
                )
            ],
        }

    def guard(self, state: AgentState) -> dict[str, Any]:
        sql = state.get("sql")
        if not sql:
            return self._failure_message("There is no SQL query to validate.")
        try:
            catalog_version = self.catalog_cache.version()
            guard = SqlGuard(
                self.settings,
                self.catalog_cache.get(catalog_version),
                self.source_engine,
                cache=self.cache,
                catalog_version=catalog_version,
            ).validate(sql)
        except Exception as exc:
            logger.exception("Agent SQL guard failed")
            return self._failure(exc)
        self.chat_store.record_attempt(
            run_id=state["run_id"],
            stage="guard",
            revision=int(state.get("revision_count", 0)),
            payload=guard.model_dump(mode="json"),
        )
        self.chat_store.update_query_run(
            state["run_id"],
            sql=guard.sql,
            sql_hash=guard.sql_hash,
            revision_count=int(state.get("revision_count", 0)),
        )
        return {
            "guard": guard,
            "sql": guard.sql,
            "sql_hash": guard.sql_hash,
            "status": "running",
            "error": None,
        }

    def review(self, state: AgentState) -> dict[str, Any]:
        guard = state.get("guard")
        sql = state.get("sql")
        plan = state.get("plan") or QueryPlan(intent=state["question"])
        retrieval = state.get("retrieval")
        if guard is None or not guard.ok or sql is None or retrieval is None:
            return self._failure_message("The SQL review stage received an invalid candidate.")
        try:
            system, human = reviewer_messages(
                question=state["question"],
                plan=plan,
                retrieval=retrieval,
                sql=sql,
                guard_warnings=guard.warnings,
                attempt=int(state.get("revision_count", 0)) + 1,
            )
            review = self.models.review(build_messages(system, human), ReviewVerdict)
        except Exception as exc:
            logger.exception("Agent SQL review failed")
            return self._failure(exc)
        if review.blocking:
            review = review.model_copy(update={"approved": False})
        self.chat_store.record_attempt(
            run_id=state["run_id"],
            stage="review",
            revision=int(state.get("revision_count", 0)),
            payload=review.model_dump(mode="json"),
        )
        self.chat_store.update_query_run(
            state["run_id"],
            reviewer_verdict=review.model_dump(mode="json"),
        )
        return {"review": review, "status": "running", "error": None}

    def revise(self, state: AgentState) -> dict[str, Any]:
        issues: list[SQLIssue] = []
        guard = state.get("guard")
        review = state.get("review")
        if guard is not None and not guard.ok:
            issues.extend(guard.errors)
        if review is not None and not review.approved:
            issues.extend(review.issues)
        if state.get("error") and not issues:
            issues.append(
                SQLIssue(
                    code="execution_error",
                    message=state["error"] or "The previous SQL could not be executed.",
                )
            )
        if not issues:
            issues.append(
                SQLIssue(
                    code="revision_requested",
                    message="The previous query needs to be revised.",
                )
            )
        return {
            "feedback": issues,
            "revision_count": int(state.get("revision_count", 0)) + 1,
            "guard": None,
            "review": None,
            "candidate": None,
            "pre_approved": False,
            "human_decision": None,
            "human_edited": False,
            "status": "running",
            "error": None,
        }

    def human_review(self, state: AgentState) -> dict[str, Any]:
        guard = state.get("guard")
        review = state.get("review")
        if guard is None or state.get("sql") is None:
            return self._failure_message("There is no validated SQL to review.")
        can_approve = bool(guard.ok) and not bool(state.get("error"))
        payload = {
            "kind": "approval",
            "thread_id": state["thread_id"],
            "run_id": state["run_id"],
            "question": state["question"],
            "sql": state.get("sql"),
            "sql_hash": state.get("sql_hash"),
            "explanation": (state.get("candidate").explanation if state.get("candidate") else ""),
            "assumptions": (state.get("candidate").assumptions if state.get("candidate") else []),
            "guard": guard.model_dump(mode="json"),
            "review": review.model_dump(mode="json") if review else None,
            "can_approve": can_approve,
            "revision_count": int(state.get("revision_count", 0)),
        }
        self.chat_store.update_query_run(state["run_id"], status="awaiting_approval")
        self.chat_store.set_pending(
            state["conversation_id"],
            thread_id=state["thread_id"],
            kind="approval",
        )
        response = interrupt(payload)
        self.chat_store.clear_pending(state["conversation_id"])
        try:
            decision = _human_decision(response)
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"Invalid human decision: {exc}",
                "final_message": "The approval response was invalid; the query was not run.",
            }

        if decision.kind == "approve":
            if not can_approve:
                return {
                    "status": "failed",
                    "error": "SQL failed deterministic validation and cannot be approved.",
                    "final_message": "This query failed safety validation and was not run.",
                }
            if decision.sql_hash and decision.sql_hash != state.get("sql_hash"):
                return {
                    "status": "failed",
                    "error": "Approval was for a different SQL hash.",
                    "final_message": "The approved SQL changed before execution; nothing was run.",
                }
            self.chat_store.update_query_run(state["run_id"], status="approved")
            return {"human_decision": decision, "status": "running", "error": None}

        if decision.kind == "edit":
            edited = (decision.sql or "").strip()
            if not edited:
                return {
                    "status": "failed",
                    "error": "The edited SQL was empty.",
                    "final_message": "No SQL was executed because the edit was empty.",
                }
            self.chat_store.update_query_run(state["run_id"], status="running")
            return {
                "human_decision": decision,
                "sql": edited,
                "sql_hash": _hash(edited),
                "guard": None,
                "review": None,
                "candidate": state.get("candidate"),
                "pre_approved": bool(decision.pre_approve),
                "human_edited": True,
                "status": "running",
                "error": None,
                "feedback": [],
            }

        if decision.kind == "revise":
            human_count = int(state.get("human_revision_count", 0)) + 1
            if human_count > self.settings.max_human_revisions:
                return {
                    "status": "failed",
                    "error": "The human revision limit was reached.",
                    "final_message": "The revision limit was reached; the query was not executed.",
                }
            feedback = decision.feedback or "Please improve the query."
            self.chat_store.update_query_run(state["run_id"], status="running")
            return {
                "human_decision": decision,
                "feedback": [
                    SQLIssue(
                        code="human_feedback",
                        message=feedback,
                        severity="blocking",
                    )
                ],
                "guard": None,
                "review": None,
                "pre_approved": False,
                "human_edited": False,
                "revision_count": 0,
                "human_revision_count": human_count,
                "status": "running",
                "error": None,
            }

        reason = decision.reason or "The query was rejected by the human reviewer."
        self.chat_store.update_query_run(state["run_id"], status="rejected", error=reason)
        return {
            "human_decision": decision,
            "status": "rejected",
            "final_message": reason,
            "error": None,
        }

    def execute(self, state: AgentState) -> dict[str, Any]:
        sql = state.get("sql")
        sql_hash = state.get("sql_hash")
        decision = state.get("human_decision")
        if not sql or not sql_hash:
            return self._failure_message("There is no approved SQL to execute.")
        approved = decision is not None and decision.kind in {"approve", "edit"}
        if not approved:
            return {
                "status": "failed",
                "error": "SQL execution requires a human approval.",
                "final_message": "No SQL was executed without human approval.",
            }
        if decision.kind == "approve" and decision.sql_hash and decision.sql_hash != sql_hash:
            return {
                "status": "failed",
                "error": "Approval hash no longer matches the SQL.",
                "final_message": "No SQL was executed because the approved query changed.",
            }

        existing = self.chat_store.get_query_run(state["run_id"])
        if existing and existing.get("status") == "completed":
            return {
                "execution": ExecutionResult(
                    ok=True,
                    row_count=int(existing.get("row_count") or 0),
                    duration_ms=int(existing.get("duration_ms") or 0),
                ),
                "status": "running",
                "error": None,
            }

        started = time.monotonic()
        try:
            with readonly_connection(self.source_engine) as connection:
                result = connection.execute(sa.text(sql))
                column_keys = result.keys()
                columns = [str(key) for key in column_keys]
                raw_rows = result.fetchmany(self.settings.max_result_rows + 1)
        except Exception as exc:
            logger.warning("Approved SQL failed to execute: %s", exc)
            issue = SQLIssue(
                code="execution_error",
                message=f"PostgreSQL execution failed: {exc}",
            )
            self.chat_store.update_query_run(
                state["run_id"],
                status="failed",
                error=str(exc),
                sql=sql,
                sql_hash=sql_hash,
            )
            return {
                "execution": ExecutionResult(ok=False, error=str(exc)),
                "feedback": [issue],
                "error": str(exc),
                "status": "running",
                "pre_approved": False,
                "human_edited": bool(state.get("human_edited")),
                "human_decision": None,
            }

        truncated = len(raw_rows) > self.settings.max_result_rows
        raw_rows = raw_rows[: self.settings.max_result_rows]
        catalog_sensitive: set[str] = set()
        try:
            catalog = self.catalog()
            guard = state.get("guard")
            for table_key in guard.tables_used if guard is not None else []:
                info = catalog.tables.get(table_key)
                if info is None:
                    continue
                for name, column in info.columns.items():
                    if column.is_sensitive:
                        catalog_sensitive.add(name.lower())
        except Exception:
            logger.debug("Could not load sensitive column metadata", exc_info=True)
        sensitive = [
            is_sensitive_column(
                column,
                extra_patterns=self.settings.extra_sensitive_patterns,
            )
            or column.lower() in catalog_sensitive
            for column in columns
        ]
        rows: list[list[str]] = []
        for raw_row in raw_rows:
            rendered: list[str] = []
            for index, value in enumerate(raw_row):
                if value is None:
                    rendered.append("")
                elif sensitive[index]:
                    rendered.append(mask_value(value, columns[index]) or "")
                else:
                    rendered.append(truncate_value(value, 1_000) or "")
            rows.append(rendered)
        duration_ms = int((time.monotonic() - started) * 1_000)
        execution = ExecutionResult(
            ok=True,
            columns=columns,
            rows=rows[: self.settings.max_preview_rows],
            row_count=len(rows),
            truncated=truncated or len(rows) > self.settings.max_preview_rows,
            duration_ms=duration_ms,
        )
        preview = {
            "columns": columns,
            "rows": rows[: self.settings.max_preview_rows],
            "truncated": execution.truncated,
        }
        self.chat_store.update_query_run(
            state["run_id"],
            status="completed",
            sql=sql,
            sql_hash=sql_hash,
            row_count=execution.row_count,
            duration_ms=duration_ms,
            preview=preview,
            executed_at=_now(),
            revision_count=int(state.get("revision_count", 0)),
        )
        return {
            "execution": execution,
            "status": "running",
            "error": None,
        }

    def persist(self, state: AgentState) -> dict[str, Any]:
        execution = state.get("execution")
        sql = state.get("sql")
        if execution is None or not execution.ok:
            return {
                "status": "failed",
                "error": "Cannot persist an unsuccessful execution.",
                "final_message": "The query did not complete successfully.",
            }
        candidate = state.get("candidate")
        explanation = candidate.explanation if candidate else "Query completed."
        content = (
            f"{explanation}\n\n"
            f"SQL:\n{sql}\n\n"
            f"Rows returned: {execution.row_count} "
            f"({execution.duration_ms} ms)"
        )
        self.chat_store.add_message(
            conversation_id=state["conversation_id"],
            thread_id=state["thread_id"],
            role="assistant",
            content=content,
            message_id=_message_id(state["run_id"], "assistant"),
            status="completed",
            sql=sql,
            query_run_id=state["run_id"],
            metadata={
                "row_count": execution.row_count,
                "duration_ms": execution.duration_ms,
                "review": state.get("review").model_dump(mode="json")
                if state.get("review")
                else None,
            },
        )
        embedding = None
        try:
            plan = state.get("plan") or QueryPlan(intent=state["question"])
            search_query_fn = getattr(self.retriever, "search_query_text", None)
            if callable(search_query_fn):
                search_query = search_query_fn(state["question"], plan)
            else:
                search_query = f"{state['question']}\n{plan.intent}"
            embedding = self.embedder.embed_query(search_query[:12_000])
        except Exception:
            logger.debug("Could not embed verified turn", exc_info=True)
        if sql:
            self.chat_store.save_verified_turn(
                turn_id=state["run_id"],
                conversation_id=state["conversation_id"],
                question=state["question"],
                sql=sql,
                embedding=embedding,
            )
        return {"status": "completed", "error": None, "final_message": None}

    def finalize(self, state: AgentState) -> dict[str, Any]:
        self.chat_store.clear_pending(state["conversation_id"])
        status = state.get("status", "failed")
        if status == "completed":
            return {"status": "completed"}
        if status not in {"failed", "rejected"}:
            status = "failed"
        message = (
            state.get("final_message")
            or state.get("error")
            or ("The query was rejected." if status == "rejected" else "The query failed.")
        )
        self.chat_store.add_message(
            conversation_id=state["conversation_id"],
            thread_id=state["thread_id"],
            role="assistant",
            content=message,
            message_id=_message_id(state["run_id"], "assistant"),
            status=status,
            sql=state.get("sql"),
            query_run_id=state["run_id"],
            metadata={"error": state.get("error")},
        )
        self.chat_store.update_query_run(
            state["run_id"],
            status=status,
            error=state.get("error") or message,
            sql=state.get("sql"),
            sql_hash=state.get("sql_hash"),
            revision_count=int(state.get("revision_count", 0)),
        )
        return {"status": status, "final_message": message}

    def _failure(self, exc: Exception) -> dict[str, Any]:
        logger.exception("Agent node failed")
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "final_message": "The agent could not complete this request.",
        }

    def _failure_message(self, message: str) -> dict[str, Any]:
        return {"status": "failed", "error": message, "final_message": message}


def _clarification_answer(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        value = response.get("answer") or response.get("text") or response.get("value")
        return str(value).strip() if value is not None else ""
    return ""


def _human_decision(response: Any) -> HumanDecision:
    if isinstance(response, HumanDecision):
        return response
    if isinstance(response, str):
        lowered = response.strip().lower()
        if lowered in {"a", "approve", "approved", "yes", "y"}:
            return HumanDecision(kind="approve")
        if lowered in {"x", "reject", "rejected", "no", "n"}:
            return HumanDecision(kind="reject")
        return HumanDecision(kind="revise", feedback=response.strip())
    if isinstance(response, dict):
        return HumanDecision.model_validate(response)
    raise ValueError("Expected an approval decision object")


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
