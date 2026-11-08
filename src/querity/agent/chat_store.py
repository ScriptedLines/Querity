from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from querity.agent.state import HistoryHit


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _fts_tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_]{2,}", text)][:16]


class ChatStore:
    """Local durable conversation and query history.

    Every conversation is bound to one database fingerprint, while chats from
    multiple databases share the same local history. A separate LangGraph
    checkpointer database is used for graph state because it has a very different
    write pattern.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()
        with contextlib.suppress(OSError):
            self._path.chmod(0o600)

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._path), timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    db_fingerprint TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    database_url TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT 'Data chat',
                    summary TEXT,
                    pending_thread_id TEXT,
                    pending_kind TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    thread_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT,
                    sql TEXT,
                    query_run_id TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS messages_conversation_created_idx
                    ON messages (conversation_id, created_at);

                CREATE TABLE IF NOT EXISTS query_runs (
                    run_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    plan TEXT,
                    sql TEXT,
                    sql_hash TEXT,
                    status TEXT NOT NULL,
                    reviewer_verdict TEXT,
                    revision_count INTEGER NOT NULL DEFAULT 0,
                    preview TEXT,
                    row_count INTEGER,
                    duration_ms INTEGER,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    executed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS query_runs_conversation_created_idx
                    ON query_runs (conversation_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS query_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS query_attempts_run_idx
                    ON query_attempts (run_id, attempt_id);

                CREATE TABLE IF NOT EXISTS verified_turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    sql TEXT NOT NULL,
                    embedding TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS verified_turns_conversation_idx
                    ON verified_turns (conversation_id, created_at DESC);
                """
            )
            for column, definition in (
                ("pending_thread_id", "TEXT"),
                ("pending_kind", "TEXT"),
                ("summary", "TEXT"),
                ("database_url", "TEXT NOT NULL DEFAULT ''"),
            ):
                with contextlib.suppress(sqlite3.OperationalError):
                    connection.execute(
                        f"ALTER TABLE conversations ADD COLUMN {column} {definition}"
                    )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS verified_turns_fts USING fts5(
                        turn_id UNINDEXED,
                        conversation_id UNINDEXED,
                        question,
                        sql,
                        tokenize = 'porter'
                    )
                    """
                )
            except sqlite3.OperationalError:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS verified_turns_fts (
                        turn_id TEXT,
                        conversation_id TEXT,
                        question TEXT,
                        sql TEXT
                    )
                    """
                )

    def get_or_create_conversation(
        self,
        db_fingerprint: str,
        source_id: str,
        *,
        title: str = "Data chat",
        database_url: str = "",
    ) -> dict[str, Any]:
        now = _now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE db_fingerprint = ? "
                "ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                (db_fingerprint,),
            ).fetchone()
            if row is None:
                conversation_id = db_fingerprint
                connection.execute(
                    "INSERT INTO conversations "
                    "(conversation_id, db_fingerprint, source_id, database_url, title, "
                    "created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        conversation_id,
                        db_fingerprint,
                        source_id,
                        database_url,
                        title,
                        now,
                        now,
                    ),
                )
            else:
                conversation_id = str(row["conversation_id"])
                connection.execute(
                    "UPDATE conversations SET source_id = ?, "
                    "database_url = CASE WHEN ? <> '' THEN ? ELSE database_url END, "
                    "updated_at = ? WHERE conversation_id = ?",
                    (source_id, database_url, database_url, now, conversation_id),
                )
            result = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(result)  # type: ignore[arg-type]

    def create_conversation(
        self,
        db_fingerprint: str,
        source_id: str,
        *,
        title: str = "New chat",
        database_url: str = "",
    ) -> dict[str, Any]:
        now = _now()
        conversation_id = uuid.uuid4().hex
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations "
                "(conversation_id, db_fingerprint, source_id, database_url, title, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    db_fingerprint,
                    source_id,
                    database_url,
                    title,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row)  # type: ignore[arg-type]

    def set_conversation_database_url(self, conversation_id: str, database_url: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET database_url = ? WHERE conversation_id = ?",
                (database_url, conversation_id),
            )

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_conversations(
        self,
        db_fingerprint: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT conversation_id, db_fingerprint, source_id, database_url, title, "
                "created_at, updated_at FROM conversations "
                "WHERE db_fingerprint = ? ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                (db_fingerprint, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_all_conversations(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT conversation_id, db_fingerprint, source_id, database_url, title, "
                "created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_conversation_title(self, conversation_id: str, title: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
                (title[:120], _now(), conversation_id),
            )

    def set_pending(
        self,
        conversation_id: str,
        *,
        thread_id: str,
        kind: str,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET pending_thread_id = ?, pending_kind = ?, "
                "updated_at = ? WHERE conversation_id = ?",
                (thread_id, kind, _now(), conversation_id),
            )

    def clear_pending(self, conversation_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET pending_thread_id = NULL, pending_kind = NULL, "
                "updated_at = ? WHERE conversation_id = ?",
                (_now(), conversation_id),
            )

    def get_pending(self, conversation_id: str) -> dict[str, str] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT pending_thread_id, pending_kind FROM conversations "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None or not row["pending_thread_id"]:
            return None
        return {
            "thread_id": row["pending_thread_id"],
            "kind": row["pending_kind"] or "approval",
        }

    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        thread_id: str | None = None,
        status: str | None = None,
        sql: str | None = None,
        query_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> str:
        identifier = (
            message_id
            or hashlib.sha256(f"{conversation_id}:{role}:{content}:{_now()}".encode()).hexdigest()[
                :32
            ]
        )
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO messages "
                "(message_id, conversation_id, thread_id, role, content, status, sql, "
                "query_run_id, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    conversation_id,
                    thread_id,
                    role,
                    content,
                    status,
                    sql,
                    query_run_id,
                    _json(metadata or {}),
                    now,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
        return identifier

    def update_message(
        self,
        message_id: str,
        *,
        content: str | None = None,
        status: str | None = None,
        sql: str | None = None,
        query_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("content", content),
            ("status", status),
            ("sql", sql),
            ("query_run_id", query_run_id),
            ("metadata", _json(metadata) if metadata is not None else None),
        ):
            if value is not None:
                updates.append(f"{column} = ?")
                params.append(value)
        if not updates:
            return
        params.append(message_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE messages SET {', '.join(updates)} WHERE message_id = ?",
                params,
            )

    def recent_messages(
        self,
        conversation_id: str,
        *,
        limit: int = 12,
        exclude_message_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if exclude_message_id:
            sql += " AND message_id != ?"
            params.append(exclude_message_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_messages(
        self,
        conversation_id: str,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?"
        )
        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, (conversation_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def conversation_context(
        self,
        conversation_id: str,
        *,
        limit: int = 12,
        exclude_message_id: str | None = None,
        max_chars: int = 8_000,
    ) -> str:
        messages = self.recent_messages(
            conversation_id,
            limit=limit,
            exclude_message_id=exclude_message_id,
        )
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).capitalize()
            content = str(message.get("content", "")).strip()
            if len(content) > 1_500:
                content = content[:1_500] + "..."
            lines.append(f"{role}: {content}")
            if message.get("sql"):
                sql = str(message["sql"]).strip()
                if len(sql) > 2_000:
                    sql = sql[:2_000] + "..."
                lines.append(f"SQL: {sql}")
        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[-max_chars:]
        return context

    def create_query_run(
        self,
        *,
        run_id: str,
        conversation_id: str,
        question: str,
    ) -> str:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO query_runs "
                "(run_id, conversation_id, question, status, created_at) "
                "VALUES (?, ?, ?, 'running', ?)",
                (run_id, conversation_id, question, _now()),
            )
        return run_id

    def update_query_run(self, run_id: str, **fields: Any) -> None:
        allowed = {
            "plan",
            "sql",
            "sql_hash",
            "status",
            "reviewer_verdict",
            "revision_count",
            "preview",
            "row_count",
            "duration_ms",
            "error",
            "executed_at",
        }
        updates: list[str] = []
        params: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            rendered = _json(value) if key in {"plan", "reviewer_verdict", "preview"} else value
            updates.append(f"{key} = ?")
            params.append(rendered)
        if not updates:
            return
        params.append(run_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE query_runs SET {', '.join(updates)} WHERE run_id = ?",
                params,
            )

    def get_query_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM query_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def record_attempt(
        self,
        *,
        run_id: str,
        stage: str,
        revision: int,
        payload: dict[str, Any] | str,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO query_attempts (run_id, stage, revision, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    stage,
                    revision,
                    payload if isinstance(payload, str) else _json(payload),
                    _now(),
                ),
            )

    def save_verified_turn(
        self,
        *,
        turn_id: str,
        conversation_id: str,
        question: str,
        sql: str,
        embedding: list[float] | None,
    ) -> None:
        embedding_json = _json(embedding) if embedding else None
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO verified_turns "
                "(turn_id, conversation_id, question, sql, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (turn_id, conversation_id, question, sql, embedding_json, _now()),
            )
            connection.execute("DELETE FROM verified_turns_fts WHERE turn_id = ?", (turn_id,))
            connection.execute(
                "INSERT INTO verified_turns_fts (turn_id, conversation_id, question, sql) "
                "VALUES (?, ?, ?, ?)",
                (turn_id, conversation_id, question, sql),
            )

    def search_verified_turns(
        self,
        *,
        conversation_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[HistoryHit]:
        hits: dict[str, HistoryHit] = {}
        tokens = _fts_tokens(query)
        if tokens:
            match = " OR ".join(f'"{token}"*' for token in tokens)
            try:
                with self._lock, self._connect() as connection:
                    rows = connection.execute(
                        """
                        SELECT t.turn_id, t.question, t.sql, t.created_at,
                               bm25(verified_turns_fts) AS rank
                        FROM verified_turns_fts f
                        JOIN verified_turns t ON t.turn_id = f.turn_id
                        WHERE f.conversation_id = ?
                          AND verified_turns_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (conversation_id, match, limit),
                    ).fetchall()
                for row in rows:
                    rank = float(row["rank"] or 0.0)
                    hits[row["turn_id"]] = HistoryHit(
                        turn_id=row["turn_id"],
                        question=row["question"],
                        sql=row["sql"],
                        created_at=row["created_at"],
                        score=1.0 / (1.0 + abs(rank)),
                    )
            except sqlite3.OperationalError:
                pass

        if query_embedding:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    "SELECT turn_id, question, sql, created_at, embedding "
                    "FROM verified_turns WHERE conversation_id = ? "
                    "AND embedding IS NOT NULL ORDER BY created_at DESC LIMIT 500",
                    (conversation_id,),
                ).fetchall()
            for row in rows:
                embedding = _loads(row["embedding"], None)
                if not isinstance(embedding, list):
                    continue
                score = _cosine(query_embedding, [float(value) for value in embedding])
                if score < 0.2:
                    continue
                existing = hits.get(row["turn_id"])
                if existing is None or score > existing.score:
                    hits[row["turn_id"]] = HistoryHit(
                        turn_id=row["turn_id"],
                        question=row["question"],
                        sql=row["sql"],
                        created_at=row["created_at"],
                        score=score,
                    )

        return sorted(hits.values(), key=lambda hit: hit.score, reverse=True)[:limit]

    def recent_verified_turns(
        self,
        conversation_id: str,
        *,
        limit: int = 5,
    ) -> list[HistoryHit]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT turn_id, question, sql, created_at FROM verified_turns "
                "WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [
            HistoryHit(
                turn_id=row["turn_id"],
                question=row["question"],
                sql=row["sql"],
                created_at=row["created_at"],
                score=0.0,
            )
            for row in rows
        ]

    def stats(self, conversation_id: str) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            messages = connection.execute(
                "SELECT count(*) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            runs = connection.execute(
                "SELECT count(*) FROM query_runs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            verified = connection.execute(
                "SELECT count(*) FROM verified_turns WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
        return {
            "messages": int(messages),
            "query_runs": int(runs),
            "verified_turns": int(verified),
        }

    def recent_runs(self, conversation_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, question, sql, status, row_count, duration_ms, error, created_at "
                "FROM query_runs WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_query_runs(self, conversation_id: str) -> dict[str, dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM query_runs WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
        return {str(row["run_id"]): dict(row) for row in rows}

    def close(self) -> None:
        return None


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if not left_values or len(left_values) != len(right_values):
        return 0.0
    dot = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
