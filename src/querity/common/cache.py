from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import struct
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_CACHE_MAX_BYTES = 256 * 1024 * 1024
_EMBEDDING_NAMESPACE = "embedding"


def hash_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}d", *vector)


def _unpack_vector(blob: bytes) -> list[float] | None:
    if not blob or len(blob) % 8:
        return None
    return list(struct.unpack(f"<{len(blob) // 8}d", blob))


class LocalCache:
    """Private, process-safe SQLite cache for local CLI and TUI usage."""

    def __init__(self, path: Path | str, *, max_bytes: int = DEFAULT_CACHE_MAX_BYTES) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._max_bytes = max(1_048_576, max_bytes)
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
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS cache_entries_created_idx ON cache_entries (created_at)"
            )

    def get(
        self,
        namespace: str,
        key: str,
        *,
        max_age: float | None = None,
        now: float | None = None,
    ) -> bytes | None:
        current = time.time() if now is None else now
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value, created_at, expires_at FROM cache_entries "
                "WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
            if row is None:
                return None
            expired = row["expires_at"] is not None and float(row["expires_at"]) <= current
            if max_age is not None and current - float(row["created_at"]) > max_age:
                expired = True
            if expired:
                connection.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
                return None
            value = row["value"]
            return bytes(value) if value is not None else None

    def set(
        self,
        namespace: str,
        key: str,
        value: bytes,
        *,
        ttl: float | None = None,
        now: float | None = None,
    ) -> None:
        current = time.time() if now is None else now
        expires_at = current + ttl if ttl is not None else None
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO cache_entries (namespace, key, value, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (namespace, key) DO UPDATE SET "
                "value = excluded.value, created_at = excluded.created_at, "
                "expires_at = excluded.expires_at",
                (namespace, key, sqlite3.Binary(value), current, expires_at),
            )
            self._prune(connection, current)

    def get_json(
        self,
        namespace: str,
        key: str,
        *,
        max_age: float | None = None,
        now: float | None = None,
    ) -> Any | None:
        value = self.get(namespace, key, max_age=max_age, now=now)
        if value is None:
            return None
        try:
            return json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.delete(namespace, key)
            return None

    def set_json(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        ttl: float | None = None,
        now: float | None = None,
    ) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
        self.set(namespace, key, payload.encode("utf-8"), ttl=ttl, now=now)

    def get_vector(self, key: str) -> list[float] | None:
        value = self.get(_EMBEDDING_NAMESPACE, key)
        return _unpack_vector(value) if value is not None else None

    def set_vector(self, key: str, vector: Sequence[float]) -> None:
        self.set(_EMBEDDING_NAMESPACE, key, _pack_vector(vector))

    def delete(self, namespace: str, key: str | None = None) -> None:
        with self._lock, self._connect() as connection:
            if key is None:
                connection.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
            else:
                connection.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM cache_entries")

    def stats(self) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS entries, COALESCE(SUM(LENGTH(value)), 0) AS bytes "
                "FROM cache_entries"
            ).fetchone()
        return {
            "entries": int(row["entries"]),
            "bytes": int(row["bytes"]),
        }

    def _prune(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        row = connection.execute(
            "SELECT COALESCE(SUM(LENGTH(value)), 0) FROM cache_entries"
        ).fetchone()
        total = int(row[0]) if row is not None else 0
        if total <= self._max_bytes:
            return
        target = int(self._max_bytes * 0.9)
        rows = connection.execute(
            "SELECT rowid, LENGTH(value) FROM cache_entries ORDER BY created_at LIMIT 10000"
        ).fetchall()
        for cached_row in rows:
            if total <= target:
                break
            connection.execute("DELETE FROM cache_entries WHERE rowid = ?", (cached_row[0],))
            total -= int(cached_row[1])


def build_local_cache(settings: Any) -> LocalCache | None:
    if not bool(getattr(settings, "cache_enabled", False)):
        return None
    path = getattr(settings, "resolved_cache_db_path", None)
    if path is None:
        return None
    max_bytes = int(getattr(settings, "cache_max_bytes", DEFAULT_CACHE_MAX_BYTES))
    return LocalCache(path, max_bytes=max_bytes)
