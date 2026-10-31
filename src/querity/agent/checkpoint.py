from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from querity.agent import state as agent_state

_ALLOWED_STATE_TYPES = [
    value
    for value in vars(agent_state).values()
    if isinstance(value, type)
    and issubclass(value, BaseModel)
    and value.__module__ == agent_state.__name__
]


@dataclass
class CheckpointStore:
    saver: SqliteSaver
    connection: sqlite3.Connection
    path: Path

    def close(self) -> None:
        self.connection.close()


def create_checkpoint_store(path: Path | str) -> CheckpointStore:
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(resolved), timeout=30, check_same_thread=False)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=_ALLOWED_STATE_TYPES,
        allowed_json_modules=_ALLOWED_STATE_TYPES,
    )
    saver = SqliteSaver(connection, serde=serde)
    saver.setup()
    with contextlib.suppress(OSError):
        resolved.chmod(0o600)
    return CheckpointStore(saver=saver, connection=connection, path=resolved)
