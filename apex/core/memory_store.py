"""SQLite-backed key-value memory store."""
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Any

from apex.core.types import Tool

_SCHEMA_LOCK = threading.Lock()


@contextmanager
def _connection(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(db_path: Path) -> None:
    """Create the memory schema and enable WAL without concurrent pragma races."""
    with _SCHEMA_LOCK:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    ts    REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def make_memory_tools(db_path: Path) -> tuple[Tool, Tool]:
    """Return memory_read and memory_write tools bound to db_path."""
    db_path = Path(db_path).expanduser()
    _ensure_schema(db_path)

    def memory_read_effect(args: dict) -> dict:
        key: str | None = args.get("key")
        with _connection(db_path) as conn:
            if key:
                row = conn.execute(
                    "SELECT value, ts FROM kv WHERE key = ?", (key,)
                ).fetchone()
                if row:
                    return {"key": key, "value": json.loads(row[0]), "ts": row[1]}
                return {"key": key, "value": None}
            rows = conn.execute(
                "SELECT key, value, ts FROM kv ORDER BY ts DESC"
            ).fetchall()
            return {
                "entries": [
                    {"key": row[0], "value": json.loads(row[1]), "ts": row[2]}
                    for row in rows
                ]
            }

    def memory_write_effect(args: dict) -> dict:
        key: str = args["key"]
        value: Any = args["value"]
        with _connection(db_path) as conn:
            conn.execute(
                "INSERT INTO kv (key, value, ts) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
                (key, json.dumps(value), time()),
            )
        return {"key": key, "written": True}

    memory_read = Tool(
        name="memory_read",
        input_spec={"key": str},
        output_spec={"entries": list, "key": str, "value": object, "ts": float},
        effect=memory_read_effect,
        required=frozenset(),
        retry_safe=True,
    )
    memory_write = Tool(
        name="memory_write",
        input_spec={"key": str, "value": object},
        output_spec={"key": str, "written": bool},
        effect=memory_write_effect,
    )
    return memory_read, memory_write
