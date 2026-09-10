"""Persist run history and tool events in SQLite."""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DB_PATH = Path(os.environ.get("APEX_HISTORY_DB_PATH", "~/.apex/runs.db")).expanduser()

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task         TEXT    NOT NULL,
    plan_json    TEXT,
    exit_code    INTEGER,
    token_count  INTEGER,
    wall_seconds REAL,
    timestamp    TEXT    NOT NULL DEFAULT (datetime('now','utc'))
);
"""

_DDL_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    step        INTEGER NOT NULL,
    tool        TEXT    NOT NULL,
    args_json   TEXT,
    result_json TEXT,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now','utc'))
);
"""


def _json(value) -> str | None:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value is not None else None


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(_DDL_RUNS)
    conn.execute(_DDL_EVENTS)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_run(
    task: str,
    plan: dict | list | None,
    exit_code: int,
    token_count: int,
    wall_seconds: float,
    events: list[dict] | None = None,
) -> int:
    """Insert one run and all supplied events atomically; return the run id."""
    with _conn() as conn:
        cursor = conn.execute(
            "INSERT INTO runs (task, plan_json, exit_code, token_count, wall_seconds) "
            "VALUES (?, ?, ?, ?, ?)",
            (task, _json(plan), exit_code, token_count, wall_seconds),
        )
        run_id = int(cursor.lastrowid)
        for event in events or []:
            conn.execute(
                "INSERT INTO events (run_id, step, tool, args_json, result_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    event["step"],
                    event["tool"],
                    _json(event.get("args")),
                    _json(event.get("result")),
                ),
            )
    return run_id


def list_runs(n: int = 20) -> list[dict]:
    limit = max(0, min(int(n), 1000))
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, task, exit_code, token_count, wall_seconds, timestamp "
            "FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_events(run_id: int) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT step, tool, args_json, result_json, timestamp FROM events "
            "WHERE run_id = ? ORDER BY step, id",
            (run_id,),
        ).fetchall()
    return [
        {
            "step": row["step"],
            "tool": row["tool"],
            "args": json.loads(row["args_json"]) if row["args_json"] else {},
            "result": json.loads(row["result_json"]) if row["result_json"] else {},
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def load_run(run_id: int, *, include_events: bool = False) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["plan"] = json.loads(result.pop("plan_json")) if result["plan_json"] else None
    if include_events:
        result["events"] = load_events(run_id)
    return result


def load_run_detail(run_id: int) -> dict | None:
    """Return the established HTTP run-detail shape with decoded plan_json."""
    with _conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["plan_json"] = (
            json.loads(result["plan_json"]) if result["plan_json"] else None
        )
        result["events"] = [
            dict(event)
            for event in conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY step, id",
                (run_id,),
            ).fetchall()
        ]
    return result


def aggregate_stats() -> dict:
    with _conn() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN exit_code = 0 THEN 1 ELSE 0 END) AS passed,
                AVG(token_count) AS avg_tokens,
                AVG(wall_seconds) AS avg_wall
            FROM runs
            """
        ).fetchone()
    total = int(row["total"] or 0)
    passed = int(row["passed"] or 0)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_tokens": round(float(row["avg_tokens"] or 0.0), 1),
        "avg_wall_seconds": round(float(row["avg_wall"] or 0.0), 3),
    }
