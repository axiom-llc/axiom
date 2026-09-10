"""Parallel subprocess task dispatch with an optional human gate."""
import sqlite3
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from time import time


@contextmanager
def _conn(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(db_path: Path) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS swarm_runs (
                run_id  TEXT NOT NULL,
                task    TEXT NOT NULL,
                status  TEXT NOT NULL,
                pid     INTEGER,
                ts      REAL NOT NULL
            )
            """
        )


def _record(db_path: Path, run_id: str, task: str, status: str, pid: int | None) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO swarm_runs (run_id, task, status, pid, ts) VALUES (?, ?, ?, ?, ?)",
            (run_id, task, status, pid, time()),
        )


def run_swarm(
    tasks: list[str],
    *,
    workers: int = 4,
    human_loop: bool = False,
    db_path: Path,
    apex_cmd: list[str],
    extra_args: list[str] | None = None,
) -> int:
    """Dispatch task batches and return the number of failed subprocesses."""
    if workers <= 0:
        raise ValueError("workers must be greater than 0")
    _ensure_schema(db_path)
    run_id = str(uuid.uuid4())
    extra_args = extra_args or []
    failures = 0
    dispatched = 0

    for start in range(0, len(tasks), workers):
        batch = tasks[start : start + workers]
        if human_loop:
            print(f"\n[swarm] Next batch ({len(batch)} task(s)):")
            for index, task in enumerate(batch, 1):
                print(f"  {index}. {task}")
            try:
                response = input(
                    "[swarm] Press Enter to dispatch, or 'skip' to skip batch: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[swarm] Aborted.")
                break
            if response == "skip":
                for task in batch:
                    _record(db_path, run_id, task, "skipped", None)
                continue

        processes: list[tuple[str, subprocess.Popen]] = []
        for task in batch:
            command = [*apex_cmd, *extra_args, task]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _record(db_path, run_id, task, "running", process.pid)
            processes.append((task, process))
            dispatched += 1

        for task, process in processes:
            stdout, stderr = process.communicate()
            status = "ok" if process.returncode == 0 else "error"
            _record(db_path, run_id, task, status, process.pid)
            print(f"\n[swarm:{status}] {task}")
            if stdout.strip():
                print(stdout.strip())
            if stderr.strip():
                print(stderr.strip(), file=sys.stderr)
            if process.returncode != 0:
                failures += 1

    print(f"\n[swarm] run_id={run_id} dispatched={dispatched} failures={failures}")
    return failures
