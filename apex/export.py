"""apex export — dump runs.db to CSV or JSONL"""
import csv
import json
import sqlite3
import sys
from apex.history import DB_PATH

_DEFAULT_FIELDS = ["id", "task", "exit_code", "token_count", "wall_seconds", "timestamp"]


def _load_runs(since: str | None) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    q = "SELECT id, task, exit_code, token_count, wall_seconds, timestamp FROM runs"
    params = []
    if since:
        q += " WHERE timestamp >= ?"
        params.append(since)
    q += " ORDER BY id ASC"
    rows = con.execute(q, params).fetchall()
    con.close()
    return [dict(zip(_DEFAULT_FIELDS, r)) for r in rows]


def _load_events(run_id: int) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT step, tool, args_json, result_json FROM events WHERE run_id = ? ORDER BY step",
        (run_id,)
    ).fetchall()
    con.close()
    return [{"step": r[0], "tool": r[1],
             "args": json.loads(r[2]) if r[2] else {},
             "result": json.loads(r[3]) if r[3] else {}} for r in rows]


def export_main(argv: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="apex export")
    p.add_argument("--format", choices=["csv", "jsonl"], default="jsonl")
    p.add_argument("--since", default=None, help="ISO timestamp filter e.g. 2026-01-01")
    p.add_argument("--fields", default=None, help="Comma-separated field list")
    p.add_argument("--events", action="store_true", help="Include step-level events")
    p.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    args = p.parse_args(argv)

    fields = args.fields.split(",") if args.fields else _DEFAULT_FIELDS
    runs = _load_runs(args.since)

    out = open(args.output, "w", newline="" if args.format == "csv" else "") if args.output else sys.stdout

    try:
        if args.format == "csv":
            writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for run in runs:
                writer.writerow(run)
        else:  # jsonl
            for run in runs:
                row = {k: run[k] for k in fields if k in run}
                if args.events:
                    row["events"] = _load_events(run["id"])
                out.write(json.dumps(row) + "\n")
    finally:
        if args.output:
            out.close()
