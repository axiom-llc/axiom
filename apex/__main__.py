"""APEX command-line entry point."""
import argparse
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from apex.config import load_config
from apex.core.loop import run
from apex.core.state import format_output
from apex.core.toolloader import build_registry


def _version() -> str:
    try:
        return version("axiom-apex")
    except PackageNotFoundError:
        return "0+local"


def _history(argv: list[str]) -> None:
    from apex.history import list_runs

    parser = argparse.ArgumentParser(prog="apex history")
    parser.add_argument("-n", type=int, default=20, help="Number of runs to show")
    args = parser.parse_args(argv)
    rows = list_runs(args.n)
    if not rows:
        print("no runs recorded")
        return
    print("{:>5}  {:>4}  {:>7}  {:>8}  {:>19}  TASK".format("ID", "EXIT", "TOKENS", "WALL(s)", "TIMESTAMP"))
    for row in rows:
        task = row["task"][:60] + ("..." if len(row["task"]) > 60 else "")
        print(
            f'{row["id"]:>5}  {row["exit_code"]:>4}  {row["token_count"]:>7}  '
            f'{row["wall_seconds"]:>8.3f}  {row["timestamp"]:>19}  {task}'
        )


def _swarm(argv: list[str]) -> int:
    from apex.core.swarm import run_swarm

    parser = argparse.ArgumentParser(prog="apex swarm")
    parser.add_argument("--tasks", required=True, help="Path to JSON task list")
    parser.add_argument("--workers", type=int, default=4, help="Maximum parallel workers")
    parser.add_argument("--human-loop", action="store_true", help="Confirm each batch")
    parser.add_argument("--trace-path", default=None, help="JSONL trace path for each worker")
    args = parser.parse_args(argv)

    if args.workers <= 0:
        parser.error("--workers must be greater than 0")
    try:
        tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot load tasks: {exc}")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) and task.strip() for task in tasks):
        parser.error("tasks file must contain a non-empty JSON array of non-empty strings")

    db_path = Path(
        os.environ.get("APEX_DB_PATH", str(Path.home() / ".apex" / "memory.db"))
    ).expanduser()
    extra_args = ["--full-trace", "--trace-path", args.trace_path] if args.trace_path else []
    return run_swarm(
        tasks,
        workers=args.workers,
        human_loop=args.human_loop,
        db_path=db_path,
        apex_cmd=[sys.executable, "-m", "apex"],
        extra_args=extra_args,
    )


def _dispatch_subcommand(argv: list[str]) -> bool:
    if not argv:
        return False
    command = argv[0]
    rest = argv[1:]
    if command == "history":
        _history(rest)
    elif command == "stats":
        from apex.history import aggregate_stats

        print(json.dumps(aggregate_stats(), indent=2))
    elif command == "rsi":
        from apex.rsi import rsi_main

        rsi_main(rest)
    elif command == "export":
        from apex.export import export_main

        export_main(rest)
    elif command == "replay":
        from apex.replay import replay_main

        replay_main(rest)
    elif command == "serve":
        from apex.server import serve_main

        serve_main(rest)
    elif command == "swarm":
        raise SystemExit(0 if _swarm(rest) == 0 else 1)
    else:
        return False
    return True


def main() -> None:
    if _dispatch_subcommand(sys.argv[1:]):
        return

    parser = argparse.ArgumentParser(
        prog="apex",
        description="APEX — Agentic Process Executor",
    )
    parser.add_argument("task", nargs="*", help="Task description")
    parser.add_argument("--dry-run", action="store_true", help="Print plan JSON without executing")
    parser.add_argument("--trace", action="store_true", help="Log execution steps to stderr")
    parser.add_argument("--full-trace", action="store_true", help="Write structured JSONL trace events")
    parser.add_argument("--trace-path", default=None, help="JSONL trace destination; default stderr")
    parser.add_argument("--paranoid", action="store_true", help="Audit the plan before execution")
    parser.add_argument("--interactive", "-i", action="store_true", help="Enter interactive prompt mode")
    parser.add_argument("--version", action="version", version=f"apex {_version()}")
    args = parser.parse_args()

    try:
        config = load_config(
            trace=args.trace,
            dry_run=args.dry_run,
            full_trace=args.full_trace,
            trace_path=Path(args.trace_path).expanduser() if args.trace_path else None,
            paranoid=args.paranoid,
        )
        registry = build_registry(config.db_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.interactive:
        print(f"APEX {_version()} — interactive mode. Ctrl-D or 'exit' to quit.")
        while True:
            try:
                task = input("apex> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not task:
                continue
            if task.lower() in ("exit", "quit"):
                break
            state = run(task, config=config, registry=registry)
            print(format_output(state))
        return

    if not args.task:
        parser.print_help()
        raise SystemExit(1)

    state = run(" ".join(args.task), config=config, registry=registry)
    print(format_output(state))
    raise SystemExit({"HALTED": 0, "ERROR": 1}.get(state.status, 2))


if __name__ == "__main__":
    main()
