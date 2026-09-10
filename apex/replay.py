"""Replay recorded APEX plans without replanning them."""
import json

from apex.history import load_events, load_run


def _recorded_plan(record: dict, events: list[dict]) -> dict:
    raw = record["plan"]
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, list):
        raise ValueError("recorded plan has an unsupported format")

    # Older APEX runs stored the remaining plan after execution. Reconstruct
    # missing tool calls only from their recorded events; do not invent args.
    tool_steps = [step for step in raw if isinstance(step, dict) and step.get("type") in ("tool", "toolcall")]
    if len(tool_steps) < len(events):
        rebuilt = [
            {"type": "tool", "name": event["tool"], "args": event["args"]}
            for event in events
        ]
        halt = next(
            (
                step
                for step in reversed(raw)
                if isinstance(step, dict) and step.get("type") == "halt"
            ),
            {"type": "halt", "reason": "recorded run complete"},
        )
        raw = [*rebuilt, halt]
    return {"goal": record["task"], "steps": raw}


def _result_dict(event) -> dict:
    from apex.core.types import Ok

    return event.result.value if isinstance(event.result, Ok) else {"error": event.result.message}


def replay_main(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="apex replay")
    parser.add_argument("run_id", type=int, help="Run ID from apex history")
    parser.add_argument("--mode", choices=["live", "dry", "simulate"], default="simulate")
    parser.add_argument("--diff", action="store_true", help="Compare live outputs with recorded outputs")
    parser.add_argument("--no-write", action="store_true", help="Reject plans that require write_file")
    args = parser.parse_args(argv)

    record = load_run(args.run_id)
    if record is None:
        raise SystemExit(f"run {args.run_id} not found")
    if record["plan"] is None:
        raise SystemExit(f"run {args.run_id} has no recorded plan")

    events = load_events(args.run_id)
    plan_data = _recorded_plan(record, events)
    print(f"[replay] run_id={args.run_id} task={record['task']!r} mode={args.mode}")

    if args.mode == "dry":
        print(json.dumps(plan_data, indent=2))
        return

    if args.mode == "simulate":
        print(f"[replay] {len(events)} recorded tool event(s)")
        for event in events:
            print(
                f"  step {event['step']:>3}  [{event['tool']}]  "
                f"recorded={json.dumps(event['result'], ensure_ascii=False)}"
            )
        print("[replay] simulate complete — no live execution performed")
        return

    from apex.config import load_config
    from apex.core.loop import run_plan
    from apex.core.planner import parse_plan
    from apex.core.state import format_output
    from apex.core.toolloader import build_registry
    from apex.core.types import Err, ToolExecution

    config = load_config(require_api_key=False)
    registry = build_registry(config.db_path)
    if args.no_write:
        registry.pop("write_file", None)

    parsed = parse_plan(json.dumps(plan_data), registry)
    if isinstance(parsed, Err):
        raise SystemExit(f"recorded plan is not executable: {parsed.message}")

    state = run_plan(record["task"], parsed, config=config, registry=registry)
    print(format_output(state))

    if args.diff:
        live = [event for event in state.history if isinstance(event, ToolExecution)]
        print("\n[diff] step comparison:")
        count = max(len(events), len(live))
        for index in range(count):
            recorded = events[index] if index < len(events) else None
            actual = live[index] if index < len(live) else None
            recorded_result = recorded["result"] if recorded else None
            live_result = _result_dict(actual) if actual else None
            tool = actual.tool if actual else (recorded["tool"] if recorded else "?")
            print(
                f"  step {index:>3}  [{tool}]  match={recorded_result == live_result} "
                f"recorded={json.dumps(recorded_result, ensure_ascii=False)} "
                f"live={json.dumps(live_result, ensure_ascii=False)}"
            )
