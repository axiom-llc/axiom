"""Bounded plan generation and tool execution."""
import json
import signal
import sys
from contextlib import contextmanager
from dataclasses import replace
from time import sleep, time

from apex.config import Config
from apex.core.planner import generate_plan
from apex.core.state import State, create_initial_state
from apex.core.trace import write_event
from apex.core.types import Err, Halt, Ok, Plan, Tool, ToolCall, ToolExecution, plan_to_dict
from apex.history import record_run
from apex.safety import audit_plan, format_audit_report

_MAX_OUTPUT_BYTES = 10_485_760
_TOOL_TIMEOUT_S = 300
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2


class ApexTimeoutError(Exception):
    pass


@contextmanager
def _timeout(seconds: int):
    def _handler(signum, frame):
        del signum, frame
        raise ApexTimeoutError()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _trace(config: Config, message: str) -> None:
    if config.trace:
        print(message, file=sys.stderr)


def _full_trace(config: Config, event: dict) -> None:
    if config.full_trace:
        write_event(event, dest=config.trace_path)


def _finish(
    state: State,
    *,
    task: str,
    wall_start: float,
    events: list[dict],
) -> State:
    run_id = record_run(
        task=task,
        plan=plan_to_dict(state.plan) if state.plan else None,
        exit_code=0 if state.status == "HALTED" else 1,
        token_count=state.token_count,
        wall_seconds=round(time() - wall_start, 3),
        events=events,
    )
    return replace(state, run_id=run_id)


def _audit(state: State, config: Config) -> State:
    if not config.paranoid or state.plan is None:
        return state
    try:
        audit = audit_plan(state.plan, api_key=config.api_key)
        report = format_audit_report(audit)
        _trace(config, report)
        _full_trace(config, {"event": "paranoid_audit", "result": audit})
        if not audit["safe"]:
            print(report, file=sys.stderr)
            return replace(state, status="ERROR")
        return state
    except Exception as exc:
        _trace(config, f"[paranoid] audit failed: {exc}")
        return replace(state, status="ERROR")


def _normalize_output(tool_name: str, output) -> Ok | Err:
    value = output if isinstance(output, dict) else {"output": output}
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return Err("ToolOutputError", f"{tool_name} returned non-JSON output: {exc}")
    if len(encoded) > _MAX_OUTPUT_BYTES:
        return Err("ToolOutputError", f"{tool_name} output exceeds 10 MiB")
    return Ok(value)


def _execute(state: State, config: Config, registry: dict[str, Tool], events: list[dict]) -> State:
    if state.status != "RUNNING" or state.plan is None:
        return state

    for step_index, step in enumerate(state.plan.steps):
        if isinstance(step, Halt):
            _trace(config, f"[halt] {step.reason}")
            _full_trace(config, {"event": "halt", "reason": step.reason})
            _full_trace(
                config,
                {"event": "run_complete", "status": "HALTED", "tokens": state.token_count},
            )
            return replace(state, status="HALTED")

        if not isinstance(step, ToolCall):
            return replace(state, status="ERROR")

        tool = registry[step.name]
        _trace(config, f"[tool] {step.name} args={step.args}")
        _full_trace(config, {"event": "tool_call", "tool": step.name, "args": step.args})

        result: Ok | Err
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with _timeout(_TOOL_TIMEOUT_S):
                    output = tool.effect(step.args)
                result = _normalize_output(step.name, output)
                break
            except ApexTimeoutError:
                result = Err("ToolTimeout", f"{step.name} exceeded {_TOOL_TIMEOUT_S}s")
            except Exception as exc:
                result = Err("ToolExecutionError", str(exc))

            if attempt < _MAX_RETRIES:
                _full_trace(
                    config,
                    {
                        "event": "tool_retry",
                        "tool": step.name,
                        "attempt": attempt,
                        "reason": result.message,
                    },
                )
                _trace(
                    config,
                    f"[retry] {step.name} attempt {attempt}/{_MAX_RETRIES}: {result.message}",
                )
                sleep(_RETRY_DELAY_S * attempt)

        _trace(
            config,
            f"[result] {'ok' if isinstance(result, Ok) else 'err: ' + result.message}",
        )
        event_result = result.value if isinstance(result, Ok) else {"error": result.message}
        events.append(
            {
                "step": step_index,
                "tool": step.name,
                "args": step.args,
                "result": event_result,
            }
        )
        _full_trace(
            config,
            {
                "event": "tool_result",
                "tool": step.name,
                "status": "ok" if isinstance(result, Ok) else "err",
                "output": result.value if isinstance(result, Ok) else result.message,
            },
        )
        state = replace(
            state,
            history=state.history + (ToolExecution(step.name, step.args, result, time()),),
        )
        if isinstance(result, Err):
            return replace(state, status="ERROR")

    return replace(state, status="ERROR")


def _run_prepared(task: str, state: State, config: Config, registry: dict[str, Tool]) -> State:
    wall_start = time()
    events: list[dict] = []
    _trace(
        config,
        f"[plan] goal={state.plan.goal if state.plan else 'NONE'} status={state.status}",
    )
    _full_trace(
        config,
        {
            "event": "plan",
            "goal": state.plan.goal if state.plan else None,
            "steps": len(state.plan.steps) if state.plan else 0,
            "status": state.status,
        },
    )
    state = _audit(state, config)
    state = _execute(state, config, registry, events)
    return _finish(state, task=task, wall_start=wall_start, events=events)


def run(input_str: str, config: Config, registry: dict[str, Tool]) -> State:
    """Generate, validate, execute, trace, and record a plan for input_str."""
    wall_start = time()
    state = generate_plan(create_initial_state(input_str), config, registry)

    if config.dry_run:
        if state.plan:
            print(json.dumps(plan_to_dict(state.plan), indent=2))
            state = replace(state, status="HALTED")
        return _finish(state, task=input_str, wall_start=wall_start, events=[])

    _trace(
        config,
        f"[plan] goal={state.plan.goal if state.plan else 'NONE'} status={state.status}",
    )
    _full_trace(
        config,
        {
            "event": "plan",
            "goal": state.plan.goal if state.plan else None,
            "steps": len(state.plan.steps) if state.plan else 0,
            "status": state.status,
        },
    )
    events: list[dict] = []
    state = _audit(state, config)
    state = _execute(state, config, registry, events)
    return _finish(state, task=input_str, wall_start=wall_start, events=events)


def run_plan(input_str: str, plan: Plan, config: Config, registry: dict[str, Tool]) -> State:
    """Execute an already validated plan exactly, without generating a new plan."""
    state = replace(create_initial_state(input_str), plan=plan)
    return _run_prepared(input_str, state, config, registry)
