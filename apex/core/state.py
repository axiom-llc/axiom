"""Immutable execution state."""
from dataclasses import dataclass
from typing import Literal

from apex.core.types import ErrorEvent, Event, Ok, Plan, ToolExecution


@dataclass(frozen=True)
class State:
    input: str
    plan: Plan | None
    history: tuple[Event, ...]
    status: Literal["RUNNING", "HALTED", "ERROR"]
    token_count: int
    run_id: int | None = None


def create_initial_state(input_str: str) -> State:
    return State(
        input=input_str,
        plan=None,
        history=(),
        status="RUNNING",
        token_count=0,
    )


def format_output(state: State) -> str:
    lines = [f"Status: {state.status}", f"Tokens: {state.token_count}"]
    if state.plan:
        executed = sum(isinstance(event, ToolExecution) for event in state.history)
        lines += [f"\nGoal: {state.plan.goal}", f"Steps executed: {executed}"]
    for event in state.history:
        if isinstance(event, ToolExecution):
            lines.append(f"\n[{event.tool}]")
            if isinstance(event.result, Ok):
                lines.append(f"  Success: {event.result.value}")
            else:
                lines.append(f"  Error: {event.result.message}")
        elif isinstance(event, ErrorEvent):
            lines.append(f"\nError: {event.message}")
    return "\n".join(lines)
