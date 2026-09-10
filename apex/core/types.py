"""Core immutable types for APEX."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Halt:
    reason: str


Step = ToolCall | Halt


@dataclass(frozen=True)
class Plan:
    goal: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Ok:
    value: dict[str, Any]


@dataclass(frozen=True)
class Err:
    error_type: str
    message: str


Result = Ok | Err


@dataclass(frozen=True)
class PlanGeneration:
    tokens: int
    timestamp: float


@dataclass(frozen=True)
class ToolExecution:
    tool: str
    args: dict[str, Any]
    result: Result
    timestamp: float


@dataclass(frozen=True)
class ErrorEvent:
    error_type: str
    message: str
    timestamp: float


Event = PlanGeneration | ToolExecution | ErrorEvent


@dataclass(frozen=True)
class Tool:
    """Describe one executable tool and its planner-visible schema."""

    name: str
    input_spec: dict[str, type]
    output_spec: dict[str, type]
    effect: Callable[[dict], dict]
    required: frozenset[str] | None = None

    @property
    def required_args(self) -> frozenset[str]:
        return self.required if self.required is not None else frozenset(self.input_spec)


def plan_to_dict(plan: Plan) -> dict[str, Any]:
    """Serialize a Plan to the stable JSON-compatible representation."""
    steps: list[dict[str, Any]] = []
    for step in plan.steps:
        if isinstance(step, Halt):
            steps.append({"type": "halt", "reason": step.reason})
        else:
            steps.append({"type": "tool", "name": step.name, "args": step.args})
    return {"goal": plan.goal, "steps": steps}
