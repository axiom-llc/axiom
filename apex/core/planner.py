"""Plan generation, rendering, and schema validation."""
import json
import os
from dataclasses import replace
from importlib import resources
from time import time
from typing import Any

from apex.config import Config
from apex.core.state import State
from apex.core.types import Err, ErrorEvent, Halt, Plan, PlanGeneration, Tool, ToolCall
from apex.llm import gemini_complete

_MAX_STEPS = 32


def _load_prompt() -> str:
    return resources.files("apex").joinpath("prompt.txt").read_text(encoding="utf-8")


def _type_name(value_type: type) -> str:
    return {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
        object: "any JSON value",
    }.get(value_type, value_type.__name__)


def _describe_tool(tool: Tool) -> str:
    required = tool.required_args
    args = ", ".join(
        f"{name}: {_type_name(value_type)}{'' if name in required else ' (optional)'}"
        for name, value_type in tool.input_spec.items()
    )
    returns = ", ".join(
        f"{name}: {_type_name(value_type)}" for name, value_type in tool.output_spec.items()
    )
    return f"- {tool.name}: args {{{args}}}; returns {{{returns}}}"


def render_prompt(task: str, registry: dict[str, Tool]) -> str:
    task = task.replace("~", os.path.expanduser("~"))
    tools_desc = "\n".join(_describe_tool(registry[name]) for name in sorted(registry))
    return (
        f"{_load_prompt()}\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        f"Task: {task}\n\n"
        "Return the plan as raw JSON."
    )


def _extract_json(response_text: str) -> Any:
    text = response_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline < 0:
            raise ValueError("Malformed fenced response")
        text = text[first_newline + 1 :]
        closing = text.rfind("```")
        if closing >= 0:
            text = text[:closing]
        text = text.strip()

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in response")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    return data


def _matches_type(value: Any, expected: type) -> bool:
    if expected is object:
        return value is None or isinstance(value, (str, int, float, bool, dict, list))
    if expected is int:
        return type(value) is int
    if expected is float:
        return type(value) in (int, float) and not isinstance(value, bool)
    if expected is bool:
        return type(value) is bool
    return isinstance(value, expected)


def _validate_args(index: int, tool: Tool, args: Any) -> Err | None:
    if not isinstance(args, dict):
        return Err("ValidationError", f"Step {index}: args must be an object")

    unknown = sorted(set(args) - set(tool.input_spec))
    if unknown:
        return Err(
            "ValidationError",
            f"Step {index}: unexpected argument(s) for {tool.name}: {', '.join(unknown)}",
        )

    missing = sorted(tool.required_args - set(args))
    if missing:
        return Err(
            "ValidationError",
            f"Step {index}: missing argument(s) for {tool.name}: {', '.join(missing)}",
        )

    for name, value in args.items():
        expected = tool.input_spec[name]
        if not _matches_type(value, expected):
            return Err(
                "ValidationError",
                f"Step {index}: argument {name!r} for {tool.name} must be {_type_name(expected)}",
            )
    return None


def parse_plan(response_text: str, registry: dict[str, Tool]) -> Plan | Err:
    """Parse and fully validate plan JSON against the active tool registry."""
    try:
        data = _extract_json(response_text)
    except (json.JSONDecodeError, ValueError) as exc:
        return Err("ParseError", str(exc))

    if not isinstance(data, dict):
        return Err("ParseError", "Plan must be a JSON object")

    goal = data.get("goal")
    raw_steps = data.get("steps")
    if not isinstance(goal, str) or not goal.strip():
        return Err("ValidationError", "Plan goal must be a non-empty string")
    if not isinstance(raw_steps, list) or not raw_steps:
        return Err("ValidationError", "Plan steps must be a non-empty array")
    if len(raw_steps) > _MAX_STEPS:
        return Err(
            "ValidationError",
            f"Plan has {len(raw_steps)} steps; maximum is {_MAX_STEPS}",
        )

    steps: list[Halt | ToolCall] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            return Err("ParseError", f"Step {index}: step must be an object")

        step_type = str(raw.get("type", "")).lower()
        if step_type == "halt":
            if index != len(raw_steps) - 1:
                return Err("ValidationError", f"Step {index}: halt must be the final step")
            reason = raw.get("reason", "Complete")
            if not isinstance(reason, str):
                return Err("ValidationError", f"Step {index}: halt reason must be a string")
            steps.append(Halt(reason=reason))
            continue

        if step_type not in ("tool", "toolcall"):
            return Err("ParseError", f"Step {index}: invalid type {step_type!r}")

        name = raw.get("name")
        if not isinstance(name, str) or name not in registry:
            return Err("ValidationError", f"Step {index}: unknown tool {name!r}")

        args = raw.get("args", {})
        error = _validate_args(index, registry[name], args)
        if error:
            return error
        steps.append(ToolCall(name=name, args=args))

    if not isinstance(steps[-1], Halt):
        return Err("ValidationError", "Plan must end with a halt step")

    return Plan(goal=goal.strip(), steps=tuple(steps))


def generate_plan(state: State, config: Config, registry: dict[str, Tool]) -> State:
    prompt = render_prompt(state.input, registry)
    response = gemini_complete(prompt, api_key=config.api_key)
    if response.get("error"):
        result: Plan | Err = Err("ProviderError", str(response["error"]))
    else:
        result = parse_plan(response.get("text") or "", registry)

    if isinstance(result, Plan):
        tokens = int(response.get("tokens") or 0)
        return replace(
            state,
            plan=result,
            history=state.history + (PlanGeneration(tokens, time()),),
            token_count=state.token_count + tokens,
        )

    return replace(
        state,
        status="ERROR",
        history=state.history + (ErrorEvent(result.error_type, result.message, time()),),
    )
