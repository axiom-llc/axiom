"""Retry only tools whose owner explicitly guarantees safe repeated execution."""
from dataclasses import replace
from unittest.mock import Mock

import pytest

from apex.config import load_config
from apex.core import loop
from apex.core.state import create_initial_state
from apex.core.types import Halt, Plan, Tool, ToolCall


@pytest.mark.parametrize("retry_safe,expected", [(False, 1), (True, 3)])
@pytest.mark.parametrize("error", [RuntimeError("failed after effect"), loop.ApexTimeoutError()])
def test_ambiguous_failure_retries_only_explicitly_safe_tools(monkeypatch, retry_safe, expected, error):
    effects = []

    def effect(args):
        effects.append("side effect")
        raise error

    tool = Tool("operation", {}, {}, effect, retry_safe=retry_safe)
    state = replace(create_initial_state("task"), plan=Plan("task", (ToolCall("operation", {}), Halt("done"))))
    monkeypatch.setattr(loop, "sleep", Mock())
    result = loop._execute(state, load_config(require_api_key=False), {"operation": tool}, [])
    assert result.status == "ERROR"
    assert len(effects) == expected


def test_successful_retry_safe_tool_executes_once():
    effect = Mock(return_value={"ok": True})
    tool = Tool("operation", {}, {}, effect, retry_safe=True)
    state = replace(create_initial_state("task"), plan=Plan("task", (ToolCall("operation", {}), Halt("done"))))
    result = loop._execute(state, load_config(require_api_key=False), {"operation": tool}, [])
    assert result.status == "HALTED"
    effect.assert_called_once()
