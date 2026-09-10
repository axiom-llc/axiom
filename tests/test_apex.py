"""Core APEX tests; integration tests require the configured live provider."""
import json
import os
import subprocess
import sys
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from apex.config import Config, load_config
from apex.core.planner import parse_plan
from apex.core.safety import static_audit
from apex.core.state import create_initial_state
from apex.core.types import Err, Halt, Plan, Tool, ToolCall, plan_to_dict
from apex.memory import make_memory_tools
from apex.tools import HTTP_GET, READ_FILE, SHELL, WRITE_FILE

APEX_CMD = [sys.executable, "-m", "apex"]


def _run_apex(*args: str, env: dict[str, str] | None = None, timeout: int = 30):
    return subprocess.run(
        [*APEX_CMD, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or os.environ.copy(),
    )


def _registry() -> dict[str, Tool]:
    return {
        "shell": Tool(
            "shell",
            {"cmd": str},
            {"stdout": str, "stderr": str, "code": int},
            lambda args: {},
        ),
        "write_file": Tool(
            "write_file",
            {"path": str, "content": str},
            {"bytes_written": int},
            lambda args: {},
        ),
        "optional": Tool(
            "optional",
            {"required": str, "extra": int},
            {"output": str},
            lambda args: {},
            required=frozenset({"required"}),
        ),
    }


class TestConfig:
    def test_gemini_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            load_config()

    def test_ollama_allows_missing_gemini_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("APEX_DB_PATH", str(tmp_path / "memory.db"))
        config = load_config()
        assert isinstance(config, Config)
        assert config.api_key == ""
        assert config.db_path == tmp_path / "memory.db"
        with pytest.raises(FrozenInstanceError):
            config.api_key = "mutated"  # type: ignore[misc]

    def test_rejects_unknown_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "invalid")
        with pytest.raises(ValueError, match="Unsupported"):
            load_config(require_api_key=False)

    def test_cli_reports_missing_key(self):
        env = {key: value for key, value in os.environ.items() if key != "GEMINI_API_KEY"}
        env["LLM_PROVIDER"] = "gemini"
        result = _run_apex("hello", env=env)
        assert result.returncode != 0
        assert "GEMINI_API_KEY" in result.stdout + result.stderr


class TestPlanSchema:
    def test_valid_plan_round_trip(self):
        raw = {
            "goal": "write hello",
            "steps": [
                {
                    "type": "tool",
                    "name": "write_file",
                    "args": {"path": "/tmp/x", "content": "hello"},
                },
                {"type": "halt", "reason": "done"},
            ],
        }
        plan = parse_plan(json.dumps(raw), _registry())
        assert isinstance(plan, Plan)
        assert plan_to_dict(plan) == raw

    @pytest.mark.parametrize(
        "raw",
        [
            {},
            {"goal": "x"},
            {"steps": [{"type": "halt"}]},
            {"goal": "x", "steps": []},
            {"goal": "x", "steps": [{"type": "unknown"}]},
            {"goal": "x", "steps": [{"type": "tool", "name": "missing", "args": {}}]},
            {"goal": "x", "steps": [{"type": "halt"}, {"type": "halt"}]},
            {"goal": "x", "steps": [{"type": "tool", "name": "shell", "args": {"cmd": "true"}}]},
        ],
    )
    def test_rejects_invalid_structure(self, raw):
        assert isinstance(parse_plan(json.dumps(raw), _registry()), Err)

    @pytest.mark.parametrize(
        "args",
        [
            {},
            {"cmd": 1},
            {"cmd": "true", "unexpected": True},
        ],
    )
    def test_rejects_invalid_tool_args(self, args):
        raw = {
            "goal": "x",
            "steps": [
                {"type": "tool", "name": "shell", "args": args},
                {"type": "halt", "reason": "done"},
            ],
        }
        assert isinstance(parse_plan(json.dumps(raw), _registry()), Err)

    def test_accepts_optional_argument_omission_and_toolcall_alias(self):
        raw = {
            "goal": "x",
            "steps": [
                {"type": "toolcall", "name": "optional", "args": {"required": "x"}},
                {"type": "halt", "reason": "done"},
            ],
        }
        assert isinstance(parse_plan(json.dumps(raw), _registry()), Plan)

    def test_rejects_more_than_32_steps(self):
        steps = [
            {"type": "tool", "name": "shell", "args": {"cmd": "true"}}
            for _ in range(32)
        ] + [{"type": "halt", "reason": "done"}]
        assert isinstance(parse_plan(json.dumps({"goal": "x", "steps": steps}), _registry()), Err)

    def test_rejects_non_json(self):
        assert isinstance(parse_plan("not json", _registry()), Err)


class TestStateAndTools:
    def test_initial_state_is_frozen(self):
        state = create_initial_state("task")
        assert state.status == "RUNNING"
        assert state.plan is None
        assert state.history == ()
        with pytest.raises(FrozenInstanceError):
            state.status = "HALTED"  # type: ignore[misc]

    def test_shell(self):
        result = SHELL.effect({"cmd": "printf apex"})
        assert result == {"stdout": "apex", "stderr": "", "code": 0}

    def test_file_roundtrip_and_utf8_byte_count(self, tmp_path):
        path = tmp_path / "nested" / "x.txt"
        result = WRITE_FILE.effect({"path": str(path), "content": "café"})
        assert result["bytes_written"] == len("café".encode("utf-8"))
        assert READ_FILE.effect({"path": str(path)})["content"] == "café"

    def test_http_get(self):
        response = Mock(text="body", status_code=200)
        with patch("apex.core.tools.requests.get", return_value=response) as request_get:
            result = HTTP_GET.effect({"url": "https://example.invalid"})
        assert result == {"body": "body", "status": 200}
        request_get.assert_called_once_with(
            "https://example.invalid", headers={}, timeout=30
        )


class TestMemory:
    def test_roundtrip_list_and_persistence(self, tmp_path):
        db = tmp_path / "memory.db"
        read, write = make_memory_tools(db)
        write.effect({"key": "a", "value": {"n": 1}})
        assert read.effect({"key": "a"})["value"] == {"n": 1}
        read2, _ = make_memory_tools(db)
        assert read2.effect({})["entries"][0]["key"] == "a"

    def test_concurrent_writes(self, tmp_path):
        db = tmp_path / "memory.db"
        errors: list[Exception] = []

        def write(index: int) -> None:
            try:
                _, tool = make_memory_tools(db)
                tool.effect({"key": f"k{index}", "value": index})
            except Exception as exc:  # pragma: no cover - diagnostic collection
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        read, _ = make_memory_tools(db)
        assert len(read.effect({})["entries"]) == 8


class TestSafety:
    def test_allows_tmp_write(self):
        plan = Plan("x", (ToolCall("write_file", {"path": "/tmp/x", "content": "x"}), Halt("done")))
        assert static_audit(plan)["safe"] is True

    def test_rejects_write_outside_home_and_tmp(self):
        plan = Plan("x", (ToolCall("write_file", {"path": "/etc/apex-x", "content": "x"}), Halt("done")))
        audit = static_audit(plan)
        assert audit["safe"] is False
        assert "outside HOME or /tmp" in audit["findings"][0]["reason"]

    def test_rejects_destructive_shell_pattern(self):
        plan = Plan("x", (ToolCall("shell", {"cmd": "rm -rf /"}), Halt("done")))
        assert static_audit(plan)["safe"] is False


class TestHistoryAndReplay:
    def test_record_run_is_atomic_and_decoded(self, tmp_path, monkeypatch):
        import apex.history as history

        monkeypatch.setattr(history, "DB_PATH", tmp_path / "runs.db")
        plan = {"goal": "x", "steps": [{"type": "halt", "reason": "done"}]}
        run_id = history.record_run(
            task="x",
            plan=plan,
            exit_code=0,
            token_count=7,
            wall_seconds=0.1,
            events=[{"step": 0, "tool": "shell", "args": {"cmd": "true"}, "result": {"code": 0}}],
        )
        record = history.load_run(run_id, include_events=True)
        assert record is not None
        assert record["plan"] == plan
        assert record["events"][0]["args"] == {"cmd": "true"}
        assert record["events"][0]["result"] == {"code": 0}

    def test_legacy_replay_reconstructs_executed_tool_calls(self):
        from apex.replay import _recorded_plan

        record = {
            "task": "legacy",
            "plan": [{"type": "halt", "reason": "done"}],
        }
        events = [{"tool": "shell", "args": {"cmd": "true"}}]
        rebuilt = _recorded_plan(record, events)
        assert rebuilt["steps"] == [
            {"type": "tool", "name": "shell", "args": {"cmd": "true"}},
            {"type": "halt", "reason": "done"},
        ]


class TestMCP:
    def test_modern_rpc_headers_and_meta(self):
        from apex import mcp

        response = Mock()
        response.headers = {"Content-Type": "application/json"}
        response.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
        response.raise_for_status.return_value = None

        with patch("apex.mcp.requests.post", return_value=response) as post:
            result = mcp._rpc(
                "https://example.invalid/mcp",
                "tools/list",
                {},
                timeout=(5, 15),
            )
        assert result == {"tools": []}
        kwargs = post.call_args.kwargs
        assert kwargs["headers"]["MCP-Protocol-Version"] == "2026-07-28"
        assert kwargs["headers"]["Mcp-Method"] == "tools/list"
        meta = kwargs["json"]["params"]["_meta"]
        assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
        assert "io.modelcontextprotocol/clientInfo" in meta
        assert "io.modelcontextprotocol/clientCapabilities" in meta


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("LLM_PROVIDER", "gemini").lower() == "gemini"
    and not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestIntegration:
    def test_dry_run_returns_valid_plan(self):
        result = _run_apex("--dry-run", "write hello to /tmp/apex-test.txt", timeout=60)
        assert result.returncode == 0
        plan, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
        assert plan["steps"][-1]["type"] == "halt"

    def test_write_and_read_task(self, tmp_path):
        path = tmp_path / "integration.txt"
        result = _run_apex(
            f"write 'integration-ok' to {path} and read it back",
            timeout=60,
        )
        assert result.returncode == 0
        assert path.read_text(encoding="utf-8") == "integration-ok"
