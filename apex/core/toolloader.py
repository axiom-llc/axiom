"""Load built-in, user-defined, and MCP tools into one registry."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from apex.core.types import Tool


def load_tools_dir(directory: Path) -> dict[str, Tool]:
    """Import module-level Tool instances from sorted Python files in a directory."""
    tools: dict[str, Tool] = {}
    if not directory.exists():
        return tools

    for py_file in sorted(directory.glob("*.py")):
        module_name = f"_apex_user_tool_{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[toolloader] skipping {py_file.name}: {exc}", file=sys.stderr)
            continue
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, Tool):
                tools[obj.name] = obj
    return tools


def _mcp_configs_from_env() -> list[dict]:
    raw = os.environ.get("APEX_MCP_SERVERS", "[]")
    try:
        configs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid APEX_MCP_SERVERS JSON: {exc}") from exc
    if not isinstance(configs, list) or not all(isinstance(item, dict) for item in configs):
        raise ValueError("APEX_MCP_SERVERS must be a JSON array of objects")
    return configs


def build_registry(
    db_path: Path,
    *,
    tools_dir: Path | None = None,
    mcp_configs: list[dict] | None = None,
) -> dict[str, Tool]:
    """Build the complete runtime registry using the project-wide defaults."""
    from apex.core.memory_store import make_memory_tools
    from apex.core.tools import HTTP_GET, RAG_MULTI_QUERY, READ_FILE, SHELL, WRITE_FILE
    from apex.mcp import load_mcp_servers

    memory_read, memory_write = make_memory_tools(db_path)
    registry: dict[str, Tool] = {
        "shell": SHELL,
        "read_file": READ_FILE,
        "write_file": WRITE_FILE,
        "http_get": HTTP_GET,
        "rag_multi_query": RAG_MULTI_QUERY,
        "memory_read": memory_read,
        "memory_write": memory_write,
    }
    registry.update(load_tools_dir(tools_dir or Path.home() / ".apex" / "tools"))

    configs = _mcp_configs_from_env() if mcp_configs is None else mcp_configs
    if configs:
        registry.update(load_mcp_servers(configs))
    return registry
