"""MCP 2026-07-28 Streamable HTTP tool adapter."""
from __future__ import annotations

import base64
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import requests

from apex.core.types import Tool

_PROTOCOL_VERSION = "2026-07-28"
_LIST_TIMEOUT = (5, 15)
_CALL_TIMEOUT = (5, 60)

try:
    _CLIENT_VERSION = version("axiom-apex")
except PackageNotFoundError:
    _CLIENT_VERSION = "0+local"


def _meta() -> dict[str, Any]:
    return {
        "io.modelcontextprotocol/protocolVersion": _PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "axiom-apex",
            "version": _CLIENT_VERSION,
        },
    }


def _parse_response(response: requests.Response) -> dict:
    content_type = response.headers.get("Content-Type", "").lower()
    messages: list[dict] = []

    if "text/event-stream" in content_type:
        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                value = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                messages.append(value)
    else:
        try:
            value = response.json()
        except ValueError:
            response.raise_for_status()
            raise RuntimeError("MCP server returned a non-JSON response")
        if isinstance(value, dict):
            messages.append(value)

    for message in reversed(messages):
        if message.get("id") == 1 and ("result" in message or "error" in message):
            if "error" in message:
                error = message["error"]
                raise RuntimeError(
                    f"MCP error {error.get('code', '?')}: {error.get('message', 'unknown error')}"
                )
            result = message.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("MCP server returned an invalid result")
            return result

    response.raise_for_status()
    raise RuntimeError("MCP server returned no matching JSON-RPC response")


def _rpc(
    endpoint: str,
    method: str,
    params: dict[str, Any],
    *,
    timeout,
    name: str | None = None,
    base_headers: dict[str, str] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict:
    body_params = {**params, "_meta": _meta()}
    headers = {
        **(base_headers or {}),
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": _PROTOCOL_VERSION,
        "Mcp-Method": method,
    }
    if name is not None:
        headers["Mcp-Name"] = name
    headers.update(extra_headers or {})

    response = requests.post(
        endpoint,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": body_params},
        headers=headers,
        timeout=timeout,
    )
    return _parse_response(response)


def _valid_header_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and all(0x21 <= ord(char) <= 0x7E and char != ":" for char in value)
    )


def _header_bindings(schema: dict) -> dict[str, str] | None:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return {}

    bindings: dict[str, str] = {}
    seen: set[str] = set()
    for argument, definition in properties.items():
        if not isinstance(definition, dict) or "x-mcp-header" not in definition:
            continue
        header_name = definition["x-mcp-header"]
        value_type = definition.get("type")
        normalized = header_name.lower() if isinstance(header_name, str) else ""
        if (
            not _valid_header_name(header_name)
            or normalized in seen
            or value_type not in {"string", "integer", "number", "boolean"}
        ):
            return None
        seen.add(normalized)
        bindings[str(argument)] = header_name
    return bindings


def _header_value(value: object) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
    else:
        text = str(value)

    unsafe = (
        text.startswith("=?base64?") and text.endswith("?=")
    ) or text != text.strip(" \t") or any(ord(char) < 0x20 or ord(char) > 0x7E for char in text)
    if unsafe:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return f"=?base64?{encoded}?="
    return text


def _spec_from_schema(schema: dict) -> dict[str, type]:
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return {}
    result: dict[str, type] = {}
    for name, definition in properties.items():
        raw_type = definition.get("type") if isinstance(definition, dict) else None
        if isinstance(raw_type, list):
            raw_type = next((item for item in raw_type if item != "null"), None)
        result[str(name)] = type_map.get(raw_type, object)
    return result


def _custom_headers(bindings: dict[str, str], args: dict) -> dict[str, str]:
    headers: dict[str, str] = {}
    for argument, header_name in bindings.items():
        value = args.get(argument)
        if argument in args and value is not None:
            headers[f"Mcp-Param-{header_name}"] = _header_value(value)
    return headers


def _make_effect(
    endpoint: str,
    tool_name: str,
    bindings: dict[str, str],
    base_headers: dict[str, str],
):
    def effect(args: dict) -> dict:
        result = _rpc(
            endpoint,
            "tools/call",
            {"name": tool_name, "arguments": args},
            timeout=_CALL_TIMEOUT,
            name=tool_name,
            base_headers=base_headers,
            extra_headers=_custom_headers(bindings, args),
        )
        result_type = result.get("resultType")
        if result_type == "input_required":
            raise RuntimeError("MCP tool requires multi-round-trip client input, which APEX does not implement")
        if result_type == "task":
            raise RuntimeError("MCP tool returned a Tasks-extension handle, which APEX does not implement")

        content = result.get("content", [])
        if not isinstance(content, list):
            content = []
        text = " ".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
        if result.get("isError"):
            raise RuntimeError(text or "MCP tool error")
        return {"output": text, "raw": content}

    return effect


def _list_tools(endpoint: str, base_headers: dict[str, str]) -> list[dict]:
    tools: list[dict] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {}
        if cursor:
            params["cursor"] = cursor
        result = _rpc(
            endpoint,
            "tools/list",
            params,
            timeout=_LIST_TIMEOUT,
            base_headers=base_headers,
        )
        page = result.get("tools", [])
        if not isinstance(page, list):
            raise RuntimeError("MCP tools/list returned an invalid tools field")
        tools.extend(item for item in page if isinstance(item, dict))
        cursor = result.get("nextCursor")
        if not cursor:
            return tools


def load_mcp_servers(server_configs: list[dict]) -> dict[str, Tool]:
    """Load synchronous tools from configured MCP 2026-07-28 HTTP endpoints."""
    registry: dict[str, Tool] = {}
    for config in server_configs:
        server_name = str(config.get("name") or "mcp")
        endpoint = str(config.get("url") or "").rstrip("/")
        base_headers = config.get("headers") or {}
        if not endpoint:
            print(f"[mcp] skipping {server_name!r}: no url", file=sys.stderr)
            continue
        if not isinstance(base_headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in base_headers.items()
        ):
            print(f"[mcp] skipping {server_name!r}: headers must be string pairs", file=sys.stderr)
            continue

        try:
            tool_defs = _list_tools(endpoint, base_headers)
        except Exception as exc:
            print(f"[mcp] skipping {server_name!r} ({endpoint}): {exc}", file=sys.stderr)
            continue

        loaded = 0
        for definition in tool_defs:
            raw_name = definition.get("name")
            schema = definition.get("inputSchema") or {}
            if not isinstance(raw_name, str) or not raw_name or not isinstance(schema, dict):
                continue
            bindings = _header_bindings(schema)
            if bindings is None:
                print(
                    f"[mcp] skipping {server_name!r}/{raw_name}: invalid x-mcp-header schema",
                    file=sys.stderr,
                )
                continue
            required = schema.get("required", [])
            required_names = frozenset(str(name) for name in required) if isinstance(required, list) else frozenset()
            namespaced = f"mcp__{server_name}__{raw_name}"
            registry[namespaced] = Tool(
                name=namespaced,
                input_spec=_spec_from_schema(schema),
                output_spec={"output": str, "raw": list},
                effect=_make_effect(endpoint, raw_name, bindings, base_headers),
                required=required_names,
            )
            loaded += 1
        print(f"[mcp] {server_name!r}: loaded {loaded} tool(s)", file=sys.stderr)
    return registry
