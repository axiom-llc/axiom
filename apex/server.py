"""Single-threaded APEX HTTP API server."""
from __future__ import annotations

import contextlib
import hmac
import io
import ipaddress
import os
import sys
from functools import wraps

from flask import Flask, Response, jsonify, request

from apex.config import load_config
from apex.core.loop import run
from apex.core.state import format_output
from apex.core.toolloader import build_registry
from apex.core.types import ToolExecution, plan_to_dict
from apex.history import list_runs, load_run_detail

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("axiom-apex")
except Exception:
    _VERSION = "unknown"

_API_KEY: str | None = os.environ.get("APEX_API_KEY") or None

app = Flask(__name__)
_REGISTRY = {}
_BASE_CONFIG = None


def _require_auth(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if _API_KEY is None:
            return function(*args, **kwargs)
        supplied = request.headers.get("X-Apex-Key", "")
        if not hmac.compare_digest(supplied, _API_KEY):
            return jsonify({"error": "unauthorized"}), 401
        return function(*args, **kwargs)

    return wrapper


def _json_body() -> dict | None:
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else None


def _capture_main(function, argv: list[str]) -> tuple[str, int]:
    """Capture stdout from a command-style function and normalize its exit code."""
    buffer = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(buffer):
            function(argv)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            code = exc.code
        else:
            code = 1
            if exc.code:
                buffer.write(str(exc.code) + "\n")
    except Exception as exc:
        return str(exc), 1
    return buffer.getvalue(), code


def _loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": _VERSION})


@app.route("/run", methods=["POST"])
@_require_auth
def api_run():
    body = _json_body()
    if body is None:
        return jsonify({"error": "JSON object body is required"}), 400
    task = body.get("task")
    if not isinstance(task, str) or not task.strip():
        return jsonify({"error": "task is required"}), 400

    state = run(task.strip(), config=_BASE_CONFIG, registry=_REGISTRY)
    exit_code = {"HALTED": 0, "ERROR": 1}.get(state.status, 2)
    return jsonify(
        {
            "run_id": state.run_id,
            "plan": plan_to_dict(state.plan) if state.plan else None,
            "exit_code": exit_code,
            "status": state.status,
            "output": format_output(state),
            "token_count": state.token_count,
            "step_count": sum(isinstance(event, ToolExecution) for event in state.history),
        }
    )


@app.route("/runs", methods=["GET"])
@_require_auth
def api_runs():
    return jsonify(list_runs(request.args.get("n", 20, type=int)))


@app.route("/runs/<int:run_id>", methods=["GET"])
@_require_auth
def api_run_detail(run_id: int):
    result = load_run_detail(run_id)
    if result is None:
        return jsonify({"error": "run not found"}), 404
    return jsonify(result)


@app.route("/replay", methods=["POST"])
@_require_auth
def api_replay():
    body = _json_body()
    if body is None:
        return jsonify({"error": "JSON object body is required"}), 400
    run_id = body.get("run_id")
    mode = body.get("mode", "simulate")
    if type(run_id) is not int:
        return jsonify({"error": "run_id is required"}), 400
    if mode not in ("simulate", "dry", "live"):
        return jsonify({"error": "mode must be simulate|dry|live"}), 400

    from apex.replay import replay_main

    text, code = _capture_main(replay_main, [str(run_id), "--mode", mode])
    return jsonify({"run_id": run_id, "mode": mode, "exit_code": code, "output": text})


@app.route("/export", methods=["GET"])
@_require_auth
def api_export():
    output_format = request.args.get("format", "jsonl")
    if output_format not in ("csv", "jsonl"):
        return jsonify({"error": "format must be csv|jsonl"}), 400

    from apex.export import export_main

    argv = ["--format", output_format]
    if request.args.get("since"):
        argv += ["--since", request.args["since"]]
    if request.args.get("fields"):
        argv += ["--fields", request.args["fields"]]
    if request.args.get("events", "false").lower() == "true":
        argv.append("--events")

    text, code = _capture_main(export_main, argv)
    if code != 0:
        return jsonify({"error": text.strip()}), 500
    mime = "application/x-ndjson" if output_format == "jsonl" else "text/csv"
    return Response(text, mimetype=mime)


def serve_main(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="apex serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    if not _loopback_host(args.host) and _API_KEY is None:
        print(
            "Error: refuse unauthenticated non-loopback bind; set APEX_API_KEY or use a loopback host",
            file=sys.stderr,
        )
        raise SystemExit(1)

    global _BASE_CONFIG, _REGISTRY
    try:
        _BASE_CONFIG = load_config()
        _REGISTRY = build_registry(_BASE_CONFIG.db_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if _API_KEY is None:
        print("Auth: disabled on loopback-only server", file=sys.stderr)
    else:
        print("Auth: X-Apex-Key required", file=sys.stderr)
    print(f"apex serve {_VERSION}  http://{args.host}:{args.port}", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
