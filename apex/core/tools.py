"""Built-in tool implementations."""
import subprocess
from pathlib import Path

import requests

from apex.core.types import Tool


def shell_effect(args: dict) -> dict:
    result = subprocess.run(args["cmd"], shell=True, capture_output=True, text=True)
    return {"stdout": result.stdout, "stderr": result.stderr, "code": result.returncode}


def read_file_effect(args: dict) -> dict:
    return {"content": Path(args["path"]).read_text(encoding="utf-8")}


def write_file_effect(args: dict) -> dict:
    path = Path(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = args["content"].encode("utf-8")
    path.write_bytes(data)
    return {"bytes_written": len(data)}


def http_get_effect(args: dict) -> dict:
    response = requests.get(args["url"], headers=args.get("headers", {}), timeout=30)
    return {"body": response.text, "status": response.status_code}


SHELL = Tool(
    name="shell",
    input_spec={"cmd": str},
    output_spec={"stdout": str, "stderr": str, "code": int},
    effect=shell_effect,
)

READ_FILE = Tool(
    name="read_file",
    input_spec={"path": str},
    output_spec={"content": str},
    effect=read_file_effect,
)

WRITE_FILE = Tool(
    name="write_file",
    input_spec={"path": str, "content": str},
    output_spec={"bytes_written": int},
    effect=write_file_effect,
)

HTTP_GET = Tool(
    name="http_get",
    input_spec={"url": str, "headers": dict},
    output_spec={"body": str, "status": int},
    effect=http_get_effect,
    required=frozenset({"url"}),
)


def _rag_multi_query_effect(args: dict) -> dict:
    import json
    import os
    import urllib.request

    question = args["question"]
    hops = max(1, min(int(args.get("hops", 2)), 5))
    base_url = os.environ.get("RAG_BASE_URL", "http://localhost:8000").rstrip("/")
    token = os.environ.get("RAG_API_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    all_sources: list[str] = []
    hop_summaries: list[str] = []
    current_question = question

    for hop in range(hops):
        payload = json.dumps({"question": current_question}).encode()
        request = urllib.request.Request(
            f"{base_url}/query",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read())
        except Exception as exc:
            raise RuntimeError(f"RAG request failed at hop {hop + 1}: {exc}") from exc

        answer = data.get("answer", "")
        sources = data.get("sources", [])
        all_sources.extend(source for source in sources if source not in all_sources)
        hop_summaries.append(f"[hop {hop + 1}] Q: {current_question}\nA: {answer}")

        if hop < hops - 1:
            questions = [line.strip() for line in answer.splitlines() if "?" in line]
            current_question = questions[0] if questions else question

    return {
        "answer": hop_summaries[-1].split("A: ", 1)[-1] if hop_summaries else "",
        "hops": hops,
        "hop_summaries": hop_summaries,
        "sources": all_sources,
    }


RAG_MULTI_QUERY = Tool(
    name="rag_multi_query",
    input_spec={"question": str, "hops": int},
    output_spec={"answer": str, "hops": int, "hop_summaries": list, "sources": list},
    effect=_rag_multi_query_effect,
    required=frozenset({"question"}),
)
