#!/usr/bin/env python3
"""Run the APEX task benchmark and compute a bounded composite score."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_APEX_CMD = [sys.executable, "-m", "apex"]
_TOKEN_RE = re.compile(r"^Tokens:\s*(\d+)\s*$", re.MULTILINE)


def _token_count(output: str) -> int:
    match = _TOKEN_RE.search(output)
    return int(match.group(1)) if match else 0


def run_task(
    task: dict[str, Any],
    *,
    mock: bool,
    trace_path: str | None,
    timeout: int,
    work_dir: str,
) -> dict[str, Any]:
    task_id = task["id"]
    prompt = task["prompt"]
    check = task.get("check")
    check_file = task.get("check_file")

    if check_file and not os.path.isabs(check_file):
        original = check_file
        check_file = str(Path(work_dir) / check_file)
        prompt = prompt.replace(original, check_file)
    if check_file:
        Path(check_file).unlink(missing_ok=True)

    if mock:
        time.sleep(0.05)
        return {
            "id": task_id,
            "duration_seconds": 0.05,
            "exit_code": 0,
            "passed": True,
            "token_count": 0,
            "mock": True,
        }

    command = list(_APEX_CMD)
    if trace_path:
        command += ["--full-trace", "--trace-path", trace_path]
    command.append(prompt)

    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "id": task_id,
            "duration_seconds": float(timeout),
            "exit_code": -1,
            "passed": False,
            "token_count": 0,
            "error": "timeout",
        }
    elapsed = time.perf_counter() - start

    passed = result.returncode == 0
    if passed and check_file:
        path = Path(check_file)
        passed = path.is_file() and path.stat().st_size > 0
        if passed and check:
            passed = check.lower() in path.read_text(encoding="utf-8", errors="replace").lower()
    elif passed and check:
        passed = check.lower() in (result.stdout + result.stderr).lower()

    return {
        "id": task_id,
        "duration_seconds": round(elapsed, 3),
        "exit_code": result.returncode,
        "passed": passed,
        "token_count": _token_count(result.stdout),
        "stdout": result.stdout[:512] if not passed else "",
        "stderr": result.stderr[:512] if not passed else "",
    }


def _score(results: list[dict]) -> tuple[float, float, float, float]:
    pass_rate = sum(result["passed"] for result in results) / len(results)
    average_duration = sum(result["duration_seconds"] for result in results) / len(results)
    speed_factor = min(1.0, max(0.01, 1.0 - (average_duration - 10.0) / 200.0))

    token_counts = [result["token_count"] for result in results if result["token_count"] > 0]
    if token_counts:
        average_tokens = sum(token_counts) / len(token_counts)
        token_efficiency = min(1.0, max(0.01, 1.0 - (average_tokens - 1000.0) / 50000.0))
    else:
        token_efficiency = 1.0
    return (
        round(pass_rate, 4),
        round(speed_factor, 4),
        round(token_efficiency, 4),
        round(pass_rate * speed_factor * token_efficiency, 6),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX benchmark harness")
    parser.add_argument("--tasks", default="benchmarks/tasks.json", help="Benchmark task JSON path")
    parser.add_argument("--mock", action="store_true", help="Skip live APEX calls")
    parser.add_argument("--trace-path", default=None, help="Pass a JSONL trace path to each APEX call")
    parser.add_argument("--out", default=None, help="Write the JSON result to this path")
    parser.add_argument("--timeout", type=int, default=180, help="Per-task timeout in seconds (default: 180)")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    tasks_path = Path(args.tasks)
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: tasks file not found: {tasks_path}", file=sys.stderr)
        raise SystemExit(2)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in {tasks_path}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if not isinstance(tasks, list) or not tasks or not all(
        isinstance(task, dict)
        and isinstance(task.get("id"), str)
        and isinstance(task.get("prompt"), str)
        for task in tasks
    ):
        print("ERROR: tasks file must be a non-empty list of {id, prompt, ...} objects", file=sys.stderr)
        raise SystemExit(2)

    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if not args.mock and provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Use --mock for CI.", file=sys.stderr)
        raise SystemExit(1)

    print(f"[bench] tasks={len(tasks)} mock={args.mock} timeout={args.timeout}s", file=sys.stderr)
    results = []
    wall_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="apex-bench-") as work_dir:
        for index, task in enumerate(tasks, 1):
            print(f"[bench] [{index}/{len(tasks)}] {task['id']} ...", file=sys.stderr)
            result = run_task(
                task,
                mock=args.mock,
                trace_path=args.trace_path,
                timeout=args.timeout,
                work_dir=work_dir,
            )
            results.append(result)
            print(
                f"[bench]   {'PASS' if result['passed'] else 'FAIL'} "
                f"{result['duration_seconds']:.2f}s",
                file=sys.stderr,
            )

    wall_seconds = round(time.perf_counter() - wall_start, 3)
    passed = sum(result["passed"] for result in results)
    failed = len(results) - passed
    pass_rate, speed_factor, token_efficiency, apex_score = _score(results)
    output = {
        "benchmark": "apex_task_harness",
        "mock": args.mock,
        "task_count": len(results),
        "passed": passed,
        "failed": failed,
        "wall_seconds": wall_seconds,
        "pass_rate": pass_rate,
        "speed_factor": speed_factor,
        "token_efficiency": token_efficiency,
        "apex_score": apex_score,
        "results": results,
    }
    json_output = json.dumps(output, indent=2)
    print(json_output)

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output + "\n", encoding="utf-8")
        print(f"[bench] results written to {output_path}", file=sys.stderr)
    print(f"\n[bench] {passed}/{len(results)} passed wall={wall_seconds:.2f}s", file=sys.stderr)
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
