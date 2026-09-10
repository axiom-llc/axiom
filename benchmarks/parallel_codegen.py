#!/usr/bin/env python3
"""Compare sequential and subprocess-parallel APEX code-generation tasks."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

TASKS = [
    "write a Python function called 'binary_search' that searches a sorted list",
    "write a Python function called 'flatten_dict' that flattens a nested dictionary",
    "write a Python function called 'retry' that retries a callable N times on exception",
    "write a Python function called 'chunk' that splits a list into chunks of size N",
    "write a Python function called 'memoize' that caches function results by arguments",
    "write a Python function called 'parse_csv_row' that handles quoted fields correctly",
    "write a Python function called 'rate_limit' that enforces calls per second",
    "write a Python function called 'deep_merge' that merges two dicts recursively",
]
_APEX_CMD = [sys.executable, "-m", "apex"]


class TaskResult(NamedTuple):
    label: str
    duration_seconds: float
    exit_code: int
    output_path: str


def _variance(task: str) -> float:
    return (zlib.crc32(task.encode("utf-8")) % 100) / 100.0


def run_real_task(task: str, output_path: str, timeout: int = 120) -> TaskResult:
    prompt = f"{task}. Write only the function (no tests, no explanation) to {output_path}"
    start = time.perf_counter()
    result = subprocess.run(
        [*_APEX_CMD, prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return TaskResult(Path(output_path).name, time.perf_counter() - start, result.returncode, output_path)


def run_mock_task(task: str, output_path: str, latency: float = 2.5) -> TaskResult:
    start = time.perf_counter()
    time.sleep(latency + _variance(task))
    Path(output_path).write_text(f"# mock output for: {task}\ndef stub(): pass\n", encoding="utf-8")
    return TaskResult(Path(output_path).name, time.perf_counter() - start, 0, output_path)


def run_sequential(tasks, work_dir, run_fn) -> tuple[list[TaskResult], float]:
    start = time.perf_counter()
    results = [
        run_fn(task, str(Path(work_dir) / f"task_{index:02d}.py"))
        for index, task in enumerate(tasks)
    ]
    return results, time.perf_counter() - start


def run_parallel(tasks, work_dir, run_fn, concurrency: int) -> tuple[list[TaskResult], float]:
    jobs = [
        (task, str(Path(work_dir) / f"task_parallel_{index:02d}.py"))
        for index, task in enumerate(tasks)
    ]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda job: run_fn(*job), jobs))
    return results, time.perf_counter() - start


def validate_outputs(results: list[TaskResult]) -> dict:
    report = {"total": len(results), "success": 0, "missing": [], "empty": [], "failed": []}
    for result in results:
        path = Path(result.output_path)
        if result.exit_code != 0:
            report["failed"].append(result.label)
        elif not path.exists():
            report["missing"].append(result.label)
        elif path.stat().st_size == 0:
            report["empty"].append(result.label)
        else:
            report["success"] += 1
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX parallelism benchmark")
    parser.add_argument("--mock", action="store_true", help="Simulate tasks without model calls")
    parser.add_argument("--tasks", type=int, default=4, help="Number of tasks (1-8; default: 4)")
    parser.add_argument("--mock-latency", type=float, default=2.5, help="Base simulated latency")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum concurrent subprocesses")
    args = parser.parse_args()

    task_count = max(1, min(args.tasks, len(TASKS)))
    concurrency = max(1, min(args.concurrency, task_count))
    tasks = TASKS[:task_count]

    if not args.mock:
        provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        if provider not in {"gemini", "ollama"}:
            print(f"ERROR: unsupported LLM_PROVIDER: {provider}", file=sys.stderr)
            raise SystemExit(1)
        if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            print("ERROR: GEMINI_API_KEY not set. Use --mock for simulation.", file=sys.stderr)
            raise SystemExit(1)

    run_fn = (
        (lambda task, path: run_mock_task(task, path, args.mock_latency))
        if args.mock
        else run_real_task
    )
    print(
        f"[benchmark] mode={'mock' if args.mock else 'real'} tasks={task_count} concurrency={concurrency}",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory(prefix="apex-parallel-") as work_dir:
        sequential, sequential_wall = run_sequential(tasks, work_dir, run_fn)
        parallel, parallel_wall = run_parallel(tasks, work_dir, run_fn, concurrency)
        sequential_validation = validate_outputs(sequential)
        parallel_validation = validate_outputs(parallel)

    speedup = sequential_wall / parallel_wall if parallel_wall > 0 else 0.0
    ideal_parallelism = min(task_count, concurrency)
    efficiency = speedup / ideal_parallelism if ideal_parallelism else 0.0
    output = {
        "benchmark": "apex_parallel_codegen",
        "mode": "mock" if args.mock else "real",
        "task_count": task_count,
        "concurrency": concurrency,
        "sequential": {
            "wall_seconds": round(sequential_wall, 3),
            "per_task_seconds": [round(result.duration_seconds, 3) for result in sequential],
            "validation": sequential_validation,
        },
        "parallel": {
            "wall_seconds": round(parallel_wall, 3),
            "per_task_seconds": [round(result.duration_seconds, 3) for result in parallel],
            "validation": parallel_validation,
        },
        "speedup_factor": round(speedup, 2),
        "parallel_efficiency": round(efficiency, 2),
    }
    print(json.dumps(output, indent=2))
    print(
        f"\n[result] sequential={sequential_wall:.2f}s parallel={parallel_wall:.2f}s speedup={speedup:.2f}x",
        file=sys.stderr,
    )
    valid = (
        sequential_validation["success"] == task_count
        and parallel_validation["success"] == task_count
    )
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
