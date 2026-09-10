"""
apex/rsi.py — Recursive Self-Improvement scaffolding

Each cycle:
  1. Run benchmark → compute apex_score
  2. Read own source files
  3. Generate patch via LLM
  4. Paranoid-validate patch
  5. Apply on git branch rsi/cycle-N
  6. Re-run benchmark → compare score
  7. Report delta; human gate before merge
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

APEX_CMD = [sys.executable, "-m", "apex"]
BENCH_CMD = [sys.executable, "-m", "apex.bench"]
REPO_ROOT = Path(__file__).parent.parent

# Source files eligible for RSI patching
RSI_SOURCE_FILES = [
    "apex/core/loop.py",
    "apex/core/planner.py",
    "apex/llm.py",
]


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def _current_branch() -> str:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def _require_clean_tree() -> None:
    if _git(["status", "--porcelain"]).stdout.strip():
        raise RuntimeError("RSI requires a clean Git working tree")


def _branch_exists(name: str) -> bool:
    return _git(["show-ref", "--verify", "--quiet", f"refs/heads/{name}"], check=False).returncode == 0


def _create_branch(name: str) -> None:
    if _branch_exists(name):
        raise RuntimeError(f"branch already exists: {name}")
    _git(["checkout", "-b", name])


def _checkout(branch: str) -> None:
    _git(["checkout", branch])


def _apply_patch(patch_text: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch_text)
        patch_path = f.name
    try:
        result = _git(["apply", "--whitespace=fix", patch_path], check=False)
        return result.returncode == 0
    finally:
        os.unlink(patch_path)


def _run_candidate(
    candidate_idx: int, patch_text: str, tasks_path: str, mock_bench: bool, k: int = 3
) -> float | None:
    """Require passing regressions and valid benchmark runs in a scratch worktree."""
    scratch = tempfile.mkdtemp(prefix=f"apex-rsi-cand{candidate_idx}-")
    try:
        _git(["worktree", "add", "--detach", scratch, "HEAD"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, dir=scratch) as f:
            f.write(patch_text)
            patch_path = f.name
        applied = subprocess.run(
            ["git", "apply", "--whitespace=fix", patch_path],
            cwd=scratch, capture_output=True, text=True,
        )
        os.unlink(patch_path)
        if applied.returncode != 0:
            return None
        validation = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-m", "not integration", "-q"],
            cwd=scratch, capture_output=True, text=True, timeout=300,
        )
        if validation.returncode != 0:
            print(f"[rsi] candidate {candidate_idx} failed regression tests", file=sys.stderr)
            return None
        scores = []
        for _ in range(k):
            cmd = BENCH_CMD + ["--tasks", tasks_path]
            if mock_bench:
                cmd.append("--mock")
            result = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return None
            try:
                score = json.loads(result.stdout)["apex_score"]
                if type(score) not in (int, float) or not 0 <= score <= 1:
                    return None
                scores.append(score)
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
        return sum(scores) / len(scores) if scores else None
    except subprocess.TimeoutExpired:
        print(f"[rsi] candidate {candidate_idx} validation timed out", file=sys.stderr)
        return None
    finally:
        _git(["worktree", "remove", "--force", scratch], check=False)


def _commit_cycle(cycle: int, score_before: float, score_after: float) -> None:
    _git(["add", "-A"])
    _git(["commit", "-m",
          f"rsi: cycle-{cycle} score {score_before:.6f} -> {score_after:.6f}"])


# ---------------------------------------------------------------------------
# Bench wrapper
# ---------------------------------------------------------------------------

def _run_bench(tasks_path: str, mock: bool = False) -> dict[str, Any]:
    cmd = BENCH_CMD + ["--tasks", tasks_path]
    if mock:
        cmd.append("--mock")
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        # bench writes JSON to stdout
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"apex_score": 0.0, "error": result.stderr[:512]}


# ---------------------------------------------------------------------------
# Patch generation via LLM
# ---------------------------------------------------------------------------

def _read_sources() -> str:
    parts = []
    for rel in RSI_SOURCE_FILES:
        p = REPO_ROOT / rel
        if p.exists():
            parts.append(f"### {rel}\n```python\n{p.read_text()}\n```")
    return "\n\n".join(parts)


def _generate_patch(api_key: str, sources: str, score: float,
                    budget_tokens: int) -> str | None:
    """Call LLM to produce a unified diff improving apex_score."""
    from apex.llm import gemini_complete

    prompt = (
        f"You are an expert Python engineer improving the APEX agentic runtime.\n"
        f"Current apex_score: {score:.6f} (higher is better).\n"
        f"apex_score = pass_rate * speed_factor * token_efficiency\n\n"
        f"Source files:\n{sources}\n\n"
        f"Produce a minimal unified diff (git diff format) that improves apex_score "
        f"by reducing token usage or wall time without breaking correctness.\n"
        f"Output ONLY the raw unified diff, no explanation, no markdown fences."
    )

    try:
        response = gemini_complete(
            prompt=prompt,
            api_key=api_key,
        )
        text = response.get('text', '') if isinstance(response, dict) else response
        if not text:
            return None
        text = text.replace('```diff', '').replace('```', '')
        lines = text.splitlines(keepends=True)
        start = next((i for i, l in enumerate(lines) if l.startswith('--- ')), None)
        if start is None:
            return None
        out = []
        in_file = False
        for l in lines[start:]:
            if l.startswith('--- '):
                if in_file and out and out[-1].strip() != '':
                    out.append('\n')
                in_file = True
            elif l.startswith('@@ ') and not in_file:
                continue
            if in_file:
                out.append(l)
        text = ''.join(out)
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"[rsi] LLM patch generation failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Paranoid validation of patch
# ---------------------------------------------------------------------------

def _validate_patch(patch_text: str) -> bool:
    """Reject unsafe additions and patches outside the explicit RSI allowlist."""
    if any(line == "GIT binary patch" or line.startswith("Binary files ")
           for line in patch_text.splitlines()):
        print("[rsi] binary patches are not permitted", file=sys.stderr)
        return False
    blocked = (
        re.compile(r"rm\s+-[rf]+\s+/"),
        re.compile(r"chmod\s+777\s+/"),
        re.compile(r"curl\s+.*\|\s*sh"),
        re.compile(r"wget\s+.*\|\s*sh"),
        re.compile(r"dd\s+if=/dev/zero"),
        re.compile(r"mkfs\."),
        re.compile(r":\(\)\{:\|:&\};"),
    )
    added = "\n".join(
        line[1:]
        for line in patch_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ).lower()
    for pattern in blocked:
        if pattern.search(added):
            print(f"[rsi] patch contains blocked pattern: {pattern.pattern}", file=sys.stderr)
            return False

    # Ask the same parser used for application: textual header scans miss
    # unprefixed/quoted paths and metadata-only changes in mixed diffs.
    parsed = subprocess.run(
        ["git", "apply", "--numstat", "-z", "-"],
        input=patch_text, cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if parsed.returncode != 0 or not parsed.stdout:
        print("[rsi] patch is malformed or contains no changes", file=sys.stderr)
        return False
    allowed = set(RSI_SOURCE_FILES)
    for record in parsed.stdout.rstrip("\0").split("\0"):
        fields = record.split("\t", 2)
        if len(fields) != 3 or fields[2] not in allowed or "-" in fields[:2]:
            print("[rsi] patch contains a non-RSI target or binary change", file=sys.stderr)
            return False

    # RSI edits existing source text only. Reject creation, deletion, renames,
    # copies and mode changes (including conversion to symbolic links).
    summary = subprocess.run(
        ["git", "apply", "--summary", "-"],
        input=patch_text, cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if summary.returncode != 0 or summary.stdout:
        print("[rsi] patch changes file identity or mode", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Governor
# ---------------------------------------------------------------------------

class CycleGovernor:
    def __init__(self, max_cycles: int, budget_tokens: int, max_wall_seconds: float):
        self.max_cycles = max_cycles
        self.budget_tokens = budget_tokens
        self.max_wall_seconds = max_wall_seconds
        self._tokens_used = 0
        self._wall_start = time.time()

    def tokens_remaining(self) -> int:
        return max(0, self.budget_tokens - self._tokens_used)

    def consume_tokens(self, n: int) -> None:
        self._tokens_used += n

    def wall_elapsed(self) -> float:
        return time.time() - self._wall_start

    def check(self, cycle: int) -> tuple[bool, str]:
        if cycle > self.max_cycles:
            return False, f"max_cycles={self.max_cycles} reached"
        if self.tokens_remaining() == 0:
            return False, f"budget_tokens={self.budget_tokens} exhausted"
        if self.wall_elapsed() > self.max_wall_seconds:
            return False, f"max_wall_seconds={self.max_wall_seconds} exceeded"
        return True, ""


# ---------------------------------------------------------------------------
# Main RSI loop
# ---------------------------------------------------------------------------

def run_rsi(
    cycles: int,
    budget_tokens: int,
    tasks_path: str,
    mock_bench: bool,
    api_key: str,
) -> None:
    _require_clean_tree()
    origin_branch = _current_branch()
    if origin_branch == "HEAD":
        raise RuntimeError("RSI requires a named Git branch")
    governor = CycleGovernor(
        max_cycles=cycles,
        budget_tokens=budget_tokens,
        max_wall_seconds=3600.0,
    )

    print(f"[rsi] starting on branch '{origin_branch}'", flush=True)
    print(f"[rsi] cycles={cycles} budget_tokens={budget_tokens}", flush=True)

    # Baseline score
    baseline = _run_bench(tasks_path, mock=mock_bench)
    score = baseline.get("apex_score", 0.0)
    print(f"[rsi] baseline apex_score={score:.6f}", flush=True)

    results_log: list[dict] = []

    for cycle in range(1, cycles + 1):
        ok, reason = governor.check(cycle)
        if not ok:
            print(f"[rsi] governor halt: {reason}", flush=True)
            break

        branch = f"rsi/cycle-{cycle}"
        print(f"\n[rsi] === cycle {cycle} / {cycles} ===", flush=True)

        # Create branch
        _checkout(origin_branch)
        _create_branch(branch)

        try:
            sources = _read_sources()
            remaining = governor.tokens_remaining()

            # Multi-candidate: generate N=3 patches, score each in isolation
            N_CANDIDATES = 3
            candidates = []
            for ci in range(N_CANDIDATES):
                patch = _generate_patch(api_key, sources, score, remaining)
                if not patch:
                    continue
                if not _validate_patch(patch):
                    print(f"[rsi] candidate {ci} rejected by paranoid", flush=True)
                    continue
                estimated_tokens = len(patch) // 4 + len(sources) // 4
                governor.consume_tokens(estimated_tokens)
                cand_score = _run_candidate(ci, patch, tasks_path, mock_bench, k=3)
                if cand_score is None:
                    continue
                print(f"[rsi] candidate {ci} score={cand_score:.6f}", flush=True)
                candidates.append((cand_score, patch, estimated_tokens))

            if not candidates:
                print(f"[rsi] no valid candidates — skipping cycle", flush=True)
                _checkout(origin_branch)
                _git(["branch", "-D", branch])
                continue

            best_score, best_patch, best_tokens = max(candidates, key=lambda item: item[0])
            print(f"[rsi] best candidate score={best_score:.6f} "
                  f"(from {len(candidates)} candidates)", flush=True)

            # Apply best patch to working tree
            if not _apply_patch(best_patch):
                print(f"[rsi] best patch apply failed — skipping cycle", flush=True)
                _checkout(origin_branch)
                _git(["branch", "-D", branch])
                continue

            new_score = best_score
            delta = new_score - score

            print(f"[rsi] score: {score:.6f} -> {new_score:.6f}  delta={delta:+.6f}",
                  flush=True)

            _commit_cycle(cycle, score, new_score)

            results_log.append({
                "cycle": cycle,
                "branch": branch,
                "score_before": score,
                "score_after": new_score,
                "delta": delta,
                "tokens_used": best_tokens,
            })

            if new_score > score:
                score = new_score
                print(f"[rsi] improvement on branch '{branch}'", flush=True)
                print(f"[rsi] HUMAN REVIEW REQUIRED before merging '{branch}' -> '{origin_branch}'",
                      flush=True)
            else:
                print(f"[rsi] no improvement — branch '{branch}' retained for inspection",
                      flush=True)

        except Exception as e:
            print(f"[rsi] cycle {cycle} error: {e}", file=sys.stderr)
            _checkout(origin_branch)
            _git(["branch", "-D", branch], check=False)

        finally:
            _checkout(origin_branch)

    print(f"\n[rsi] complete. wall={governor.wall_elapsed():.1f}s "
          f"tokens_used={governor.budget_tokens - governor.tokens_remaining()}",
          flush=True)
    print(json.dumps({"rsi_results": results_log}, indent=2))


# ---------------------------------------------------------------------------
# CLI entry (called from __main__.py dispatch)
# ---------------------------------------------------------------------------

def rsi_main(argv: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="apex rsi")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of RSI cycles (default: 3)")
    parser.add_argument("--budget-tokens", type=int, default=50000,
                        help="Max total tokens across all cycles (default: 50000)")
    parser.add_argument("--tasks", default="benchmarks/tasks.json",
                        help="Benchmark tasks.json path")
    parser.add_argument("--mock-bench", action="store_true",
                        help="Use mock benchmark (no real apex calls)")
    args = parser.parse_args(argv)
    if args.cycles <= 0:
        parser.error("--cycles must be greater than 0")
    if args.budget_tokens <= 0:
        parser.error("--budget-tokens must be greater than 0")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if (
        not api_key
        and os.environ.get("LLM_PROVIDER", "gemini").lower() == "gemini"
    ):
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    run_rsi(
        cycles=args.cycles,
        budget_tokens=args.budget_tokens,
        tasks_path=args.tasks,
        mock_bench=args.mock_bench,
        api_key=api_key,
    )
