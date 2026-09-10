"""Static and LLM-backed pre-execution plan auditing."""
import json
import re
from pathlib import Path

from apex.core.types import Halt, Plan, ToolCall

_BLOCK_PATTERNS = (
    re.compile(r"rm\s+-[rf]+\s+/"),
    re.compile(r"chmod\s+777\s+/"),
    re.compile(r"curl\s+.*\|\s*sh"),
    re.compile(r"wget\s+.*\|\s*sh"),
    re.compile(r"dd\s+if=/dev/zero"),
    re.compile(r"mkfs\."),
    re.compile(r">\s*/dev/sd"),
    re.compile(r":\(\)\{:\|:&\};"),
)
_ALLOWED_WRITE_ROOTS = (Path.home(), Path("/tmp"))


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _static_check(plan: Plan) -> list[dict]:
    violations: list[dict] = []
    for index, step in enumerate(plan.steps):
        if not isinstance(step, ToolCall):
            continue
        args_string = json.dumps(step.args, ensure_ascii=False)
        for pattern in _BLOCK_PATTERNS:
            if pattern.search(args_string):
                violations.append(
                    {
                        "step": index,
                        "tool": step.name,
                        "reason": f"blocked pattern: {pattern.pattern}",
                    }
                )
        if step.name == "write_file":
            raw_path = step.args.get("path")
            if raw_path:
                path = Path(str(raw_path)).expanduser()
                if not any(_under(path, root) for root in _ALLOWED_WRITE_ROOTS):
                    violations.append(
                        {
                            "step": index,
                            "tool": step.name,
                            "reason": f"write outside HOME or /tmp: {raw_path}",
                        }
                    )
    return violations


def static_audit(plan: Plan) -> dict:
    violations = _static_check(plan)
    if violations:
        return {
            "safe": False,
            "risk_level": "critical",
            "findings": violations,
            "summary": f"Static prefilter blocked {len(violations)} pattern(s)",
        }
    return {
        "safe": True,
        "risk_level": "none",
        "findings": [],
        "summary": "Static check passed",
    }


_SAFETY_PROMPT = """You are a security auditor for an AI agent execution system.

Audit every tool call for destructive filesystem operations, exfiltration, privilege escalation,
self-modification outside an explicit RSI context, irreversible operations outside /tmp or HOME,
and resource-exhaustion patterns.

Respond ONLY with a JSON object:
{
  "safe": true,
  "risk_level": "none|low|medium|high|critical",
  "findings": [{"step": 0, "tool": "name", "reason": "reason"}],
  "summary": "one sentence"
}

If safe is true, return an empty findings array. Do not output text outside the JSON object.

Plan to audit:
"""


def _validate_audit(value: object) -> dict:
    if not isinstance(value, dict) or type(value.get("safe")) is not bool:
        raise ValueError("Safety auditor returned an invalid audit object")
    if value.get("risk_level") not in {"none", "low", "medium", "high", "critical"}:
        raise ValueError("Safety auditor returned an invalid risk_level")
    if not isinstance(value.get("findings"), list) or not isinstance(value.get("summary"), str):
        raise ValueError("Safety auditor returned invalid findings or summary")
    return value


def audit_plan(plan: Plan, *, api_key: str, provider: str = "gemini") -> dict:
    """Run deterministic checks, then request an LLM audit for plans that pass them."""
    del provider  # Preserve the established signature; LLM_PROVIDER selects the active provider.

    static = static_audit(plan)
    if not static["safe"]:
        return static

    steps = []
    for index, step in enumerate(plan.steps):
        if isinstance(step, ToolCall):
            steps.append({"index": index, "type": "tool", "name": step.name, "args": step.args})
        elif isinstance(step, Halt):
            steps.append({"index": index, "type": "halt", "reason": step.reason})

    from apex.llm import gemini_complete

    response = gemini_complete(
        _SAFETY_PROMPT + json.dumps({"goal": plan.goal, "steps": steps}, indent=2),
        api_key=api_key,
    )
    if response.get("error"):
        raise ValueError(f"Safety auditor provider failed: {response['error']}")
    text = (response.get("text") or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    try:
        return _validate_audit(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Safety auditor returned unparseable response: {exc}") from exc


def format_audit_report(audit: dict) -> str:
    lines = [f"[paranoid] risk={audit.get('risk_level', '?')} safe={audit.get('safe')}"]
    lines.append(f"[paranoid] {audit.get('summary', '')}")
    for finding in audit.get("findings", []):
        lines.append(
            f"[paranoid] step {finding.get('step', '?')} "
            f"({finding.get('tool', '?')}): {finding.get('reason', '')}"
        )
    return "\n".join(lines)
