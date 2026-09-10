"""JSONL structured trace writer."""
import json
import sys
from pathlib import Path
from time import time
from typing import Any


def write_event(event: dict[str, Any], *, dest: Path | None = None) -> None:
    """Write a single JSONL trace event to file or stderr."""
    record = {"ts": time(), **event}
    line = json.dumps(record, default=str)
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    else:
        print(line, file=sys.stderr)
