"""Adapter: stacktraces.json → one log line per Python traceback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    sanitize_field,
    stream_json_array_to_txt,
)

SOURCE_JSON = "data/raw/python_tracebacks/stacktraces.json"
STAGED_TXT = "data/staged/python_tracebacks.txt"

_TRACE_MARKERS = ("Traceback", 'File "', "Error")


def render_item(item: Any) -> str | None:
    if not isinstance(item, list) or len(item) < 4:
        return None
    text = item[3]
    if not text or not isinstance(text, str):
        return None
    if not any(marker in text for marker in _TRACE_MARKERS):
        return None
    line = sanitize_field(text)
    return collapse_whitespace(line)


def stage(repo_root: Path) -> tuple[int, int]:
    return stream_json_array_to_txt(
        repo_root / SOURCE_JSON,
        repo_root / STAGED_TXT,
        array_key="data",
        render_item=render_item,
    )
