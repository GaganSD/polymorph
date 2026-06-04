"""Shared helpers for CSV→log-line corpus adapters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TextIO


def sanitize_field(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\n", " ").replace("\r", " ")


def collapse_whitespace(line: str) -> str:
    return " ".join(line.split())


def has_required_columns(row: dict[str, str | None], columns: Iterable[str]) -> bool:
    return all(col in row for col in columns)


def row_field(row: dict[str, str | None], column: str) -> str:
    """Return a sanitized field value, or empty string if the column is absent."""
    if column not in row:
        return ""
    return sanitize_field(row.get(column))


def stream_csv_to_txt(
    csv_path: Path,
    out_path: Path,
    *,
    render_row: Callable[[dict[str, str | None]], str | None],
    required_columns: Iterable[str],
) -> tuple[int, int]:
    """Stream *csv_path* to *out_path*, one rendered line per row.

    Returns ``(written_count, skipped_count)``.
    """
    written = 0
    skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        reader = csv.DictReader(src)
        for row in reader:
            if not has_required_columns(row, required_columns):
                skipped += 1
                continue
            line = render_row(row)
            if line is None:
                skipped += 1
                continue
            dst.write(line + "\n")
            written += 1

    return written, skipped


def _find_array_start(reader: TextIO, array_key: str) -> bool:
    """Advance *reader* to the opening ``[`` of ``"{array_key}": [``."""
    needle = f'"{array_key}"'
    tail = ""
    chunk_size = 65536
    while True:
        chunk = reader.read(chunk_size)
        if not chunk:
            return False
        search = tail + chunk
        idx = search.find(needle)
        if idx == -1:
            tail = search[-len(needle) :]
            continue
        rest = search[idx + len(needle) :]
        bracket = rest.find("[")
        if bracket == -1:
            tail = rest
            continue
        # Preserve any bytes after '[' for the element iterator.
        overflow = rest[bracket + 1 :]
        reader._overflow = overflow  # type: ignore[attr-defined]
        return True


def _iter_json_array_elements(reader: TextIO, array_key: str) -> Iterator[Any]:
    """Yield each top-level element of a JSON array keyed by *array_key*."""
    if not _find_array_start(reader, array_key):
        return

    overflow: str = getattr(reader, "_overflow", "")
    del reader._overflow  # type: ignore[attr-defined]

    in_string = False
    escape = False
    collecting = False
    element_depth = 0
    element_parts: list[str] = []
    pending = overflow

    while True:
        chunk = pending if pending else reader.read(65536)
        pending = ""
        if not chunk:
            return

        i = 0
        while i < len(chunk):
            ch = chunk[i]
            if not collecting:
                if ch in " \t\r\n,":
                    i += 1
                    continue
                if ch == "]":
                    return
                collecting = True
                element_parts = [ch]
                if ch == '"':
                    in_string = True
                    element_depth = 0
                elif ch in "[{":
                    element_depth = 1
                else:
                    element_depth = 0
                i += 1
                if element_depth == 0 and ch not in '"[{':
                    yield json.loads("".join(element_parts))
                    collecting = False
                    element_parts = []
                continue

            element_parts.append(ch)

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                    if element_depth == 0:
                        yield json.loads("".join(element_parts))
                        collecting = False
                        element_parts = []
                i += 1
                continue

            if ch == '"':
                in_string = True
                i += 1
                continue
            if ch in "[{":
                element_depth += 1
                i += 1
                continue
            if ch in "]}":
                element_depth -= 1
                if element_depth == 0:
                    yield json.loads("".join(element_parts))
                    collecting = False
                    element_parts = []
                i += 1
                continue
            i += 1


def stream_json_array_to_txt(
    json_path: Path,
    out_path: Path,
    *,
    array_key: str,
    render_item: Callable[[Any], str | None],
) -> tuple[int, int]:
    """Stream elements from a large JSON object's array field to log lines."""
    written = 0
    skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open(encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for element in _iter_json_array_elements(src, array_key):
            line = render_item(element)
            if line is None:
                skipped += 1
                continue
            dst.write(line + "\n")
            written += 1

    return written, skipped
