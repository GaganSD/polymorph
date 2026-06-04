"""Shared helpers for CSV→log-line corpus adapters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable


def sanitize_field(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\n", " ").replace("\r", " ")


def collapse_whitespace(line: str) -> str:
    return " ".join(line.split())


def has_required_columns(row: dict[str, str | None], columns: Iterable[str]) -> bool:
    return all(col in row for col in columns)


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
