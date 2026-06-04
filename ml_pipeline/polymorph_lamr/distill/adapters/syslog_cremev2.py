"""Adapter: CREMEv2 original_label_syslog CSV → uniform log lines."""

from __future__ import annotations

from pathlib import Path

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    row_field,
    stream_csv_to_txt,
)

_REQUIRED = ("Time", "HostName", "Component", "Content")

SOURCE_CSV = "data/raw/cremev2/original_label_syslog.csv"
STAGED_TXT = "data/staged/syslog_cremev2.txt"


def render_row(row: dict[str, str | None]) -> str | None:
    parts = [
        row_field(row, "Time"),
        row_field(row, "HostName"),
        f"{row_field(row, 'Component')}[{row_field(row, 'PID_or_IP')}]:",
        row_field(row, "Content"),
        f"tactic={row_field(row, 'Tactic')}",
        f"technique={row_field(row, 'Technique')}",
    ]
    return collapse_whitespace(" ".join(parts))


def stage(repo_root: Path) -> tuple[int, int]:
    return stream_csv_to_txt(
        repo_root / SOURCE_CSV,
        repo_root / STAGED_TXT,
        render_row=render_row,
        required_columns=_REQUIRED,
    )
