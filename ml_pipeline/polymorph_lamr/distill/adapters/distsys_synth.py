"""Adapter: distsys_synth CSV → uniform log lines."""

from __future__ import annotations

from pathlib import Path

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    sanitize_field,
    stream_csv_to_txt,
)

_REQUIRED = (
    "Timestamp",
    "LogLevel",
    "Service",
    "Message",
    "RequestID",
    "User",
    "ClientIP",
    "TimeTaken",
)

SOURCE_CSV = "data/raw/distsys_synth/logdata.csv"
STAGED_TXT = "data/staged/distsys_synth.txt"


def render_row(row: dict[str, str | None]) -> str | None:
    parts = [
        sanitize_field(row["Timestamp"]),
        sanitize_field(row["LogLevel"]),
        f"[{sanitize_field(row['Service'])}]",
        sanitize_field(row["Message"]),
        f"request_id={sanitize_field(row['RequestID'])}",
        f"user={sanitize_field(row['User'])}",
        f"client_ip={sanitize_field(row['ClientIP'])}",
        f"time_taken={sanitize_field(row['TimeTaken'])}",
    ]
    return collapse_whitespace(" ".join(parts))


def stage(repo_root: Path) -> tuple[int, int]:
    csv_path = repo_root / SOURCE_CSV
    out_path = repo_root / STAGED_TXT
    return stream_csv_to_txt(
        csv_path, out_path, render_row=render_row, required_columns=_REQUIRED
    )
