"""Adapter: api_failures CSV → uniform log lines."""

from __future__ import annotations

from pathlib import Path

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    sanitize_field,
    stream_csv_to_txt,
)

_REQUIRED = (
    "timestamp",
    "api_name",
    "service_owner",
    "environment",
    "http_method",
    "endpoint",
    "status_code",
    "error_type",
    "root_cause",
    "latency_ms",
    "request_size_bytes",
    "response_size_bytes",
    "retry_count",
    "is_retry_successful",
    "client_ip",
    "region",
    "container_id",
    "host_id",
    "thread_id",
    "log_level",
    "error_message",
    "resolution_action",
)

SOURCE_CSV = "data/raw/api_failures/api_error_logs_with_root_causes_220k_rows.csv"
STAGED_TXT = "data/staged/api_failures.txt"


def render_row(row: dict[str, str | None]) -> str | None:
    msg = sanitize_field(row["error_message"])
    root = sanitize_field(row["root_cause"])
    resolution = sanitize_field(row["resolution_action"])
    parts = [
        sanitize_field(row["timestamp"]),
        sanitize_field(row["log_level"]),
        sanitize_field(row["api_name"]),
        sanitize_field(row["http_method"]),
        sanitize_field(row["endpoint"]),
        f"status={sanitize_field(row['status_code'])}",
        f"error_type={sanitize_field(row['error_type'])}",
        f"latency_ms={sanitize_field(row['latency_ms'])}",
        f"retry={sanitize_field(row['retry_count'])}",
        f"retry_ok={sanitize_field(row['is_retry_successful'])}",
        f"env={sanitize_field(row['environment'])}",
        f"region={sanitize_field(row['region'])}",
        f"container={sanitize_field(row['container_id'])}",
        f"host={sanitize_field(row['host_id'])}",
        f"thread={sanitize_field(row['thread_id'])}",
        f"client_ip={sanitize_field(row['client_ip'])}",
        f"owner={sanitize_field(row['service_owner'])}",
        f"req_bytes={sanitize_field(row['request_size_bytes'])}",
        f"resp_bytes={sanitize_field(row['response_size_bytes'])}",
        f'msg="{msg}"',
        f'root_cause="{root}"',
        f'resolution="{resolution}"',
    ]
    return collapse_whitespace(" ".join(parts))


def stage(repo_root: Path) -> tuple[int, int]:
    csv_path = repo_root / SOURCE_CSV
    out_path = repo_root / STAGED_TXT
    return stream_csv_to_txt(
        csv_path, out_path, render_row=render_row, required_columns=_REQUIRED
    )
