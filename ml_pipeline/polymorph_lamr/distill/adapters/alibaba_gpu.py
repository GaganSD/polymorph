"""Adapter: alibaba_gpu job CSV → uniform log lines."""

from __future__ import annotations

from pathlib import Path

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    sanitize_field,
    stream_csv_to_txt,
)

_REQUIRED = (
    "job_name",
    "organization",
    "gpu_model",
    "cpu_request",
    "gpu_request",
    "worker_num",
    "submit_time",
    "duration",
    "job_type",
)

SOURCE_CSV = "data/raw/alibaba_gpu/job_info_df.csv"
STAGED_TXT = "data/staged/alibaba_gpu.txt"


def render_row(row: dict[str, str | None]) -> str | None:
    parts = [
        f"submit_time={sanitize_field(row['submit_time'])}",
        f"job={sanitize_field(row['job_name'])}",
        f"org={sanitize_field(row['organization'])}",
        f"type={sanitize_field(row['job_type'])}",
        f"gpu_model={sanitize_field(row['gpu_model'])}",
        f"gpu_request={sanitize_field(row['gpu_request'])}",
        f"cpu_request={sanitize_field(row['cpu_request'])}",
        f"workers={sanitize_field(row['worker_num'])}",
        f"duration={sanitize_field(row['duration'])}",
    ]
    return collapse_whitespace(" ".join(parts))


def stage(repo_root: Path) -> tuple[int, int]:
    csv_path = repo_root / SOURCE_CSV
    out_path = repo_root / STAGED_TXT
    return stream_csv_to_txt(
        csv_path, out_path, render_row=render_row, required_columns=_REQUIRED
    )
