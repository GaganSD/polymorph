"""Adapter: cicd_failures CSV → uniform log lines."""

from __future__ import annotations

from pathlib import Path

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    sanitize_field,
    stream_csv_to_txt,
)

_REQUIRED = (
    "pipeline_id",
    "run_id",
    "timestamp",
    "ci_tool",
    "repository",
    "branch",
    "commit_hash",
    "author",
    "language",
    "os",
    "cloud_provider",
    "build_duration_sec",
    "test_duration_sec",
    "deploy_duration_sec",
    "failure_stage",
    "failure_type",
    "error_code",
    "error_message",
    "severity",
    "cpu_usage_pct",
    "memory_usage_mb",
    "retry_count",
    "is_flaky_test",
    "rollback_triggered",
    "incident_created",
)

SOURCE_CSV = "data/raw/cicd_failures/ci_cd_pipeline_failure_logs_dataset.csv"
STAGED_TXT = "data/staged/cicd_failures.txt"


def render_row(row: dict[str, str | None]) -> str | None:
    msg = sanitize_field(row["error_message"])
    parts = [
        sanitize_field(row["timestamp"]),
        sanitize_field(row["severity"]),
        sanitize_field(row["ci_tool"]),
        f"pipeline={sanitize_field(row['pipeline_id'])}",
        f"run={sanitize_field(row['run_id'])}",
        f"repo={sanitize_field(row['repository'])}",
        f"branch={sanitize_field(row['branch'])}",
        f"commit={sanitize_field(row['commit_hash'])}",
        f"lang={sanitize_field(row['language'])}",
        f"os={sanitize_field(row['os'])}",
        f"cloud={sanitize_field(row['cloud_provider'])}",
        f"stage={sanitize_field(row['failure_stage'])}",
        f"failure_type={sanitize_field(row['failure_type'])}",
        f"error_code={sanitize_field(row['error_code'])}",
        f"build_s={sanitize_field(row['build_duration_sec'])}",
        f"test_s={sanitize_field(row['test_duration_sec'])}",
        f"deploy_s={sanitize_field(row['deploy_duration_sec'])}",
        f"cpu_pct={sanitize_field(row['cpu_usage_pct'])}",
        f"mem_mb={sanitize_field(row['memory_usage_mb'])}",
        f"retry={sanitize_field(row['retry_count'])}",
        f"flaky={sanitize_field(row['is_flaky_test'])}",
        f"rollback={sanitize_field(row['rollback_triggered'])}",
        f"incident={sanitize_field(row['incident_created'])}",
        f"author={sanitize_field(row['author'])}",
        f'msg="{msg}"',
    ]
    return collapse_whitespace(" ".join(parts))


def stage(repo_root: Path) -> tuple[int, int]:
    csv_path = repo_root / SOURCE_CSV
    out_path = repo_root / STAGED_TXT
    return stream_csv_to_txt(
        csv_path, out_path, render_row=render_row, required_columns=_REQUIRED
    )
