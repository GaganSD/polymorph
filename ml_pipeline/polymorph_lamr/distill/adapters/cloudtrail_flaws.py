"""Adapter: AWS CloudTrail nineteenFeaturesDf CSV → uniform log lines."""

from __future__ import annotations

from pathlib import Path

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    row_field,
    stream_csv_to_txt,
)

_REQUIRED = ("eventTime", "eventName", "eventSource")

SOURCE_CSV = "data/raw/aws_cloudtrail/nineteenFeaturesDf.csv"
STAGED_TXT = "data/staged/cloudtrail_flaws.txt"


def render_row(row: dict[str, str | None]) -> str | None:
    agent = row_field(row, "userAgent")
    err_msg = row_field(row, "errorMessage")
    parts = [
        row_field(row, "eventTime"),
        row_field(row, "eventName"),
        row_field(row, "eventSource"),
        f"region={row_field(row, 'awsRegion')}",
        f"identity_type={row_field(row, 'userIdentitytype')}",
        f"event_type={row_field(row, 'eventType')}",
        f"arn={row_field(row, 'userIdentityarn')}",
        f"user={row_field(row, 'userIdentityuserName')}",
        f"src_ip={row_field(row, 'sourceIPAddress')}",
        f'agent="{agent}"',
        f"principal={row_field(row, 'userIdentityprincipalId')}",
        f"err={row_field(row, 'errorCode')}",
        f'err_msg="{err_msg}"',
        f"req_instance_type={row_field(row, 'requestParametersinstanceType')}",
    ]
    return collapse_whitespace(" ".join(parts))


def stage(repo_root: Path) -> tuple[int, int]:
    return stream_csv_to_txt(
        repo_root / SOURCE_CSV,
        repo_root / STAGED_TXT,
        render_row=render_row,
        required_columns=_REQUIRED,
    )
