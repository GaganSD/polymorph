"""Adapter: security_synth vendor CSVs → eight staged log-line corpora."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    row_field,
    stream_csv_to_txt,
)

SOURCE_DIR = "data/raw/security_synth"


def _render_okta(row: dict[str, str | None]) -> str | None:
    parts = [
        row_field(row, "published"),
        row_field(row, "severity"),
        "okta",
        f"event={row_field(row, 'eventType')}",
        f'msg="{row_field(row, "displayMessage")}"',
        f"actor={row_field(row, 'actor.displayName')}",
        f"actor_id={row_field(row, 'actor.alternateId')}",
        f"result={row_field(row, 'outcome.result')}",
        f'reason="{row_field(row, "outcome.reason")}"',
        f"ip={row_field(row, 'client.ipAddress')}",
        f"city={row_field(row, 'client.geographicalContext.city')}",
        f'ua="{row_field(row, "client.userAgent.rawUserAgent")}"',
    ]
    return collapse_whitespace(" ".join(parts))


def _render_cisco_dns(row: dict[str, str | None]) -> str | None:
    parts = [
        row_field(row, "Timestamp"),
        "cisco_umbrella",
        f"identity={row_field(row, 'Identities')}",
        f"int_ip={row_field(row, 'InternalIp')}",
        f"ext_ip={row_field(row, 'ExternalIp')}",
        f"action={row_field(row, 'Action')}",
        f"qtype={row_field(row, 'QueryType')}",
        f"rcode={row_field(row, 'ResponseCode')}",
        f"domain={row_field(row, 'Domain')}",
        f"categories={row_field(row, 'Categories')}",
        f"verdict={row_field(row, 'Verdict')}",
        f"url={row_field(row, 'URL')}",
    ]
    return collapse_whitespace(" ".join(parts))


def _render_crowdstrike_network(row: dict[str, str | None]) -> str | None:
    cmd = row_field(row, "CommandLine")
    parts = [
        row_field(row, "timestamp"),
        "crowdstrike",
        row_field(row, "EventType"),
        f"dir={row_field(row, 'ConnectionDirection')}",
        f"local={row_field(row, 'LocalAddressIP4')}:{row_field(row, 'LocalPort')}",
        f"remote={row_field(row, 'RemoteAddressIP4')}:{row_field(row, 'RemotePort')}",
        f"proto={row_field(row, 'Protocol')}",
        f"status={row_field(row, 'Status')}",
        f"image={row_field(row, 'ImageFileName')}",
        f'cmd="{cmd}"',
        f"user={row_field(row, 'UserName')}",
        f"host={row_field(row, 'ComputerName')}",
        f"technique={row_field(row, 'Technique')}",
    ]
    return collapse_whitespace(" ".join(parts))


def _render_crowdstrike_process(row: dict[str, str | None]) -> str | None:
    cmd = row_field(row, "CommandLine")
    parts = [
        row_field(row, "ProcessStartTime"),
        "crowdstrike",
        row_field(row, "EventType"),
        f"image={row_field(row, 'ImageFileName')}",
        f'cmd="{cmd}"',
        f"parent={row_field(row, 'ParentBaseFileName')}",
        f"user={row_field(row, 'UserName')}",
        f"host={row_field(row, 'ComputerName')}",
        f"technique={row_field(row, 'Technique')}",
    ]
    return collapse_whitespace(" ".join(parts))


def _render_crowdstrike_registry(row: dict[str, str | None]) -> str | None:
    reg_str = row_field(row, "RegStringValue")
    parts = [
        row_field(row, "timestamp"),
        "crowdstrike",
        row_field(row, "EventType"),
        f"reg_obj={row_field(row, 'RegObjectName')}",
        f"reg_val={row_field(row, 'RegValueName')}",
        f'reg_str="{reg_str}"',
        f"op={row_field(row, 'RegOperationType')}",
        f"image={row_field(row, 'ImageFileName')}",
        f"user={row_field(row, 'UserName')}",
        f"host={row_field(row, 'ComputerName')}",
        f"technique={row_field(row, 'Technique')}",
    ]
    return collapse_whitespace(" ".join(parts))


def _render_proofpoint_email(row: dict[str, str | None]) -> str | None:
    subject = row_field(row, "subject")
    parts = [
        row_field(row, "time"),
        "proofpoint",
        f"sender={row_field(row, 'sender')}",
        f"recipient={row_field(row, 'recipient')}",
        f'subject="{subject}"',
        f"from={row_field(row, 'headerFrom')}",
        f"spam={row_field(row, 'spamScore')}",
        f"phish={row_field(row, 'phishScore')}",
        f"mlx_label={row_field(row, 'mlxLabel')}",
        f"action={row_field(row, 'action')}",
        f"malware={row_field(row, 'malwareName')}",
        f"sending_ip={row_field(row, 'sendingIp')}",
    ]
    return collapse_whitespace(" ".join(parts))


def _render_zscaler_proxy(row: dict[str, str | None]) -> str | None:
    ua = row_field(row, "useragent")
    parts = [
        row_field(row, "timestamp"),
        "zscaler",
        f"user={row_field(row, 'user')}",
        f"dept={row_field(row, 'department')}",
        f"loc={row_field(row, 'location')}",
        f"client_ip={row_field(row, 'clientip')}",
        f"server_ip={row_field(row, 'serverip')}",
        f"host={row_field(row, 'hostname')}",
        f"url={row_field(row, 'url')}",
        f"cat={row_field(row, 'urlcategory')}",
        f"action={row_field(row, 'action')}",
        f"method={row_field(row, 'requestmethod')}",
        f"status={row_field(row, 'responsecode')}",
        f"ctype={row_field(row, 'contenttype')}",
        f'ua="{ua}"',
        f"threat={row_field(row, 'threatname')}",
        f"rule={row_field(row, 'rulelabel')}",
        f"app={row_field(row, 'appname')}",
    ]
    return collapse_whitespace(" ".join(parts))


def _render_cloudtrail_synth(row: dict[str, str | None]) -> str | None:
    agent = row_field(row, "userAgent")
    parts = [
        row_field(row, "eventTime"),
        row_field(row, "eventName"),
        row_field(row, "eventSource"),
        f"region={row_field(row, 'awsRegion')}",
        f"src_ip={row_field(row, 'sourceIPAddress')}",
        f"identity_type={row_field(row, 'userIdentity.type')}",
        f"principal={row_field(row, 'userIdentity.principalId')}",
        f"arn={row_field(row, 'userIdentity.arn')}",
        f"user={row_field(row, 'userIdentity.userName')}",
        f'agent="{agent}"',
        f"err={row_field(row, 'errorCode')}",
        f"bucket={row_field(row, 'requestParameters.bucketName')}",
    ]
    return collapse_whitespace(" ".join(parts))


_VENDOR_SPECS: list[tuple[str, str, tuple[str, ...], Callable[[dict[str, str | None]], str | None]]] = [
    ("okta_auth", "okta_system_log.csv", ("published", "severity", "eventType"), _render_okta),
    (
        "cisco_dns",
        "cisco_umbrella_dns.csv",
        ("Timestamp", "Identities", "InternalIp"),
        _render_cisco_dns,
    ),
    (
        "crowdstrike_network",
        "crowdstrike_network.csv",
        ("timestamp", "EventType", "ConnectionDirection"),
        _render_crowdstrike_network,
    ),
    (
        "crowdstrike_process",
        "crowdstrike_process.csv",
        ("ProcessStartTime", "EventType", "ImageFileName"),
        _render_crowdstrike_process,
    ),
    (
        "crowdstrike_registry",
        "crowdstrike_registry.csv",
        ("timestamp", "EventType", "RegObjectName"),
        _render_crowdstrike_registry,
    ),
    (
        "proofpoint_email",
        "proofpoint_email.csv",
        ("time", "sender", "recipient"),
        _render_proofpoint_email,
    ),
    (
        "zscaler_proxy",
        "zscaler_proxy.csv",
        ("timestamp", "user", "department"),
        _render_zscaler_proxy,
    ),
    (
        "cloudtrail_synth",
        "aws_cloudtrail.csv",
        ("eventTime", "eventName", "eventSource"),
        _render_cloudtrail_synth,
    ),
]


def stage_corpora(repo_root: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for name, csv_name, required, render in _VENDOR_SPECS:
        source = f"{SOURCE_DIR}/{csv_name}"
        staged_path = f"data/staged/{name}.txt"
        written, skipped = stream_csv_to_txt(
            repo_root / source,
            repo_root / staged_path,
            render_row=render,
            required_columns=required,
        )
        results.append(
            {
                "name": name,
                "source": source,
                "staged_path": staged_path,
                "written_rows": written,
                "skipped_rows": skipped,
            }
        )
    return results
