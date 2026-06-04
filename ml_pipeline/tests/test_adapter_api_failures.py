"""Tests for api_failures CSV adapter."""

from pathlib import Path

from polymorph_lamr.distill.adapters import api_failures
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt

FIXTURE_CSV = """\
timestamp,api_name,service_owner,environment,http_method,endpoint,status_code,error_type,root_cause,latency_ms,request_size_bytes,response_size_bytes,retry_count,is_retry_successful,client_ip,region,container_id,host_id,thread_id,log_level,error_message,resolution_action
2024-01-01 00:00:00,inventory-api,team-beta,dev,DELETE,/v1/lljugd,503,Timeout,High latency in network,9403,19735,56072,1,True,192.168.97.252,ap-south-1,xuqbiprmtjwu,zjcdzcnlzt,1258,WARN,Internal server error,Refresh token
"""

EXPECTED = (
    '2024-01-01 00:00:00 WARN inventory-api DELETE /v1/lljugd status=503 '
    "error_type=Timeout latency_ms=9403 retry=1 retry_ok=True env=dev "
    "region=ap-south-1 container=xuqbiprmtjwu host=zjcdzcnlzt thread=1258 "
    "client_ip=192.168.97.252 owner=team-beta req_bytes=19735 resp_bytes=56072 "
    'msg="Internal server error" root_cause="High latency in network" '
    'resolution="Refresh token"'
)


def test_render_row_exact_line():
    row = {
        "timestamp": "2024-01-01 00:00:00",
        "api_name": "inventory-api",
        "service_owner": "team-beta",
        "environment": "dev",
        "http_method": "DELETE",
        "endpoint": "/v1/lljugd",
        "status_code": "503",
        "error_type": "Timeout",
        "root_cause": "High latency in network",
        "latency_ms": "9403",
        "request_size_bytes": "19735",
        "response_size_bytes": "56072",
        "retry_count": "1",
        "is_retry_successful": "True",
        "client_ip": "192.168.97.252",
        "region": "ap-south-1",
        "container_id": "xuqbiprmtjwu",
        "host_id": "zjcdzcnlzt",
        "thread_id": "1258",
        "log_level": "WARN",
        "error_message": "Internal server error",
        "resolution_action": "Refresh token",
    }
    assert api_failures.render_row(row) == EXPECTED


def test_stream_csv_to_txt(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(FIXTURE_CSV, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=api_failures.render_row,
        required_columns=api_failures._REQUIRED,
    )
    assert written == 1
    assert skipped == 0
    assert out_path.read_text(encoding="utf-8").strip() == EXPECTED


def test_stage_integration(tmp_path: Path):
    raw = tmp_path / "data/raw/api_failures"
    raw.mkdir(parents=True)
    (raw / "api_error_logs_with_root_causes_220k_rows.csv").write_text(
        FIXTURE_CSV, encoding="utf-8"
    )
    written, skipped = api_failures.stage(tmp_path)
    assert written == 1
    assert skipped == 0
