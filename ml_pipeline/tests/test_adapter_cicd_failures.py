"""Tests for cicd_failures CSV adapter."""

from pathlib import Path

from polymorph_lamr.distill.adapters import cicd_failures
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt

FIXTURE_CSV = """\
pipeline_id,run_id,timestamp,ci_tool,repository,branch,commit_hash,author,language,os,cloud_provider,build_duration_sec,test_duration_sec,deploy_duration_sec,failure_stage,failure_type,error_code,error_message,severity,cpu_usage_pct,memory_usage_mb,retry_count,is_flaky_test,rollback_triggered,incident_created
pipe_2032,run_0,2025-12-29T07:58:16.927259,Jenkins,repo_469,release,53820d0dddb2d97f40cbf0e1b4566169f480b86e,user_689,Python,windows-latest,On-Prem,1996,805,532,deploy,Security Scan Failure,ERR_621,ERROR: test failure,MEDIUM,41.15,3132,3,True,True,True
"""

EXPECTED = (
    "2025-12-29T07:58:16.927259 MEDIUM Jenkins pipeline=pipe_2032 run=run_0 "
    "repo=repo_469 branch=release commit=53820d0dddb2d97f40cbf0e1b4566169f480b86e "
    "lang=Python os=windows-latest cloud=On-Prem stage=deploy "
    "failure_type=Security Scan Failure error_code=ERR_621 build_s=1996 test_s=805 "
    "deploy_s=532 cpu_pct=41.15 mem_mb=3132 retry=3 flaky=True rollback=True "
    'incident=True author=user_689 msg="ERROR: test failure"'
)


def test_render_row_exact_line():
    row = {
        "pipeline_id": "pipe_2032",
        "run_id": "run_0",
        "timestamp": "2025-12-29T07:58:16.927259",
        "ci_tool": "Jenkins",
        "repository": "repo_469",
        "branch": "release",
        "commit_hash": "53820d0dddb2d97f40cbf0e1b4566169f480b86e",
        "author": "user_689",
        "language": "Python",
        "os": "windows-latest",
        "cloud_provider": "On-Prem",
        "build_duration_sec": "1996",
        "test_duration_sec": "805",
        "deploy_duration_sec": "532",
        "failure_stage": "deploy",
        "failure_type": "Security Scan Failure",
        "error_code": "ERR_621",
        "error_message": "ERROR: test failure",
        "severity": "MEDIUM",
        "cpu_usage_pct": "41.15",
        "memory_usage_mb": "3132",
        "retry_count": "3",
        "is_flaky_test": "True",
        "rollback_triggered": "True",
        "incident_created": "True",
    }
    assert cicd_failures.render_row(row) == EXPECTED


def test_stream_csv_to_txt(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(FIXTURE_CSV, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=cicd_failures.render_row,
        required_columns=cicd_failures._REQUIRED,
    )
    assert written == 1
    assert skipped == 0
    assert out_path.read_text(encoding="utf-8").strip() == EXPECTED


def test_stage_integration(tmp_path: Path):
    raw = tmp_path / "data/raw/cicd_failures"
    raw.mkdir(parents=True)
    (raw / "ci_cd_pipeline_failure_logs_dataset.csv").write_text(
        FIXTURE_CSV, encoding="utf-8"
    )
    written, skipped = cicd_failures.stage(tmp_path)
    assert written == 1
    assert skipped == 0
