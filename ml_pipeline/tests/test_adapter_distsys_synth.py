"""Tests for distsys_synth CSV adapter."""

from pathlib import Path

from polymorph_lamr.distill.adapters import distsys_synth
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt

FIXTURE_CSV = """\
,Timestamp,LogLevel,Service,Message,RequestID,User,ClientIP,TimeTaken
0,2023-11-20T08:40:50.664842,WARNING,ServiceA,Performance Warnings,6743,User96,192.168.1.102,28ms
"""

EXPECTED = (
    "2023-11-20T08:40:50.664842 WARNING [ServiceA] Performance Warnings "
    "request_id=6743 user=User96 client_ip=192.168.1.102 time_taken=28ms"
)


def test_render_row_exact_line():
    row = {
        "Timestamp": "2023-11-20T08:40:50.664842",
        "LogLevel": "WARNING",
        "Service": "ServiceA",
        "Message": "Performance Warnings",
        "RequestID": "6743",
        "User": "User96",
        "ClientIP": "192.168.1.102",
        "TimeTaken": "28ms",
    }
    assert distsys_synth.render_row(row) == EXPECTED


def test_render_row_collapses_embedded_newlines():
    row = {
        "Timestamp": "2023-11-20T08:40:50.664842",
        "LogLevel": "WARNING",
        "Service": "ServiceA",
        "Message": "line1\nline2",
        "RequestID": "6743",
        "User": "User96",
        "ClientIP": "192.168.1.102",
        "TimeTaken": "28ms",
    }
    line = distsys_synth.render_row(row)
    assert "\n" not in line
    assert "line1 line2" in line


def test_stream_csv_to_txt(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(FIXTURE_CSV, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=distsys_synth.render_row,
        required_columns=distsys_synth._REQUIRED,
    )
    assert written == 1
    assert skipped == 0
    assert out_path.read_text(encoding="utf-8").strip() == EXPECTED


def test_stream_skips_missing_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Timestamp,LogLevel\n2023-11-20,INFO\n", encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=distsys_synth.render_row,
        required_columns=distsys_synth._REQUIRED,
    )
    assert written == 0
    assert skipped == 1


def test_stage_integration(tmp_path: Path):
    raw = tmp_path / "data/raw/distsys_synth"
    raw.mkdir(parents=True)
    (raw / "logdata.csv").write_text(FIXTURE_CSV, encoding="utf-8")
    written, skipped = distsys_synth.stage(tmp_path)
    out = tmp_path / distsys_synth.STAGED_TXT
    assert written == 1
    assert skipped == 0
    assert out.read_text(encoding="utf-8").strip() == EXPECTED
