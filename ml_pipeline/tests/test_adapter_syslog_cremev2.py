"""Tests for syslog_cremev2 CSV adapter."""

from pathlib import Path

from polymorph_lamr.distill.adapters import syslog_cremev2
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt

FIXTURE_CSV = """\
Time,HostName,Component,PID_or_IP,Content,EventId,EventTemplate,ParameterList,Timestamp,Label,Tactic,Technique,SubTechnique,Label_lifecycle
2023-04-20T13:26:15+08:00,target-server,proftpd,4311,localhost session opened.,E1,T1,[],1700000000,0,Normal,Normal,,
"""

EXPECTED = (
    "2023-04-20T13:26:15+08:00 target-server proftpd[4311]: localhost session opened. "
    "tactic=Normal technique=Normal"
)


def test_render_row_exact_line():
    row = {
        "Time": "2023-04-20T13:26:15+08:00",
        "HostName": "target-server",
        "Component": "proftpd",
        "PID_or_IP": "4311",
        "Content": "localhost session opened.",
        "Tactic": "Normal",
        "Technique": "Normal",
    }
    assert syslog_cremev2.render_row(row) == EXPECTED


def test_stream_csv_to_txt(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(FIXTURE_CSV, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=syslog_cremev2.render_row,
        required_columns=syslog_cremev2._REQUIRED,
    )
    assert written == 1
    assert skipped == 0
    assert out_path.read_text(encoding="utf-8").strip() == EXPECTED


def test_stage_integration(tmp_path: Path):
    raw = tmp_path / "data/raw/cremev2"
    raw.mkdir(parents=True)
    (raw / "original_label_syslog.csv").write_text(FIXTURE_CSV, encoding="utf-8")
    written, skipped = syslog_cremev2.stage(tmp_path)
    assert written == 1
    assert skipped == 0
    assert (tmp_path / syslog_cremev2.STAGED_TXT).read_text(encoding="utf-8").strip() == EXPECTED
