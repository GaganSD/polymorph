"""Tests for alibaba_gpu CSV adapter."""

from pathlib import Path

from polymorph_lamr.distill.adapters import alibaba_gpu
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt

FIXTURE_CSV = """\
job_name,organization,gpu_model,cpu_request,gpu_request,worker_num,submit_time,duration,job_type
239255,13,A10,20.0,1.0,1,0.0,2764799.0,HP
"""

EXPECTED = (
    "submit_time=0.0 job=239255 org=13 type=HP gpu_model=A10 gpu_request=1.0 "
    "cpu_request=20.0 workers=1 duration=2764799.0"
)


def test_render_row_exact_line():
    row = {
        "job_name": "239255",
        "organization": "13",
        "gpu_model": "A10",
        "cpu_request": "20.0",
        "gpu_request": "1.0",
        "worker_num": "1",
        "submit_time": "0.0",
        "duration": "2764799.0",
        "job_type": "HP",
    }
    assert alibaba_gpu.render_row(row) == EXPECTED


def test_stream_csv_to_txt(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(FIXTURE_CSV, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=alibaba_gpu.render_row,
        required_columns=alibaba_gpu._REQUIRED,
    )
    assert written == 1
    assert skipped == 0
    assert out_path.read_text(encoding="utf-8").strip() == EXPECTED


def test_stage_integration(tmp_path: Path):
    raw = tmp_path / "data/raw/alibaba_gpu"
    raw.mkdir(parents=True)
    (raw / "job_info_df.csv").write_text(FIXTURE_CSV, encoding="utf-8")
    written, skipped = alibaba_gpu.stage(tmp_path)
    assert written == 1
    assert skipped == 0
