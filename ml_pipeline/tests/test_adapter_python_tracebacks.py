"""Tests for python_tracebacks JSON adapter."""

from pathlib import Path

from polymorph_lamr.distill.adapters import python_tracebacks
from polymorph_lamr.distill.adapters._common import stream_json_array_to_txt

FIXTURE_JSON = """\
{"infile": "/tmp/issues.db", "data": [
[1, "https://example.com/issues/1", "cpython", "Traceback (most recent call last):\\r  File \\"app.py\\", line 1\\rValueError: boom"],
[2, "https://example.com/issues/2", "cpython", "short message without traceback markers"]
]}
"""

EXPECTED = (
    'Traceback (most recent call last): File "app.py", line 1 ValueError: boom'
)


def test_render_item_exact_line():
    item = [
        1,
        "https://example.com/issues/1",
        "cpython",
        'Traceback (most recent call last):\r  File "app.py", line 1\rValueError: boom',
    ]
    assert python_tracebacks.render_item(item) == EXPECTED


def test_render_item_skips_invalid():
    assert python_tracebacks.render_item([1, "u", "cpython", "no markers"]) is None
    assert python_tracebacks.render_item([1, "u"]) is None


def test_stream_json_array_to_txt(tmp_path: Path):
    json_path = tmp_path / "stacktraces.json"
    json_path.write_text(FIXTURE_JSON, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_json_array_to_txt(
        json_path,
        out_path,
        array_key="data",
        render_item=python_tracebacks.render_item,
    )
    assert written == 1
    assert skipped == 1
    assert out_path.read_text(encoding="utf-8").strip() == EXPECTED
