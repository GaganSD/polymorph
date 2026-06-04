"""Tests for adapter shared utilities and manifest builder."""

import json
from pathlib import Path

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    has_required_columns,
    row_field,
    sanitize_field,
    stream_json_array_to_txt,
)
from polymorph_lamr.distill.adapters.manifest import build_manifest, write_manifest


def test_sanitize_field_replaces_newlines():
    assert sanitize_field("a\nb\rc") == "a b c"
    assert sanitize_field(None) == ""


def test_collapse_whitespace():
    assert collapse_whitespace("  a   b  c  ") == "a b c"


def test_has_required_columns():
    row = {"a": "1", "b": "2"}
    assert has_required_columns(row, ["a", "b"])
    assert not has_required_columns(row, ["a", "c"])


def test_row_field_missing_column():
    assert row_field({"a": "1"}, "a") == "1"
    assert row_field({"a": "1"}, "missing") == ""


def test_stream_json_array_to_txt(tmp_path: Path):
    json_path = tmp_path / "data.json"
    json_path.write_text(
        '{"meta": 1, "data": [[1, "ok"], [2, "skip"]]}',
        encoding="utf-8",
    )
    out_path = tmp_path / "out.txt"

    def render(item):
        if item[1] == "ok":
            return f"line={item[0]}"
        return None

    written, skipped = stream_json_array_to_txt(
        json_path, out_path, array_key="data", render_item=render
    )
    assert written == 1
    assert skipped == 1
    assert out_path.read_text(encoding="utf-8").strip() == "line=1"


def test_build_manifest_staged_and_referenced(tmp_path: Path):
    staged = tmp_path / "data/staged/sample.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("line one\nline two\nline three\n", encoding="utf-8")

    bench = tmp_path / "data/bench/trainticket_logs"
    bench.mkdir(parents=True)
    (bench / "a.txt").write_text("tt line 1\ntt line 2\n", encoding="utf-8")

    apache = tmp_path / "data/raw/server_logs"
    apache.mkdir(parents=True)
    (apache / "logfiles.log").write_text("apache 1\napache 2\napache 3\n", encoding="utf-8")

    staged_entries = [
        {
            "name": "sample",
            "source": "data/raw/sample.csv",
            "staged_path": "data/staged/sample.txt",
            "skipped_rows": 0,
        }
    ]
    manifest = build_manifest(tmp_path, staged_entries)
    assert len(manifest) == 3
    assert manifest[0]["name"] == "sample"
    assert manifest[0]["line_count"] == 3
    assert len(manifest[0]["samples"]) == 3
    assert manifest[1]["name"] == "trainticket_traces"
    assert "source_glob" in manifest[1]
    assert manifest[2]["name"] == "apache_access"
    assert "staged_path" not in manifest[2]
    assert manifest[2]["source"] == "data/raw/server_logs/logfiles.log"

    out = write_manifest(tmp_path, manifest)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded[0]["samples"][0] == "line one"
