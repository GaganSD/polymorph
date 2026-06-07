"""Phase 0b label-ceiling QA: the $0 SOTA kill gate."""

import json
from pathlib import Path

from polymorph_lamr.bench.label_ceiling import CeilingCounts, format_report, measure


def _write_distilled(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "distilled.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_measure_counts_needle_survival(tmp_path):
    # A chunk whose needle (a 5xx status) survives the teacher's compression but
    # would be dropped by keep-severity (it lives on a non-severe line).
    original = (
        "\n".join(f"INFO request ok latency={i}ms" for i in range(20))
        + "\nPOST /v1/checkout HTTP status=503 service unavailable"
    )
    compressed = "POST /v1/checkout HTTP status=503 service unavailable"
    path = _write_distilled(tmp_path, [{"original": original, "compressed": compressed, "src_path": "api.txt"}])

    c = measure(path)
    assert c.chunks == 1
    assert c.with_needle == 1
    assert c.needle_in_original == 1
    # Teacher kept the 503 line -> needle survives the teacher and the label.
    assert c.teacher_survived == 1
    assert c.label_survived == 1
    assert "http_status" in c.by_type


def test_skips_blank_and_malformed_and_empty(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        "\n".join(
            [
                "",
                "{ not json",
                json.dumps({"original": "", "compressed": "x"}),
                json.dumps({"original": "ERROR boom", "compressed": ""}),
            ]
        )
        + "\n"
    )
    c = measure(p)
    assert c.chunks == 0


def test_limit_caps_records(tmp_path):
    rows = [
        {"original": f"ERROR code {i} ValueError: bad", "compressed": f"ValueError: bad {i}", "src_path": "x"}
        for i in range(10)
    ]
    path = _write_distilled(tmp_path, rows)
    c = measure(path, limit=3)
    assert c.chunks == 3


def test_verdict_strings_cover_branches():
    # GATE FAILS: teacher below baseline.
    fail = CeilingCounts(needle_in_original=10, teacher_survived=4, keepsev_survived=8)
    assert "GATE FAILS" in format_report(fail)
    # CLEARS: teacher above baseline.
    clear = CeilingCounts(needle_in_original=10, teacher_survived=9, keepsev_survived=5)
    assert "GATE CLEARS" in format_report(clear)
    # MARGINAL: teacher == baseline.
    marg = CeilingCounts(needle_in_original=10, teacher_survived=7, keepsev_survived=7)
    assert "MARGINAL" in format_report(marg)
