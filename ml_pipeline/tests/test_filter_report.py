"""Filter report + IO paths."""

import json
from dataclasses import asdict
from pathlib import Path

from polymorph_lamr.qc.filter import (
    _top_pct_threshold,
    filter_records,
    records_to_jsonl,
    write_report,
)
from polymorph_lamr.qc.metrics import QCRecord


def test_top_pct_threshold_handles_empty():
    import math

    assert _top_pct_threshold([], 5.0) == math.inf


def test_filter_empty_records_returns_empty():
    survivors, report = filter_records([])
    assert survivors == []
    assert report["total"] == 0


def test_report_has_histograms():
    recs = [QCRecord.compute("alpha beta gamma " * 3, " ".join(["alpha", "beta"] + [f"x{i}" for i in range(i)])) for i in range(8)]
    _, report = filter_records(recs, vr_hard_floor=1.0)
    assert "vr_hist" in report and len(report["vr_hist"]) == 10
    assert "ag_hist" in report and len(report["ag_hist"]) == 10
    # vr_hist is computed AFTER the hard floor; with vr_hard_floor=1.0 all survive.
    assert sum(report["vr_hist"]) == len(recs)


def test_records_to_jsonl_roundtrip(tmp_path):
    rec = QCRecord.compute("alpha beta", "alpha")
    out = tmp_path / "qc.jsonl"
    records_to_jsonl([rec], out)
    line = out.read_text().strip()
    blob = json.loads(line)
    assert blob["original"] == "alpha beta"
    assert blob["compressed"] == "alpha"


def test_write_report_creates_parent(tmp_path):
    out = tmp_path / "nested" / "report.json"
    write_report({"hello": "world"}, out)
    assert out.exists()
    assert json.loads(out.read_text())["hello"] == "world"


def test_metrics_handle_empty_inputs():
    from polymorph_lamr.qc.metrics import alignment_gap, hitting_rate, matching_rate, variation_rate

    assert variation_rate("", "") == 0.0
    assert matching_rate("", "anything") == 0.0
    assert hitting_rate("", "") == 0.0
    # AG well-defined on empty.
    assert alignment_gap("", "") == 0.0
