"""Tests for the dedup-gate + stratified sampler.

Covers the allocation math, dedup/trash counting, deterministic selection, and a
full fixture-driven `build_sample` run (no network, no teacher calls).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polymorph_lamr.distill.sampler import (
    build_parser,
    build_sample,
    chunk_representatives,
    dedup_and_gate,
    main,
    select_chunks,
    water_fill,
    _stable_order,
)

# --- a structured template that collapses, repeated with varying num/blob -----
_BLOBS = [
    "RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z",
    "g8tprJBHkhLu3r6EtkU5E0Y51Gda8lfG5iCOMWoFH",
    "2eykZAqkscUJ4KUxihlYXskJiDSG3TF0CXhFXZp9Tf",
    "XSxiDN8XrGhYd9legdlv1fCtI9ILHaTM94tucUUNwL",
    "dfutS465iBuUtmcDowLmZts0LG4y70lpe4I6iefsVT",
]


def _cicd_line(i: int, *, tool: str, stage: str, ftype: str) -> str:
    return (
        f"2025-12-{(i % 28) + 1:02d}T07:58:16 MEDIUM {tool} stage={stage} "
        f"failure_type={ftype} error_code=ERR_{i} retry={i % 5} "
        f'msg="ERROR: {_BLOBS[i % len(_BLOBS)]}"'
    )


def test_water_fill_equal_split():
    alloc = water_fill({"a": 100, "b": 100, "c": 100}, 30, max_share=1.0)
    assert alloc == {"a": 10, "b": 10, "c": 10}


def test_water_fill_redistributes_from_small_corpus():
    # 'a' can only give 2; the surplus flows to b and c.
    alloc = water_fill({"a": 2, "b": 100, "c": 100}, 30, max_share=1.0)
    assert alloc["a"] == 2
    assert alloc["b"] + alloc["c"] == 28
    assert abs(alloc["b"] - alloc["c"]) <= 1


def test_water_fill_max_share_caps_domination():
    # One giant corpus must not exceed 25% of the target.
    alloc = water_fill({"a": 100000}, 100, max_share=0.25)
    assert alloc["a"] == 25  # capped, even though the pool is huge


def test_water_fill_balanced_under_cap():
    alloc = water_fill({"a": 1000, "b": 1000, "c": 1000, "d": 1000}, 100, max_share=0.25)
    assert alloc == {"a": 25, "b": 25, "c": 25, "d": 25}


def test_water_fill_target_exceeds_supply():
    alloc = water_fill({"a": 3, "b": 4}, 100, max_share=1.0)
    assert alloc == {"a": 3, "b": 4}  # take everything, no more


def test_water_fill_edge_cases():
    assert water_fill({}, 10) == {}
    assert water_fill({"a": 5}, 0) == {"a": 0}
    assert water_fill({"a": 0, "b": 0}, 10) == {"a": 0, "b": 0}


def test_water_fill_remainder_distributed_one_by_one():
    # 5 corpora × 10, target 12: first pass gives 2 each (=10); the leftover 2
    # rounds to share 0 and is handed out one slot at a time by headroom→name.
    alloc = water_fill({"a": 10, "b": 10, "c": 10, "d": 10, "e": 10}, 12, max_share=1.0)
    assert sum(alloc.values()) == 12
    assert sorted(alloc.values()) == [2, 2, 2, 3, 3]
    # Tie broken deterministically by name: earliest names get the extras.
    assert alloc["a"] == 3 and alloc["b"] == 3


def test_water_fill_deterministic():
    pools = {"a": 7, "b": 50, "c": 3, "d": 90}
    assert water_fill(pools, 40, max_share=0.5) == water_fill(pools, 40, max_share=0.5)


def test_dedup_collapses_templates_and_drops_trash():
    lines = []
    # 5 rows, same template A (collapse to 1).
    lines += [_cicd_line(i, tool="Jenkins", stage="deploy", ftype="Security Scan Failure") for i in range(5)]
    # 3 rows, template B (collapse to 1).
    lines += [_cicd_line(i, tool="GitLab", stage="build", ftype="Network Error") for i in range(3)]
    # 2 distinct pure-trash lines (no structural signal).
    lines.append(f"ERROR: {_BLOBS[0]}{_BLOBS[1]}")
    lines.append(f"FATAL exception {_BLOBS[2]}{_BLOBS[3]}")

    reps, raw, uniq, dropped = dedup_and_gate(iter(lines))
    assert raw == 10
    assert uniq == 4  # A, B, trash1, trash2
    assert dropped == 2  # both trash templates
    assert len(reps) == 2  # only A and B survive


def test_dedup_skips_blank_lines():
    reps, raw, uniq, dropped = dedup_and_gate(iter(["", "   ", "real log line value here"]))
    assert raw == 1
    assert len(reps) == 1


def test_chunk_representatives_empty():
    assert chunk_representatives([], max_tokens=128) == []


def test_chunk_representatives_packs():
    reps = [f"service=svc{i} status=ok latency={i}ms handled request fine" for i in range(20)]
    chunks = chunk_representatives(reps, max_tokens=32)
    assert len(chunks) > 1  # 20 lines at 32 tokens/chunk → multiple chunks
    # No content lost: every rep line appears somewhere.
    joined = "\n".join(chunks)
    for r in reps:
        assert r in joined


def test_select_chunks_returns_all_when_k_ge_len():
    chunks = ["a", "b", "c"]
    assert select_chunks(chunks, 5) == chunks


def test_select_chunks_exact_k_and_deterministic():
    chunks = [f"chunk number {i} with some words" for i in range(50)]
    a = select_chunks(chunks, 10)
    b = select_chunks(chunks, 10)
    assert len(a) == 10
    assert a == b  # deterministic
    assert set(a).issubset(set(chunks))


def test_stable_order_is_content_hash_not_position():
    chunks = ["zzz", "aaa", "mmm"]
    order = _stable_order(chunks)
    assert sorted(order) == [0, 1, 2]  # valid permutation of all indices
    # Order tracks CONTENT, not position: the same three strings shuffled yield
    # the same content sequence when their indices are applied.
    shuffled = ["aaa", "zzz", "mmm"]
    assert [chunks[i] for i in order] == [shuffled[i] for i in _stable_order(shuffled)]
    # And it is deterministic across calls.
    assert _stable_order(chunks) == order


# --- integration: build_sample over a fixture manifest -----------------------
def _write_staged(tmp_path: Path) -> Path:
    staged = tmp_path / "data" / "staged"
    staged.mkdir(parents=True)

    # cicd: 8 structured (2 templates) + 2 trash lines.
    cicd_lines = (
        [_cicd_line(i, tool="Jenkins", stage="deploy", ftype="Security Scan Failure") for i in range(8)]
        + [_cicd_line(i, tool="GitLab", stage="build", ftype="Network Error") for i in range(8)]
        + [f"ERROR: {_BLOBS[i % len(_BLOBS)]}{_BLOBS[(i + 1) % len(_BLOBS)]}" for i in range(4)]
    )
    (staged / "cicd.txt").write_text("\n".join(cicd_lines) + "\n", encoding="utf-8")

    # apache: distinct access lines — diversity is in the URL PATH (words), not
    # in numbers/IPs (which the template key masks). 6 real templates.
    paths = [
        "/api/users/profile",
        "/api/orders/checkout",
        "/health/ready",
        "/api/login/session",
        "/static/dashboard/app.js",
        "/metrics/prometheus/scrape",
    ]
    apache_lines = [
        f'10.0.0.{i} - - [27/Dec/2037:12:00:{i % 60:02d} +0530] "GET {paths[i % len(paths)]}" 200 {i * 13}'
        for i in range(30)
    ]
    (staged / "apache.txt").write_text("\n".join(apache_lines) + "\n", encoding="utf-8")

    manifest = [
        {"name": "cicd", "format": "cicd", "staged_path": "data/staged/cicd.txt"},
        {"name": "apache", "format": "apache_access", "staged_path": "data/staged/apache.txt"},
        {"name": "ghost", "format": "missing", "staged_path": "data/staged/does_not_exist.txt"},
    ]
    mpath = staged / "MANIFEST.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return mpath


def test_build_sample_end_to_end(tmp_path: Path):
    mpath = _write_staged(tmp_path)
    out = tmp_path / "data" / "sampled" / "v0.jsonl"
    summary = build_sample(
        mpath, out, target=20, max_tokens=48, max_share=0.6, min_ratio=0.30, root=tmp_path
    )

    assert out.exists()
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == summary.total_selected
    assert summary.total_selected <= 20
    for r in records:
        assert set(r) == {"corpus", "src_path", "chunk_id", "text"}
        assert r["corpus"] in {"cicd", "apache"}

    by_name = {s.name: s for s in summary.per_corpus}
    # cicd: 20 lines, 2 real templates + trash; dedup + trash gate applied.
    assert by_name["cicd"].dropped_trash >= 1
    assert by_name["cicd"].unique_templates < by_name["cicd"].raw_lines
    # ghost corpus resolved to nothing → zero selected, no crash.
    assert by_name["ghost"].selected == 0
    assert by_name["ghost"].chunk_pool == 0


def test_build_sample_source_glob(tmp_path: Path):
    bench = tmp_path / "data" / "bench" / "tt"
    bench.mkdir(parents=True)
    (bench / "a.txt").write_text("trace span service-a handled ok latency low\n", encoding="utf-8")
    (bench / "b.txt").write_text("trace span service-b handled ok latency high\n", encoding="utf-8")
    staged = tmp_path / "data" / "staged"
    staged.mkdir(parents=True)
    manifest = [{"name": "tt", "format": "traces", "source_glob": "data/bench/tt/*.txt"}]
    mpath = staged / "MANIFEST.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    out = tmp_path / "out.jsonl"
    summary = build_sample(mpath, out, target=10, max_tokens=64, root=tmp_path)
    assert summary.total_selected >= 1
    assert summary.per_corpus[0].raw_lines == 2


def test_build_sample_resolves_absolute_path(tmp_path: Path):
    # staged_path given as an absolute path must resolve directly.
    f = tmp_path / "abs_corpus.txt"
    f.write_text(
        "service=auth status=ok handled login request fine here\n"
        "service=auth status=ok handled logout request fine here\n",
        encoding="utf-8",
    )
    staged = tmp_path / "data" / "staged"
    staged.mkdir(parents=True)
    manifest = [{"name": "abs", "format": "x", "staged_path": str(f)}]
    mpath = staged / "MANIFEST.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    summary = build_sample(mpath, tmp_path / "out.jsonl", target=10, max_tokens=64, root=tmp_path)
    assert summary.per_corpus[0].raw_lines == 2
    assert summary.total_selected >= 1


def test_build_sample_rejects_non_list_manifest(tmp_path: Path):
    mpath = tmp_path / "MANIFEST.json"
    mpath.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        build_sample(mpath, tmp_path / "out.jsonl", root=tmp_path)


def test_main_missing_manifest_returns_1(tmp_path: Path, capsys):
    rc = main(["--manifest", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o.jsonl")])
    assert rc == 1
    assert "manifest not found" in capsys.readouterr().err


def test_main_end_to_end(tmp_path: Path):
    mpath = _write_staged(tmp_path)
    out = tmp_path / "sampled.jsonl"
    rc = main(
        [
            "--manifest", str(mpath),
            "--out", str(out),
            "--target", "15",
            "--max-tokens", "48",
            "--root", str(tmp_path),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.manifest.endswith("MANIFEST.json")
    assert args.target == 30000
    assert args.max_share == 0.25
    assert args.min_ratio == 0.30
