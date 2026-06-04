"""OpenRouter multi-teacher ensemble: select_best, distill_ensemble[_many],
_parse_teachers, pair-mode CLI, and log-aware chunking."""

import asyncio
import json
import sys
import types

import pytest


def _fake_litellm(monkeypatch, raiser=False):
    fake = types.ModuleType("litellm")

    async def acompletion(model, messages, **kwargs):
        if raiser:
            raise RuntimeError("provider down")
        body = messages[0]["content"].split("ORIGINAL:\n", 1)[-1].split("\n\nCOMPRESSED:", 1)[0]
        toks = body.split()
        return {"choices": [{"message": {"content": " ".join(toks[::2])}}]}

    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)


# ----- select_best ----------------------------------------------------------


def test_select_best_empty_outputs():
    from polymorph_lamr.distill.client import select_best

    assert select_best("orig", {}) == ("", "", {})


def test_select_best_empty_strings_skipped():
    from polymorph_lamr.distill.client import select_best

    assert select_best("a b c", {"t": "   "}) == ("", "", {})


def test_select_best_prefers_extractive():
    from polymorph_lamr.distill.client import select_best

    original = "alpha beta gamma delta epsilon"
    outputs = {
        "good": "alpha gamma epsilon",  # strict subsequence -> VR 0
        "bad": "alpha zzz gamma",       # 'zzz' is novel -> VR > 0
    }
    name, text, qc = select_best(original, outputs)
    assert name == "good"
    assert qc["vr"] == 0.0
    assert text == "alpha gamma epsilon"


def test_select_best_falls_back_when_none_extractive():
    from polymorph_lamr.distill.client import select_best

    name, text, qc = select_best("a b c", {"t1": "x y", "t2": "z"})
    assert name in ("t1", "t2")
    assert text
    assert "vr" in qc


# ----- distill_ensemble -----------------------------------------------------


def test_distill_ensemble_aggregates(monkeypatch):
    _fake_litellm(monkeypatch)
    from polymorph_lamr.distill.client import EnsembleConfig, TeacherSpec, distill_ensemble

    cfg = EnsembleConfig(
        teachers=[TeacherSpec("a", "m/a"), TeacherSpec("b", "m/b")], num_retries=0
    )
    r = asyncio.run(distill_ensemble("one two three four five six", "mem", 0, cfg))
    assert set(r.outputs) <= {"a", "b"}
    assert r.outputs  # at least one teacher produced output
    assert r.compressed
    assert r.chosen_teacher in r.outputs
    assert {"vr", "ag", "mr", "hr"} <= set(r.qc)
    # to_json round-trips
    assert json.loads(r.to_json())["chosen_teacher"] == r.chosen_teacher


def test_distill_ensemble_all_fail_record(monkeypatch):
    _fake_litellm(monkeypatch, raiser=True)
    from polymorph_lamr.distill.client import EnsembleConfig, TeacherSpec, distill_ensemble

    cfg = EnsembleConfig(
        teachers=[TeacherSpec("a", "m/a")], num_retries=0, failure_policy="record"
    )
    r = asyncio.run(distill_ensemble("text here", "mem", 0, cfg))
    assert r.outputs == {}
    assert r.compressed == ""
    assert r.errors


def test_distill_ensemble_all_fail_raise(monkeypatch):
    _fake_litellm(monkeypatch, raiser=True)
    from polymorph_lamr.distill.client import EnsembleConfig, TeacherSpec, distill_ensemble

    cfg = EnsembleConfig(
        teachers=[TeacherSpec("a", "m/a")], num_retries=0, failure_policy="raise"
    )
    with pytest.raises(RuntimeError):
        asyncio.run(distill_ensemble("text", "mem", 0, cfg))


def test_distill_ensemble_bad_failure_policy():
    from polymorph_lamr.distill.client import EnsembleConfig, distill_ensemble

    cfg = EnsembleConfig(failure_policy="explode")
    with pytest.raises(ValueError):
        asyncio.run(distill_ensemble("text", "mem", 0, cfg))


def test_distill_ensemble_many_yields_all(monkeypatch):
    _fake_litellm(monkeypatch)
    from polymorph_lamr.distill.client import (
        EnsembleConfig,
        TeacherSpec,
        distill_ensemble_many,
    )

    cfg = EnsembleConfig(teachers=[TeacherSpec("a", "m/a")], num_retries=0)
    items = [("alpha beta gamma delta", "f", i) for i in range(3)]

    async def collect():
        return [r async for r in distill_ensemble_many(items, cfg=cfg, concurrency=2)]

    assert len(asyncio.run(collect())) == 3


def test_default_teachers_are_openrouter():
    from polymorph_lamr.distill.client import default_teachers

    ts = default_teachers()
    assert ts and all(t.model.startswith("openrouter/") for t in ts)


# ----- run_distill helpers + pair mode --------------------------------------


def test_parse_teachers():
    from polymorph_lamr.distill.run_distill import _parse_teachers

    assert _parse_teachers(None)  # falls back to defaults
    specs = _parse_teachers(["qwen=openrouter/qwen/q", "openrouter/x/y"])
    assert specs[0].name == "qwen" and specs[0].model == "openrouter/qwen/q"
    assert specs[1].name == "y" and specs[1].model == "openrouter/x/y"


def test_run_distill_pair_mode(monkeypatch, tmp_path):
    _fake_litellm(monkeypatch)
    from polymorph_lamr.distill.run_distill import main

    src = tmp_path / "s"
    src.mkdir()
    (src / "a.md").write_text("alpha beta gamma delta epsilon zeta eta theta")
    out = tmp_path / "o.jsonl"
    rc = main(["--in", str(src), "--out", str(out), "--mode", "pair", "--max-tokens", "32"])
    assert rc == 0
    rec = json.loads(out.read_text().strip().split("\n")[0])
    assert "claude" in rec and "gpt4o" in rec  # legacy pair schema


def test_run_distill_ensemble_warns_without_key(monkeypatch, tmp_path, capsys):
    _fake_litellm(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from polymorph_lamr.distill.run_distill import main

    src = tmp_path / "s"
    src.mkdir()
    (src / "a.log").write_text("svc up\nsvc up\nsvc up\nsvc up\n")
    out = tmp_path / "o.jsonl"
    rc = main(["--in", str(src), "--out", str(out), "--teachers", "a=m/a", "--max-tokens", "32"])
    assert rc == 0
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


# ----- chunker log mode -----------------------------------------------------


def test_chunker_log_mode_and_detect():
    from polymorph_lamr.distill.chunker import chunk, detect_mode

    assert detect_mode("x.log", "") == "log"
    assert detect_mode("x.jsonl", "") == "log"
    # many short newline records -> log heuristic for .txt
    txt = "\n".join(f"short line {i}" for i in range(20))
    assert detect_mode("x.txt", txt) == "log"
    chunks = chunk("l1\nl2\nl3\nl4\nl5", max_tokens=4, mode="log")
    assert chunks and all(c.strip() for c in chunks)


def test_chunker_hard_split_oversized_unit():
    from polymorph_lamr.distill.chunker import chunk

    big = "word " * 500  # one prose unit far over the cap
    chunks = chunk(big, max_tokens=16, mode="prose")
    assert len(chunks) > 1
