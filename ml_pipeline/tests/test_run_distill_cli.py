"""CLI for run_distill: file discovery, chunking, mocked litellm round-trip."""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest


def _install_fake_litellm(monkeypatch):
    fake = types.ModuleType("litellm")

    async def acompletion(model, messages, **kwargs):
        body = messages[0]["content"].split("ORIGINAL:\n", 1)[-1].split("\n\nCOMPRESSED:", 1)[0]
        toks = body.split()
        return {"choices": [{"message": {"content": " ".join(toks[::2])}}]}

    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def test_run_distill_end_to_end(monkeypatch, tmp_path):
    _install_fake_litellm(monkeypatch)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("alpha beta gamma delta. epsilon zeta eta theta.")
    (src / "b.py").write_text("def f(x):\n    return x\n\ndef g(y):\n    return y\n")

    out = tmp_path / "distilled.jsonl"

    from polymorph_lamr.distill.run_distill import main

    rc = main(
        [
            "--in", str(src),
            "--out", str(out),
            "--concurrency", "2",
            "--max-tokens", "32",
        ]
    )
    assert rc == 0
    lines = out.read_text().strip().split("\n")
    assert len(lines) >= 2
    for line in lines:
        rec = json.loads(line)
        # Ensemble schema (OpenRouter multi-teacher; default mode).
        assert rec["original"]
        assert "compressed" in rec
        assert isinstance(rec["outputs"], dict) and rec["outputs"]
        assert rec["chosen_teacher"] in rec["outputs"]
        assert set(rec["qc"]) >= {"vr", "ag", "mr", "hr"}


def test_run_distill_empty_dir_exits_nonzero(monkeypatch, tmp_path, capsys):
    _install_fake_litellm(monkeypatch)
    src = tmp_path / "empty"
    src.mkdir()
    out = tmp_path / "out.jsonl"
    from polymorph_lamr.distill.run_distill import main

    rc = main(["--in", str(src), "--out", str(out)])
    assert rc != 0
    assert "no items" in capsys.readouterr().err.lower()


def test_distill_pair_retries_then_succeeds(monkeypatch):
    """First call fails, second succeeds — verifies the outer asyncio backoff loop."""
    fake = types.ModuleType("litellm")
    attempts = {"n": 0}

    async def acompletion(model, messages, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient")
        return {"choices": [{"message": {"content": "ok"}}]}

    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)

    from polymorph_lamr.distill.client import DistillConfig, distill_pair

    cfg = DistillConfig(num_retries=2, request_timeout_s=1.0)
    # Both providers share the same fake, so attempts double — that's fine.
    result = asyncio.run(distill_pair("hello world", "mem", 0, cfg))
    assert result.errors == []
    assert result.claude == "ok"


def test_distill_pair_records_errors_after_exhausting_retries(monkeypatch):
    fake = types.ModuleType("litellm")

    async def acompletion(model, messages, **kwargs):
        raise RuntimeError("permanent")

    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)

    from polymorph_lamr.distill.client import DistillConfig, distill_pair

    cfg = DistillConfig(num_retries=1, request_timeout_s=1.0)
    result = asyncio.run(distill_pair("text", "mem", 0, cfg))
    assert len(result.errors) == 2  # one per provider
    assert all("permanent" in e for e in result.errors)


def test_distill_pair_parses_object_response_and_cost(monkeypatch):
    fake = types.ModuleType("litellm")

    class Message:
        content = "object ok"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]
        _hidden_params = {"response_cost": 0.25}

    async def acompletion(model, messages, **kwargs):
        return Response()

    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)

    from polymorph_lamr.distill.client import DistillConfig, distill_pair

    result = asyncio.run(distill_pair("text", "mem", 0, DistillConfig(num_retries=0)))
    assert result.errors == []
    assert result.claude == "object ok"
    assert result.gpt4o == "object ok"
    assert result.cost_usd == 0.5


def test_distill_pair_parses_dict_cost_from_usage(monkeypatch):
    fake = types.ModuleType("litellm")

    async def acompletion(model, messages, **kwargs):
        return {"choices": [{"message": {"content": "dict ok"}}], "usage": {"cost": "0.125"}}

    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)

    from polymorph_lamr.distill.client import DistillConfig, distill_pair

    result = asyncio.run(distill_pair("text", "mem", 0, DistillConfig(num_retries=0)))
    assert result.errors == []
    assert result.cost_usd == 0.25


def test_distill_pair_raise_policy_errors(monkeypatch):
    fake = types.ModuleType("litellm")

    async def acompletion(model, messages, **kwargs):
        return {"choices": [{"message": {"content": ""}}]}

    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)

    from polymorph_lamr.distill.client import DistillConfig, distill_pair

    cfg = DistillConfig(num_retries=0, failure_policy="raise")
    with pytest.raises(RuntimeError, match="empty response"):
        asyncio.run(distill_pair("text", "mem", 0, cfg))


def test_prompts_render_inlines_text():
    from polymorph_lamr.distill.prompts import CLAUDE_MAX_COMPRESSION, GPT4O_REASONING_PRESERVED, render

    text = "the body"
    c = render(CLAUDE_MAX_COMPRESSION, text)
    g = render(GPT4O_REASONING_PRESERVED, text)
    assert "the body" in c
    assert "the body" in g
    assert "COMPRESSED:" in c
    assert "COMPRESSED:" in g
