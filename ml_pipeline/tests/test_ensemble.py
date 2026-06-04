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


def test_default_teachers_routing():
    from polymorph_lamr.distill.client import default_teachers

    ts = {t.name: t for t in default_teachers()}
    # deepseek-v32 is the primary teacher via AWS Bedrock (no key, AWS chain)
    assert ts["deepseek-v32"].model == "bedrock/deepseek.v3.2"
    assert ts["deepseek-v32"].api_base is None
    assert ts["deepseek-v32"].api_key_env is None
    # kimi routes through OpenRouter (litellm-native, no custom base)
    assert ts["kimi"].model.startswith("openrouter/")
    assert ts["kimi"].api_base is None
    assert ts["kimi"].api_key_env == "OPENROUTER_API_KEY"


# ----- run_distill helpers + pair mode --------------------------------------


def test_parse_teachers():
    from polymorph_lamr.distill.run_distill import _parse_teachers

    assert _parse_teachers(None)  # falls back to defaults
    specs = _parse_teachers(["qwen=openrouter/qwen/q", "openrouter/x/y"])
    assert specs[0].name == "qwen" and specs[0].model == "openrouter/qwen/q"
    assert specs[0].api_key_env == "OPENROUTER_API_KEY"
    assert specs[1].name == "y" and specs[1].model == "openrouter/x/y"


def test_parse_teachers_vercel_guard():
    from polymorph_lamr.distill.run_distill import _parse_teachers

    # the strict guard fires through the CLI parsing path too
    with pytest.raises(ValueError):
        _parse_teachers(["bad=vercel/some/other-model"])
    ok = _parse_teachers(["q=vercel/alibaba/qwen3.7-max"])
    assert ok[0].model == "openai/alibaba/qwen3.7-max"
    assert ok[0].api_key_env == "VERCEL_AI_GATEWAY_KEY"


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
    rc = main(["--in", str(src), "--out", str(out),
               "--teachers", "a=openrouter/x/y", "--max-tokens", "32"])
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


# ----- provider routing + strict guard --------------------------------------


def test_resolve_routing_vercel_allowed():
    from polymorph_lamr.distill.providers import VERCEL_AI_GATEWAY_BASE, resolve_routing

    r = resolve_routing("vercel/alibaba/qwen3.7-max")
    assert r.model == "openai/alibaba/qwen3.7-max"  # litellm openai-compatible id
    assert r.api_base == VERCEL_AI_GATEWAY_BASE
    assert r.api_key_env == "VERCEL_AI_GATEWAY_KEY"


def test_resolve_routing_vercel_guard_rejects_other_models():
    from polymorph_lamr.distill.providers import resolve_routing

    with pytest.raises(ValueError):
        resolve_routing("vercel/openai/gpt-4o")
    with pytest.raises(ValueError):
        resolve_routing("vercel/alibaba/qwen-2.5-72b-instruct")


def test_resolve_routing_openrouter_and_passthrough():
    from polymorph_lamr.distill.providers import resolve_routing

    r = resolve_routing("openrouter/moonshotai/kimi-k2.6:free")
    assert r.model == "openrouter/moonshotai/kimi-k2.6:free"
    assert r.api_base is None and r.api_key_env == "OPENROUTER_API_KEY"

    p = resolve_routing("anthropic/claude-3-5-sonnet-latest")
    assert p.model == "anthropic/claude-3-5-sonnet-latest"
    assert p.api_base is None and p.api_key_env is None


def test_resolve_routing_bedrock(monkeypatch):
    from polymorph_lamr.distill.providers import resolve_routing

    monkeypatch.setenv("AWS_REGION", "eu-north-1")
    r = resolve_routing("bedrock/deepseek.v3.2")
    assert r.model == "bedrock/deepseek.v3.2"  # litellm-native, passed through
    assert r.api_base is None and r.api_key_env is None  # AWS credential chain
    assert r.aws_region == "eu-north-1"


def test_resolve_routing_bedrock_region_fallback(monkeypatch):
    from polymorph_lamr.distill.providers import resolve_routing

    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    assert resolve_routing("bedrock/deepseek.v3.2").aws_region == "us-west-2"
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    # no region env -> None (boto3 falls back to the active profile)
    assert resolve_routing("bedrock/deepseek.v3.2").aws_region is None


def test_call_one_forwards_aws_region(monkeypatch):
    captured = _capturing_litellm(monkeypatch)
    from polymorph_lamr.distill.client import DistillConfig, _call_one

    asyncio.run(
        _call_one("bedrock/deepseek.v3.2", "p", DistillConfig(num_retries=0),
                  aws_region="eu-north-1")
    )
    assert captured[0]["aws_region_name"] == "eu-north-1"
    assert "api_key" not in captured[0]  # bedrock uses the AWS chain, no key


def test_teacher_from_spec_guard():
    from polymorph_lamr.distill.client import TeacherSpec

    with pytest.raises(ValueError):
        TeacherSpec.from_spec("bad", "vercel/some/other-model")


def _capturing_litellm(monkeypatch, content="ok"):
    captured: list[dict] = []
    fake = types.ModuleType("litellm")

    async def acompletion(model, messages, **kwargs):
        captured.append({"model": model, **kwargs})
        return {"choices": [{"message": {"content": content}}]}

    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return captured


def test_call_one_forwards_api_base_and_key(monkeypatch):
    captured = _capturing_litellm(monkeypatch)
    from polymorph_lamr.distill.client import DistillConfig, _call_one

    text, _cost, err = asyncio.run(
        _call_one(
            "openai/alibaba/qwen3.7-max",
            "p",
            DistillConfig(num_retries=0),
            api_base="https://gw/v1",
            api_key="sek",
        )
    )
    assert err is None and text == "ok"
    assert captured[0]["api_base"] == "https://gw/v1"
    assert captured[0]["api_key"] == "sek"


def test_call_one_omits_routing_when_unset(monkeypatch):
    captured = _capturing_litellm(monkeypatch)
    from polymorph_lamr.distill.client import DistillConfig, _call_one

    asyncio.run(_call_one("m", "p", DistillConfig(num_retries=0)))
    assert "api_base" not in captured[0] and "api_key" not in captured[0]


def test_distill_ensemble_uses_teacher_routing(monkeypatch):
    captured = _capturing_litellm(monkeypatch, content="alpha gamma")
    monkeypatch.setenv("VERCEL_AI_GATEWAY_KEY", "vck-test")
    from polymorph_lamr.distill.client import (
        EnsembleConfig,
        TeacherSpec,
        distill_ensemble,
    )

    cfg = EnsembleConfig(
        teachers=[TeacherSpec.from_spec("qwen3-max", "vercel/alibaba/qwen3.7-max")],
        num_retries=0,
    )
    asyncio.run(distill_ensemble("alpha beta gamma delta", "m", 0, cfg))
    assert captured[0]["model"] == "openai/alibaba/qwen3.7-max"
    assert captured[0]["api_base"].endswith("ai-gateway.vercel.sh/v1")
    assert captured[0]["api_key"] == "vck-test"
