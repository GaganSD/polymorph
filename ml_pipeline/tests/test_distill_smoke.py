"""Distillation smoke test using a mocked litellm.acompletion.

This test never hits a real API. The `RUN_REAL_API=1` opt-in path lives below
but is skipped unless the environment variable is set.
"""

import asyncio
import os
import sys
import types
from pathlib import Path

import pytest


def _install_fake_litellm(monkeypatch):
    fake = types.ModuleType("litellm")

    async def acompletion(model, messages, **kwargs):
        prompt = messages[0]["content"]
        # Extract the body after 'ORIGINAL:\n' as the 'compression'.
        body = prompt.split("ORIGINAL:\n", 1)[-1].split("\n\nCOMPRESSED:", 1)[0]
        # Mock 'extractive compression': just drop every other whitespace-token.
        toks = body.split()
        compressed = " ".join(toks[::2])
        return {
            "choices": [{"message": {"content": compressed}}],
        }

    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def test_distill_pair_with_mock(monkeypatch, tmp_path):
    _install_fake_litellm(monkeypatch)
    from polymorph_lamr.distill.client import DistillConfig, distill_pair, write_jsonl

    text = "alpha beta gamma delta epsilon zeta eta theta"
    cfg = DistillConfig(num_retries=0, request_timeout_s=5.0)
    result = asyncio.run(distill_pair(text, src_path="mem://", chunk_id=0, cfg=cfg))
    assert result.original == text
    assert result.claude
    assert result.gpt4o
    # Both compressed outputs must be subsets of the original word stream
    # (mock guarantees this).
    orig_words = set(text.split())
    for txt in (result.claude, result.gpt4o):
        for w in txt.split():
            assert w in orig_words

    out = tmp_path / "results.jsonl"
    write_jsonl([result], out)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 1


@pytest.mark.skipif(os.getenv("RUN_REAL_API") != "1", reason="real API opt-in")
def test_distill_pair_real_api():
    """Opt-in: hits Claude + GPT-4o on a tiny fixture. Budget < $0.05."""
    from polymorph_lamr.distill.client import DistillConfig, distill_pair

    text = "The quick brown fox jumps over the lazy dog. This sentence has redundant filler that should be removed."
    cfg = DistillConfig()
    result = asyncio.run(distill_pair(text, "real-api", 0, cfg))
    assert result.errors == [], f"errors: {result.errors}"
    orig_lower = text.lower()
    for word in result.claude.lower().split():
        # extractive: every compressed word should appear in the original
        assert word.strip(".,!?") in orig_lower
