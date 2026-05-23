"""Async distillation client over litellm.

`distill_pair(text)` fires both teachers concurrently and returns a dict with
both compressed variants. Retries are delegated to litellm's `num_retries`
plus an outer asyncio backoff for non-retried errors.

The litellm import is kept inside the function so importing this module
during tests doesn't drag in the heavy SDK.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompts import CLAUDE_MAX_COMPRESSION, GPT4O_REASONING_PRESERVED, render


@dataclass
class DistillConfig:
    claude_model: str = "anthropic/claude-3-5-sonnet-latest"
    gpt_model: str = "openai/gpt-4o"
    num_retries: int = 4
    request_timeout_s: float = 60.0
    max_tokens: int = 2048
    temperature: float = 0.2


@dataclass
class DistillResult:
    src_path: str
    chunk_id: int
    original: str
    claude: str
    gpt4o: str
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "src_path": self.src_path,
                "chunk_id": self.chunk_id,
                "original": self.original,
                "claude": self.claude,
                "gpt4o": self.gpt4o,
                "cost_usd": self.cost_usd,
                "errors": self.errors,
            }
        )


async def _call_one(model: str, prompt: str, cfg: DistillConfig) -> tuple[str, float, str | None]:
    """Single call. Returns (text, cost_usd, error_or_None)."""
    import litellm  # local import keeps tests light

    last_err: Exception | None = None
    for attempt in range(cfg.num_retries + 1):
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.request_timeout_s,
                num_retries=0,  # we do the outer retry
            )
            text = resp["choices"][0]["message"]["content"] or ""
            cost = 0.0
            try:
                cost = float(getattr(resp, "_hidden_params", {}).get("response_cost") or 0.0)
            except Exception:
                cost = 0.0
            return text.strip(), cost, None
        except Exception as e:
            last_err = e
            if attempt < cfg.num_retries:
                backoff = (2 ** attempt) + random.random()
                await asyncio.sleep(backoff)
    return "", 0.0, f"{type(last_err).__name__}: {last_err}"


async def distill_pair(
    text: str,
    src_path: str = "",
    chunk_id: int = 0,
    cfg: DistillConfig | None = None,
) -> DistillResult:
    cfg = cfg or DistillConfig()
    claude_prompt = render(CLAUDE_MAX_COMPRESSION, text)
    gpt_prompt = render(GPT4O_REASONING_PRESERVED, text)

    claude_task = asyncio.create_task(_call_one(cfg.claude_model, claude_prompt, cfg))
    gpt_task = asyncio.create_task(_call_one(cfg.gpt_model, gpt_prompt, cfg))
    (claude_text, claude_cost, claude_err), (gpt_text, gpt_cost, gpt_err) = await asyncio.gather(
        claude_task, gpt_task
    )

    errs = [e for e in (claude_err, gpt_err) if e]
    return DistillResult(
        src_path=src_path,
        chunk_id=chunk_id,
        original=text,
        claude=claude_text,
        gpt4o=gpt_text,
        cost_usd=claude_cost + gpt_cost,
        errors=errs,
    )


async def distill_many(
    items: Sequence[tuple[str, str, int]],  # (text, src_path, chunk_id)
    cfg: DistillConfig | None = None,
    concurrency: int = 8,
) -> AsyncIterator[DistillResult]:
    """Yield results as each completes. On cancellation, all in-flight LLM
    requests are cancelled before propagating (prevents $$$ leak under Ctrl-C).
    """
    cfg = cfg or DistillConfig()
    sem = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue[DistillResult] = asyncio.Queue()

    async def _bounded(text: str, path: str, idx: int) -> None:
        async with sem:
            result = await distill_pair(text, path, idx, cfg)
            await queue.put(result)

    pending: list[asyncio.Task] = [
        asyncio.create_task(_bounded(t, p, i)) for (t, p, i) in items
    ]
    try:
        produced = 0
        total = len(pending)
        while produced < total:
            result = await queue.get()
            produced += 1
            yield result
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        # Drain cancellation exceptions so the event loop doesn't warn.
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


def write_jsonl(results: list[DistillResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(r.to_json() + "\n")
