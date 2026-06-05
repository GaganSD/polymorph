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

from .prompts import (
    CLAUDE_MAX_COMPRESSION,
    GPT4O_REASONING_PRESERVED,
    LOG_TRACE_EXTRACTIVE,
    render,
)
from .providers import DEFAULT_TEACHER_SPECS, resolve_routing


@dataclass
class DistillConfig:
    claude_model: str = "anthropic/claude-3-5-sonnet-latest"
    gpt_model: str = "openai/gpt-4o"
    num_retries: int = 4
    request_timeout_s: float = 60.0
    max_tokens: int = 2048
    temperature: float = 0.2
    failure_policy: str = "record"  # "record" or "raise"


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


async def _call_one(
    model: str,
    prompt: str,
    cfg: DistillConfig,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    aws_region: str | None = None,
) -> tuple[str, float, str | None]:
    """Single call. Returns (text, cost_usd, error_or_None).

    ``api_base``/``api_key`` are forwarded to litellm when set (OpenAI-compatible
    custom endpoints such as the Vercel AI Gateway); ``aws_region`` is forwarded as
    ``aws_region_name`` for Bedrock. When all are None, litellm falls back to its
    default per-provider credential resolution.
    """
    import litellm  # local import keeps tests light

    extra: dict[str, Any] = {}
    if api_base:
        extra["api_base"] = api_base
    if api_key:
        extra["api_key"] = api_key
    if aws_region:
        extra["aws_region_name"] = aws_region

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
                **extra,
            )
            text = _response_text(resp)
            cost = _response_cost(resp)
            if not text.strip():
                return "", cost, f"{model}: empty response content"
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
    if cfg.failure_policy not in {"record", "raise"}:
        raise ValueError(f"unknown failure_policy: {cfg.failure_policy}")
    claude_prompt = render(CLAUDE_MAX_COMPRESSION, text)
    gpt_prompt = render(GPT4O_REASONING_PRESERVED, text)

    claude_task = asyncio.create_task(_call_one(cfg.claude_model, claude_prompt, cfg))
    gpt_task = asyncio.create_task(_call_one(cfg.gpt_model, gpt_prompt, cfg))
    (claude_text, claude_cost, claude_err), (gpt_text, gpt_cost, gpt_err) = await asyncio.gather(
        claude_task, gpt_task
    )

    errs = [e for e in (claude_err, gpt_err) if e]
    if errs and cfg.failure_policy == "raise":
        raise RuntimeError("; ".join(errs))
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


def write_jsonl(results: list, out_path: Path) -> None:
    """Write any result objects exposing `.to_json()` (DistillResult or
    EnsembleResult)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(r.to_json() + "\n")


# --------------------------------------------------------------------------- #
# Multi-teacher ensemble (E3): fan out across the Bedrock teachers and keep the
# per-chunk best-QC output. This is the primary distillation path; the legacy
# `distill_pair` (Claude + GPT-4o) above is retained for back-compat.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TeacherSpec:
    """A single distillation teacher: a label + a litellm model id + routing.

    ``api_base``/``api_key_env`` are populated by :meth:`from_spec` based on the
    provider prefix; constructing directly (name+model only) keeps the legacy
    default-provider behaviour for tests and back-compat.
    """

    name: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    aws_region: str | None = None

    @classmethod
    def from_spec(cls, name: str, spec_model: str) -> "TeacherSpec":
        """Build a teacher from a provider-prefixed spec-string (e.g.
        ``bedrock/deepseek.v3.2``, ``bedrock/minimax.minimax-m2.1``, or the legacy
        ``vercel/alibaba/qwen3.7-max``). Enforces the Vercel strict-model guard
        via :func:`resolve_routing`."""
        r = resolve_routing(spec_model)
        return cls(
            name=name,
            model=r.model,
            api_base=r.api_base,
            api_key_env=r.api_key_env,
            aws_region=r.aws_region,
        )


def default_teachers() -> list[TeacherSpec]:
    return [TeacherSpec.from_spec(n, m) for (n, m) in DEFAULT_TEACHER_SPECS]


@dataclass
class EnsembleConfig:
    teachers: list[TeacherSpec] = field(default_factory=default_teachers)
    prompt_template: str = LOG_TRACE_EXTRACTIVE
    num_retries: int = 4
    request_timeout_s: float = 60.0
    max_tokens: int = 2048
    temperature: float = 0.0  # deterministic-leaning teachers
    failure_policy: str = "record"


@dataclass
class EnsembleResult:
    src_path: str
    chunk_id: int
    original: str
    outputs: dict[str, str]  # teacher_name -> compressed text
    compressed: str  # the selected best-QC output (training target)
    chosen_teacher: str
    qc: dict[str, float]  # vr/ag/mr/hr of the chosen output
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "src_path": self.src_path,
                "chunk_id": self.chunk_id,
                "original": self.original,
                "outputs": self.outputs,
                "compressed": self.compressed,
                "chosen_teacher": self.chosen_teacher,
                "qc": self.qc,
                "cost_usd": self.cost_usd,
                "errors": self.errors,
            }
        )


def select_best(original: str, outputs: dict[str, str]) -> tuple[str, str, dict[str, float]]:
    """Pick the best teacher output by QC. Strictly-extractive outputs (VR==0) win;
    among them, lowest Alignment Gap, then highest compression (shortest) as the
    tie-breaker. Cross-teacher agreement is NOT required (it is only an implicit
    tie-break via compression). Returns (teacher_name, text, qc_dict). Empty
    outputs yield ("", "", {})."""
    from polymorph_lamr.qc.metrics import QCRecord  # local import keeps module light

    scored = []
    for name, text in outputs.items():
        if text and text.strip():
            qc = QCRecord.compute(original, text)
            scored.append((name, text, qc))
    if not scored:
        return ("", "", {})
    extractive = [s for s in scored if s[2].vr == 0.0]
    pool = extractive if extractive else scored
    pool.sort(key=lambda s: (s[2].vr, s[2].ag, len(s[1])))
    name, text, qc = pool[0]
    return name, text, {"vr": qc.vr, "ag": qc.ag, "mr": qc.mr, "hr": qc.hr}


async def distill_ensemble(
    text: str,
    src_path: str = "",
    chunk_id: int = 0,
    cfg: EnsembleConfig | None = None,
) -> EnsembleResult:
    cfg = cfg or EnsembleConfig()
    if cfg.failure_policy not in {"record", "raise"}:
        raise ValueError(f"unknown failure_policy: {cfg.failure_policy}")
    prompt = render(cfg.prompt_template, text)

    # Reuse the single-call machinery (retries/cost/error handling).
    call_cfg = DistillConfig(
        num_retries=cfg.num_retries,
        request_timeout_s=cfg.request_timeout_s,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        failure_policy=cfg.failure_policy,
    )
    tasks = [
        asyncio.create_task(
            _call_one(
                t.model,
                prompt,
                call_cfg,
                api_base=t.api_base,
                api_key=os.environ.get(t.api_key_env) if t.api_key_env else None,
                aws_region=t.aws_region,
            )
        )
        for t in cfg.teachers
    ]
    results = await asyncio.gather(*tasks)

    outputs: dict[str, str] = {}
    errors: list[str] = []
    total_cost = 0.0
    for teacher, (out_text, cost, err) in zip(cfg.teachers, results):
        total_cost += cost
        if err:
            errors.append(f"{teacher.name}: {err}")
        if out_text:
            outputs[teacher.name] = out_text

    if not outputs and cfg.failure_policy == "raise":
        raise RuntimeError("all teachers failed: " + "; ".join(errors))

    chosen_teacher, compressed, qc = select_best(text, outputs)
    return EnsembleResult(
        src_path=src_path,
        chunk_id=chunk_id,
        original=text,
        outputs=outputs,
        compressed=compressed,
        chosen_teacher=chosen_teacher,
        qc=qc,
        cost_usd=total_cost,
        errors=errors,
    )


async def distill_ensemble_many(
    items: Sequence[tuple[str, str, int]],  # (text, src_path, chunk_id)
    cfg: EnsembleConfig | None = None,
    concurrency: int = 8,
) -> AsyncIterator[EnsembleResult]:
    """Yield EnsembleResults as each chunk completes. Cancels in-flight requests
    on teardown (prevents $$$ leak under Ctrl-C)."""
    cfg = cfg or EnsembleConfig()
    sem = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue[EnsembleResult] = asyncio.Queue()

    async def _bounded(text: str, path: str, idx: int) -> None:
        async with sem:
            result = await distill_ensemble(text, path, idx, cfg)
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
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _response_text(resp: Any) -> str:
    choices = _get(resp, "choices", [])
    if not choices:
        return ""
    first = choices[0]
    message = _get(first, "message", {})
    content = _get(message, "content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                text = _get(item, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(content or "")


def _response_cost(resp: Any) -> float:
    hidden = _get(resp, "_hidden_params", {}) or {}
    for source in (hidden, resp):
        for key in ("response_cost", "cost", "cost_usd"):
            value = _get(source, key, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
    usage = _get(resp, "usage", {}) or {}
    value = _get(usage, "cost", None) or _get(usage, "response_cost", None)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
