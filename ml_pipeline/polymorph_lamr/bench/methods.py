"""Compression methods compared by the answer-survival benchmark.

Each method takes a log chunk and a target drop rate R (fraction of tokens to
remove) and returns the compressed text. Survival is then measured on that text
(see ``survival.py``). Methods that aren't rate-tunable (deterministic dedup)
ignore R and report a single achieved ratio.

Always-available (no extra deps, GPU-free, deterministic):
  * ``DeterministicDedup``    — line-normalize + run-length collapse (a faithful
                                Python mirror of the Rust ``src/dedup.rs`` sweep).
  * ``KeepSeverityHeuristic`` — keep the most-severe lines until the token budget.
  * ``RandomDropFloor``       — deterministic pseudo-random token drop (the floor
                                any real method must beat).

Optional (skipped with a reason if the dep/model is absent):
  * ``LLMLingua2Method`` — Microsoft LLMLingua-2 token compression (needs the
                           ``llmlingua`` package; runs on CPU).
  * ``LaMRMethod``       — our trained pruner at the target rate (needs a
                           checkpoint compatible with the current single-logit
                           architecture).
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import tiktoken

_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def token_count(text: str) -> int:
    return len(_enc().encode(text))


class CompressionMethod(Protocol):
    name: str
    tunable: bool  # does it respond to the target drop rate?

    def available(self) -> tuple[bool, str]:
        """(is_available, reason_if_not)."""
        ...

    def compress(self, text: str, target_drop_rate: float) -> str:
        ...


# ---------------------------------------------------------------------------
# Deterministic dedup (mirror of src/dedup.rs)
# ---------------------------------------------------------------------------

# Ordered normalization patterns — the same classes src/dedup.rs masks, so two
# lines differing only in their variable parts collapse to the same key.
_NORM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    (re.compile(r"\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s*[+-]\d{4}"), "<TS>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b\d+\b"), "<NUM>"),
]


def _normalize_key(line: str) -> str:
    key = line
    for pat, repl in _NORM_PATTERNS:
        key = pat.sub(repl, key)
    return key.strip()


@dataclass
class DeterministicDedup:
    """Collapse runs of consecutive lines sharing a normalized template into
    head + "(N lines elided)" + tail. Reversible in production (the Rust path
    caches the elided middle); here we only need the reduced text for survival.
    """

    name: str = "deterministic"
    tunable: bool = False
    min_run: int = 3  # runs longer than head+tail get an elision summary

    def available(self) -> tuple[bool, str]:
        return True, ""

    def compress(self, text: str, target_drop_rate: float) -> str:  # noqa: ARG002
        lines = text.splitlines()
        if not lines:
            return text
        out: list[str] = []
        i = 0
        n = len(lines)
        while i < n:
            key = _normalize_key(lines[i])
            j = i + 1
            while j < n and _normalize_key(lines[j]) == key:
                j += 1
            run = j - i
            if run >= self.min_run:
                out.append(lines[i])
                out.append(f"... {run - 2} lines elided ...")
                out.append(lines[j - 1])
            else:
                out.extend(lines[i:j])
            i = j
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Keep-severity heuristic
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = ["FATAL", "CRITICAL", "ERROR", "EXCEPTION", "TRACEBACK", "WARN"]


def _severity_rank(line: str) -> int:
    up = line.upper()
    for rank, kw in enumerate(_SEVERITY_ORDER):
        if kw in up:
            return rank
    return len(_SEVERITY_ORDER)  # non-severe


@dataclass
class KeepSeverityHeuristic:
    """Keep the most-severe lines first until the keep budget (1 - R of the input
    tokens) is spent; emit kept lines in original order. A simple, strong baseline
    for "just keep the errors."
    """

    name: str = "keep-severity"
    tunable: bool = True

    def available(self) -> tuple[bool, str]:
        return True, ""

    def compress(self, text: str, target_drop_rate: float) -> str:
        lines = text.splitlines()
        if not lines:
            return text
        total = token_count(text)
        budget = max(1, round((1.0 - target_drop_rate) * total))
        # Stable priority: (severity_rank, original_index) — severe lines first,
        # ties broken by original order.
        order = sorted(range(len(lines)), key=lambda idx: (_severity_rank(lines[idx]), idx))
        keep: set[int] = set()
        spent = 0
        for idx in order:
            cost = token_count(lines[idx]) + 1  # +1 ~ newline
            if spent + cost > budget and keep:
                continue
            keep.add(idx)
            spent += cost
            if spent >= budget:
                break
        return "\n".join(lines[i] for i in sorted(keep))


# ---------------------------------------------------------------------------
# Random-drop floor (deterministic)
# ---------------------------------------------------------------------------

def _decode_with_drop_order(
    ids: list[int],
    drop_order: list[int],
    target_drop_rate: float,
    force_keep: list[bool] | None = None,
    tokenizer: str = "cl100k",
) -> str:
    """Drop the first round(R*n) tokens in ``drop_order`` (most-droppable first),
    skipping any token marked in ``force_keep`` (the structural floor), then decode
    the survivors. Force-kept tokens are never dropped — at high R this can leave
    the achieved drop below target, which is the intended high-recall trade.
    """
    from ..label.align import decode_tokens

    n = len(ids)
    if n == 0:
        return decode_tokens(ids, tokenizer)
    k = min(n, max(0, round(max(0.0, min(1.0, target_drop_rate)) * n)))
    fk = force_keep or [False] * n
    dropped: set[int] = set()
    for i in drop_order:
        if len(dropped) >= k:
            break
        if fk[i]:
            continue
        dropped.add(i)
    return decode_tokens([tid for i, tid in enumerate(ids) if i not in dropped], tokenizer)


@dataclass
class RandomDropFloor:
    """Deterministically drop ~R of the tokens (keyed by content + index). The
    floor a real ranker must beat: same compression ratio, blind to salience.

    With ``floor=True`` the structural locker (``structural.py``) force-keeps
    high-salience atoms. A *blind* ranker plus the floor beating keep-severity is
    the non-circular demonstration that a token-level structural prior keeps
    needles that a line-level severity prior drops.
    """

    name: str = "random"
    tunable: bool = True
    floor: bool = False
    span: str | None = None        # None = token-level; else 'word' / 'chunk:N'
    aggregator: str = "max"        # empirical + Rust default (see spandecode.py)

    def __post_init__(self) -> None:
        suffix = ""
        if self.span:
            suffix += "+span"
        if self.floor:
            suffix += "+floor"
        if suffix and self.name == "random":
            self.name = "random" + suffix

    def available(self) -> tuple[bool, str]:
        return True, ""

    def compress(self, text: str, target_drop_rate: float) -> str:
        force_keep: list[bool] | None = None
        spans: list[tuple[int, int]] | None = None
        if self.floor:
            from .structural import structural_keep_mask

            ids, spans, force_keep = structural_keep_mask(text)
        elif self.span:
            from ..label.align import encode_with_spans

            ids, spans = encode_with_spans(text)
        else:
            ids = _enc().encode(text)
        if not ids:
            return text
        seed = zlib.crc32(text.encode("utf-8", "ignore"))
        # Deterministic pseudo-random per-token "drop-prob". The token-level path
        # keeps the original ascending-crc rule (most-droppable = lowest crc), so
        # mapping crc -> drop_prob must be DESCENDING in crc to match it.
        crc = [zlib.crc32(f"{seed}:{i}".encode()) & 0xFFFF_FFFF for i in range(len(ids))]
        if self.span:
            from .spandecode import span_decode

            # Higher drop_prob = more droppable. Lower crc was more droppable, so
            # invert: drop_prob = 1 - crc/2^32.
            drop_probs = [1.0 - (c / float(0x1_0000_0000)) for c in crc]
            return span_decode(
                ids, spans, text, drop_probs, target_drop_rate,
                span=self.span, aggregator=self.aggregator, force_keep=force_keep,
            )
        # Most-droppable first: ascending crc (mirrors the old threshold rule).
        order = sorted(range(len(ids)), key=lambda i: crc[i])
        return _decode_with_drop_order(ids, order, target_drop_rate, force_keep)


# ---------------------------------------------------------------------------
# LLMLingua-2 (optional)
# ---------------------------------------------------------------------------

@dataclass
class LLMLingua2Method:
    """Microsoft LLMLingua-2 token compression at the target rate. CPU-only.

    Lazily constructs the compressor on first use; ``available`` reports whether
    the package imports (the model is downloaded on first compress).
    """

    name: str = "llmlingua2"
    tunable: bool = True
    model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    _compressor: object | None = None

    def available(self) -> tuple[bool, str]:
        try:
            import llmlingua  # noqa: F401
            return True, ""
        except Exception as e:  # pragma: no cover - env-dependent
            return False, f"llmlingua not importable: {e}"

    def _get(self):
        if self._compressor is None:  # pragma: no cover - heavy / env-dependent
            from llmlingua import PromptCompressor

            self._compressor = PromptCompressor(
                model_name=self.model_name, use_llmlingua2=True, device_map="cpu"
            )
        return self._compressor

    def compress(self, text: str, target_drop_rate: float) -> str:  # pragma: no cover
        rate = max(0.0, min(1.0, 1.0 - target_drop_rate))  # llmlingua "rate" = fraction kept
        comp = self._get()
        result = comp.compress_prompt(text, rate=rate, force_tokens=["\n"])
        return result.get("compressed_prompt", "") if isinstance(result, dict) else str(result)


# ---------------------------------------------------------------------------
# LaMR (our trained pruner; optional)
# ---------------------------------------------------------------------------

@dataclass
class LaMRMethod:
    """Our trained pruner: encode -> per-token drop prob -> drop the top-R by
    probability (the calibrated target-rate decode) -> decode the kept tokens.

    Needs a checkpoint compatible with the current single-logit architecture.
    """

    ckpt: Path
    name: str = "lamr"
    tunable: bool = True
    max_seq_len: int = 512  # inference window = training window (configs/modernbert.yaml)
    floor: bool = False  # apply the structural insurance floor at decode
    span: str | None = None        # None = token-level; else 'word' / 'chunk:N'
    aggregator: str = "max"        # empirical + Rust default (see spandecode.py)
    # Token space for encode/floor/decode. "auto" reads it from the checkpoint's
    # backbone (modernbert -> ModernBERT tokenizer, else cl100k); a ModernBERT
    # checkpoint MUST be fed ModernBERT ids or it sees garbage.
    tokenizer: str = "auto"
    _tokenizer: str = "cl100k"
    _model: object | None = None
    _device: object | None = None
    _err: str = ""
    # Per-text forward cache: id(text)->(ids, spans, force_keep, probs). The
    # iso-ratio gate bisects the drop rate by calling compress(same_text, varying
    # rate) ~14x; the windowed forward is identical across those calls (only the
    # decode threshold changes), so caching it turns 14 forwards into 1. Keyed by
    # the text string itself (small benchmark corpus; exact-match reuse).
    _score_cache: dict | None = None

    def __post_init__(self) -> None:
        suffix = ""
        if self.span:
            suffix += "+span"
        if self.floor:
            suffix += "+floor"
        if suffix and self.name == "lamr":
            self.name = "lamr" + suffix

    def available(self) -> tuple[bool, str]:
        if self._model is not None:
            return True, ""
        if not Path(self.ckpt).is_file():
            return False, f"checkpoint not found: {self.ckpt}"
        try:
            import torch

            from ..export.to_onnx import _load_checkpoint
            from ..train.loop import _pick_device

            model, cfg = _load_checkpoint(Path(self.ckpt))
            device = _pick_device()
            model.to(device)
            model.eval()
            self._model = model
            self._device = device
            self._torch = torch
            # Resolve the token space: "auto" follows the checkpoint's backbone.
            if self.tokenizer == "auto":
                self._tokenizer = "modernbert" if getattr(cfg, "backbone", "") == "modernbert" else "cl100k"
            else:
                self._tokenizer = self.tokenizer
            return True, ""
        except Exception as e:
            self._err = str(e)
            return False, f"load failed ({type(e).__name__}: {e})"

    def _score(self, text: str):
        """Encode ``text`` and run windowed inference to per-token drop-probs.

        Returns (ids, spans, force_keep, probs). Cached by text so the iso-ratio
        bisection (many compresses of the same text at different rates) pays the
        forward once. Windowed because real log chunks exceed the training window:
        truncating to one window would silently drop every token past it (and any
        needle in the tail — the costliest error for high recall), so the whole doc
        is scored in consecutive max_seq_len windows.
        """
        if self._score_cache is None:
            self._score_cache = {}
        cached = self._score_cache.get(text)
        if cached is not None:
            return cached
        torch = self._torch
        tok = self._tokenizer
        force_keep = None
        if self.floor:
            from .structural import structural_keep_mask

            ids, spans, force_keep = structural_keep_mask(text, tok)
        else:
            from ..label.align import encode_with_spans

            ids, spans = encode_with_spans(text, tok)
        if not ids:
            result = (ids, spans, force_keep, None)
            self._score_cache[text] = result
            return result
        import numpy as np

        W = self.max_seq_len
        chunks = []
        with torch.no_grad():
            for s in range(0, len(ids), W):
                window = ids[s : s + W]
                t_ids = torch.tensor([window], dtype=torch.long, device=self._device)
                mask = torch.ones((1, len(window)), dtype=torch.bool, device=self._device)
                chunks.append(torch.sigmoid(self._model(t_ids, mask).float())[0].cpu().numpy())
        probs = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        result = (ids, spans, force_keep, probs)
        self._score_cache[text] = result
        return result

    def compress(self, text: str, target_drop_rate: float) -> str:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(f"LaMR unavailable: {reason}")
        tok = self._tokenizer
        ids, spans, force_keep, probs = self._score(text)
        if not ids:
            return text
        import numpy as np

        if self.span:
            from .spandecode import span_decode

            return span_decode(
                ids, spans, text, probs, target_drop_rate,
                span=self.span, aggregator=self.aggregator, force_keep=force_keep,
                tokenizer=tok,
            )
        # Calibrated target-rate decode: drop the top-k highest-prob tokens, but
        # never a force-kept structural atom.
        drop_order = np.argsort(-probs).tolist()
        return _decode_with_drop_order(ids, drop_order, target_drop_rate, force_keep, tok)


def default_methods(lamr_ckpt: Path | None = None, include_llmlingua: bool = True) -> list[CompressionMethod]:
    """The standard comparison set. LaMR is included only if a checkpoint is given;
    LLMLingua-2 only if requested (it's heavy)."""
    methods: list[CompressionMethod] = [
        DeterministicDedup(),
        KeepSeverityHeuristic(),
        RandomDropFloor(),
        RandomDropFloor(floor=True),  # structural-floor demonstration
    ]
    if include_llmlingua:
        methods.append(LLMLingua2Method())
    if lamr_ckpt is not None:
        methods.append(LaMRMethod(ckpt=Path(lamr_ckpt)))
        methods.append(LaMRMethod(ckpt=Path(lamr_ckpt), floor=True))
    return methods
