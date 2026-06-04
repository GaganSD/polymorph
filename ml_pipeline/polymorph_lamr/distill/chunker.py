"""Sentence-bounded chunker.

Per LLMLingua-2 §Dataset Distillation: split on sentence boundaries, cap each
chunk at `max_tokens` (cl100k). This prevents the teacher from "aggressive
over-compression in long-context scenarios" (GPT-4 distillation failure mode).

For source code, we split on blank lines (block boundaries) since punctuation
boundaries are misleading.
"""

from __future__ import annotations

import re
from functools import lru_cache

import tiktoken

_SENT_END_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z\(\[])")
_BLANK_LINE_RE = re.compile(r"\n\s*\n")


@lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc().encode_ordinary(text))


def _split_units(text: str, mode: str) -> list[str]:
    if mode == "code":
        parts = _BLANK_LINE_RE.split(text)
    elif mode == "log":
        # Line-oriented: each log record / trace line is a unit.
        parts = text.split("\n")
    else:
        parts = _SENT_END_RE.split(text)
    return [p for p in parts if p.strip()]


def _join(units: list[str], mode: str) -> str:
    if mode == "code":
        return "\n\n".join(units)
    if mode == "log":
        return "\n".join(units)
    return " ".join(units)


def chunk(text: str, max_tokens: int = 512, mode: str = "prose") -> list[str]:
    """Greedy pack of units into <= max_tokens chunks. Falls back to hard
    splits on overlong single units (rare but possible)."""
    if not text.strip():
        return []
    units = _split_units(text, mode)
    chunks: list[str] = []
    cur: list[str] = []
    cur_toks = 0

    for u in units:
        u_toks = _token_count(u)
        if u_toks > max_tokens:
            # Flush, then hard-split this oversized unit.
            if cur:
                chunks.append(_join(cur, mode))
                cur, cur_toks = [], 0
            chunks.extend(_hard_split(u, max_tokens))
            continue

        if cur_toks + u_toks > max_tokens and cur:
            chunks.append(_join(cur, mode))
            cur, cur_toks = [], 0
        cur.append(u)
        cur_toks += u_toks

    if cur:
        chunks.append(_join(cur, mode))
    return chunks


def _hard_split(text: str, max_tokens: int) -> list[str]:
    enc = _enc()
    ids = enc.encode_ordinary(text)
    pieces: list[str] = []
    for i in range(0, len(ids), max_tokens):
        pieces.append(enc.decode(ids[i : i + max_tokens]))
    return pieces


def detect_mode(path: str | None, text: str) -> str:
    if path:
        if path.endswith((".log", ".jsonl")):
            return "log"
        if path.endswith((".py", ".rs", ".ts", ".tsx", ".js", ".go", ".java", ".c", ".cpp", ".h")):
            return "code"
        if path.endswith(".json"):
            return "code"
        if path.endswith(".txt"):
            # Heuristic: many short newline-delimited records => treat as logs.
            lines = text.splitlines()
            if len(lines) >= 8 and (sum(len(l) for l in lines) / max(len(lines), 1)) < 240:
                return "log"
    return "prose"
