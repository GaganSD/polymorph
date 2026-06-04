"""Prompt templates for the two distillation regimes.

Both prompts enforce EXTRACTIVE output. The teacher is instructed to return a
sequence that is a strict subsequence (word-level) of the original — any
paraphrase, tense shift, or reorder corrupts the downstream token-alignment.
The QC filter (`qc.metrics.variation_rate`) will catch and drop violators, but
we minimise their rate at source.

Claude 3.5 Sonnet (max_compression): aggressive deletion — keep only the
minimum tokens required for structural coherence.

GPT-4o (reasoning_preserved): conservative deletion — preserve multi-step
logical chains and named entities.
"""

from __future__ import annotations

_BASE_RULES = """\
Rules (ALL must hold):
1. Return a strict subsequence of the ORIGINAL text. Do NOT paraphrase, rename, \
re-tense, pluralize, or reorder any word.
2. Output ONLY the compressed text. No preamble, no commentary, no code fences, \
no quotes.
3. Preserve syntactic anchors: brackets, braces, parentheses, commas, colons, \
semicolons, and operators that appear in the original.
4. You may delete: filler phrases, transition words, redundant restatements, \
greeting/closing fluff, and rhetorical hedges.
5. NEVER add a word that does not appear in the original.
"""

CLAUDE_MAX_COMPRESSION = (
    "You are a precise extractive compression tool. Your goal is MAXIMUM "
    "deletion while keeping the text structurally coherent.\n\n"
    + _BASE_RULES
    + "\nAim to delete 40-60% of tokens. If the input is already minimal, "
    "delete less; never invent text to hit a target.\n\n"
    "ORIGINAL:\n{text}\n\nCOMPRESSED:"
)

GPT4O_REASONING_PRESERVED = (
    "You are a precise extractive compression tool. Your goal is to compress "
    "while PRESERVING all multi-step reasoning, named entities, numeric "
    "constants, and logical connectives.\n\n"
    + _BASE_RULES
    + "\nAim to delete 20-35% of tokens. Preserve every chain-of-thought node, "
    "every cited fact, every numeric value. Delete only conversational and "
    "rhetorical fluff.\n\n"
    "ORIGINAL:\n{text}\n\nCOMPRESSED:"
)


LOG_TRACE_EXTRACTIVE = (
    "You are a precise extractive compression tool for CLOUD-NATIVE LOGS and "
    "DISTRIBUTED TRACE payloads. Your goal is to delete redundant, low-information "
    "tokens while keeping the record forensically complete.\n\n"
    + _BASE_RULES
    + "\n6. NEVER delete: log levels (INFO/WARN/ERROR/FATAL), HTTP status codes, "
    "error/exception names, trace/span IDs, resource identifiers, numeric "
    "constants, hex offsets, or state-transition keywords.\n"
    "7. You MAY delete: repeated boilerplate, verbose stack-frame chatter, "
    "redundant timestamps, and filler prose around the signal.\n\n"
    "Aim to delete 30-50% of tokens. Preserve everything an on-call engineer "
    "would need to localize a fault.\n\n"
    "ORIGINAL:\n{text}\n\nCOMPRESSED:"
)


# NOTE: the default teacher ensemble + provider routing now live in
# `providers.py` (DEFAULT_TEACHER_SPECS / resolve_routing) — open-weight models
# are individually weaker at the strict extractive constraint, so we fan out
# across teachers and keep the per-chunk best-QC output.


def render(template: str, text: str) -> str:
    return template.format(text=text)
