"""LLMLingua-2 quality-control metrics: Variation Rate and Alignment Gap.

These operate at the *word* level (whitespace + punctuation tokenization), not
BPE — see research/Advanced_Data_Distillation_for_Token_Deletion.md §Quality
Control Frameworks. VR/AG measure whether the teacher LLM stayed extractive;
they do not depend on the downstream tokenizer.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    """Lowercased word + punctuation tokenization for VR/AG bookkeeping."""
    return [tok.lower() for tok in _WORD_RE.findall(text)]


def variation_rate(original: str, compressed: str) -> float:
    """VR = fraction of compressed tokens absent from the original.

    A high VR indicates the teacher hallucinated / paraphrased instead of
    extracting. Range [0, 1].
    """
    orig = set(tokenize_words(original))
    comp = tokenize_words(compressed)
    if not comp:
        return 0.0
    novel = sum(1 for w in comp if w not in orig)
    return novel / len(comp)


def matching_rate(original: str, compressed: str) -> float:
    """MR = fraction of original tokens that were successfully mapped to
    compressed (multiset-style — repeated words only match up to their count
    in the compressed text)."""
    orig = tokenize_words(original)
    if not orig:
        return 0.0
    comp_counts = Counter(tokenize_words(compressed))
    matched = 0
    for w in orig:
        if comp_counts.get(w, 0) > 0:
            comp_counts[w] -= 1
            matched += 1
    return matched / len(orig)


def hitting_rate(original: str, compressed: str) -> float:
    """HR = (count of compressed words that exist in original) / |original|.

    Per LLMLingua-2 §Quality Control. Note the denominator is |original|, not
    |compressed| — this is intentional so HR ≥ MR for fully extractive
    compressions, making AG = HR − MR ≥ 0 the right "alignment defect" signal.
    """
    orig_set = set(tokenize_words(original))
    if not orig_set:
        return 0.0
    comp = tokenize_words(compressed)
    hits = sum(1 for w in comp if w in orig_set)
    orig_len = len(tokenize_words(original))
    if orig_len == 0:
        return 0.0
    return hits / orig_len


def alignment_gap(original: str, compressed: str) -> float:
    """AG = HR − MR. Zero for perfectly extractive + alignable compressions."""
    return hitting_rate(original, compressed) - matching_rate(original, compressed)


@dataclass(frozen=True)
class QCRecord:
    original: str
    compressed: str
    vr: float
    ag: float
    mr: float
    hr: float

    @classmethod
    def compute(cls, original: str, compressed: str) -> "QCRecord":
        mr = matching_rate(original, compressed)
        hr = hitting_rate(original, compressed)
        return cls(
            original=original,
            compressed=compressed,
            vr=variation_rate(original, compressed),
            ag=hr - mr,
            mr=mr,
            hr=hr,
        )
