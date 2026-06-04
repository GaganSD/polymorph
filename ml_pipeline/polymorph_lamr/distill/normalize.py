"""Training-time normalization + trash detection for the distillation sampler.

Two jobs, kept deliberately separate:

1. **Template keying** (`normalize_line` / `template_key`) — collapse lines that
   are structurally identical modulo their variable tokens, so a corpus of 45k
   near-duplicate records (CI/CD runs, API error rows) dedups to its handful of
   real *templates* before we ever spend a teacher call on it.

   `normalize_line` is a faithful Python mirror of `src/dedup.rs::normalize_line`
   (same ordered patterns: TS / UUID / IP / HEX / NUM). It MUST stay in sync so
   the training distribution matches what the Rust runtime dedup produces. The
   runtime collapses redundancy *before* the neural pruner sees text, so training
   on deduped representatives is the correct distribution — not a shortcut.

   `template_key` layers one extra masker on top, `<RAND>`, for the high-entropy
   random blobs that survive the Rust patterns (mixed-case base62 strings are not
   pure hex/UUID, so steps 3/5/6 never match them). This is a *training-only*
   extension — it is not part of the reversible runtime dedup.

2. **Trash gating** (`signal_ratio` / `is_low_signal`) — drop lines/chunks that
   carry no real linguistic signal, so synthetic random-blob payloads (the CI/CD
   `error_message` field is literally `ERROR: RjyJqtFYmKiXBA5qwUE5HeQgJ2A...`)
   don't dominate the training set. A line keeps its *structural* fields; only a
   line that is essentially nothing but a random blob is dropped.

The random-token detector is calibrated (see tests/test_normalize.py) so that
real values like ``Database connection failure``, ``inventory-api``,
``Security Scan Failure`` and even long camelCase identifiers survive, while the
synthetic base62 blobs in the cicd / api_failures corpora are caught.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

# --- Rust-mirror normalization patterns (order matters; see src/dedup.rs) -----
# Each (regex, replacement) masks a class of variable token so two lines that
# differ only in their variable parts share a normalized key.
_NORMALIZE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ISO-8601 / RFC3339 timestamps
    (
        re.compile(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
        ),
        "<TS>",
    ),
    # Apache/CLF timestamps: 27/Dec/2037:12:00:00 +0530
    (
        re.compile(r"\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s*[+-]\d{4}"),
        "<TS>",
    ),
    # UUIDs
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "<UUID>",
    ),
    # IPv4
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
    # 0x-prefixed hex
    (re.compile(r"\b0[xX][0-9a-fA-F]+\b"), "<HEX>"),
    # long bare hex (>=16 chars: object ids, hashes, span ids)
    (re.compile(r"\b[0-9a-fA-F]{16,}\b"), "<HEX>"),
    # bare numbers (ints/floats) — last so it doesn't chew the classes above
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<NUM>"),
]

# Placeholders produced by normalization; the random detector must never flag
# them (they are short and uppercase, so the length gate already excludes them,
# but we list them for `signal_ratio` classification).
_PLACEHOLDERS = frozenset({"TS", "UUID", "IP", "HEX", "NUM", "RAND"})

# 'y' counts as a vowel for these heuristics: it routinely acts as one in real
# words/identifiers (synchronized, Unauthorized), and including it stops those
# from tripping the low-vowel / consonant-run randomness signals.
_VOWELS = frozenset("aeiouyAEIOUY")
# Alphanumeric runs are the unit the random detector scores.
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+")

# Training-only: any leftover digit run, even when glued to an identifier by an
# underscore (`ERR_621`, `pipe_2032`, `run_0`, `user_689`). The Rust NUM pattern
# is `\b`-anchored and cannot mask these, so without this every CI/CD row stays
# unique and dedup collapses nothing. Safe here because `template_key` is only a
# grouping key — the verbatim representative line keeps its real numbers.
_ANY_DIGIT_RUN = re.compile(r"\d+")

# Training-only: collapse a run of 2+ whitespace-separated placeholders into one.
# The *count* of noise tokens is itself noise — a synthetic `error_message` blob
# that splits into a variable number of space-separated random segments
# (`<RAND>`, `<RAND> <RAND>`, `<RAND> <RAND> <RAND>`) would otherwise keep
# otherwise-identical templates distinct and defeat dedup. The placeholders are
# already information-free at this point (normalize masked the real values), so
# collapsing their count loses nothing the template key still carries.
_PLACEHOLDER_RUN = re.compile(
    r"<(?:TS|UUID|IP|HEX|NUM|RAND)>(?:\s+<(?:TS|UUID|IP|HEX|NUM|RAND)>)+"
)


def normalize_line(line: str) -> str:
    """Mask variable tokens into a normalized template key.

    Faithful mirror of ``src/dedup.rs::normalize_line``: fixed pattern set applied
    in fixed order. Deterministic by construction.
    """
    out = line
    for pat, repl in _NORMALIZE_PATTERNS:
        out = pat.sub(repl, out)
    return out


def _entropy(s: str) -> float:
    """Shannon entropy (bits/char) over the character distribution of ``s``."""
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _case_transition_rate(s: str) -> float:
    """Fraction of adjacent letter pairs that flip case (RjYq... churn)."""
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 2:
        return 0.0
    flips = sum(
        1
        for a, b in zip(letters, letters[1:])
        if a.islower() != b.islower()
    )
    return flips / (len(letters) - 1)


def _max_consonant_run(s: str) -> int:
    """Longest run of consecutive consonants (random blobs cluster consonants)."""
    best = run = 0
    for c in s:
        if c.isalpha() and c not in _VOWELS:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def is_random_token(tok: str) -> bool:
    """True if ``tok`` looks like a high-entropy random blob (no linguistic signal).

    Length gate first: tokens under 12 chars are never confidently "random" —
    real short identifiers (endpoints, host ids) live inside otherwise-structured
    lines and are caught by the line-level signal gate, not here.

    Above that, we take a *vote* over independent randomness signals rather than
    any single hard gate (a real word can trip one signal by chance; a true
    random blob trips several):

      * low vowel ratio (< 0.30) — uniform base62 averages ~16% vowels
      * heavy case churn (>= 0.30) — RjYqWb... flips case constantly; no real
        identifier does, not even camelCase
      * long consonant run (>= 5) — clustered consonants are unpronounceable
      * high absolute entropy (>= 4.0 bits/char) — long real identifiers rarely
        exceed ~3.9
      * digit/letter mix — embedded digits among letters

    Two or more signals ⇒ random. This flags the synthetic base62 payloads while
    leaving real long identifiers (getUserAccountBalance, synchronized) untouched.
    """
    if len(tok) < 12:
        return False
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return False  # all-digit runs are handled by the NUM mask upstream
    vowel_ratio = sum(c in _VOWELS for c in letters) / len(letters)
    signals = (
        vowel_ratio < 0.30,
        _case_transition_rate(tok) >= 0.30,
        _max_consonant_run(tok) >= 5,
        _entropy(tok) >= 4.0,
        any(c.isdigit() for c in tok) and len(letters) >= 4,
    )
    return sum(signals) >= 2


def mask_random(text: str) -> str:
    """Replace random-blob alphanumeric runs with ``<RAND>``.

    Intended to run *after* `normalize_line` so the structured-variable patterns
    have already fired; the leftover high-entropy blobs are what this catches.
    """
    return _ALNUM_RUN.sub(
        lambda m: "<RAND>" if is_random_token(m.group(0)) else m.group(0),
        text,
    )


def template_key(line: str) -> str:
    """Sampling-time dedup key: normalize variables, then mask random blobs.

    Two lines that are structurally identical modulo timestamps/ids/numbers AND
    modulo their random payloads collapse to the same key. More aggressive than
    the reversible runtime dedup — that's intentional for *training-set
    selection* (we want one representative per structural template).

    Order matters: `mask_random` runs before the leftover-digit pass so the
    detector still sees a blob's digits (a digit/letter mix is a randomness
    signal); then any underscore-glued numeric id is folded to ``<NUM>``, and
    finally runs of adjacent placeholders collapse so a variable count of noise
    segments doesn't fragment the key.
    """
    key = _ANY_DIGIT_RUN.sub("<NUM>", mask_random(normalize_line(line)))
    return _PLACEHOLDER_RUN.sub("<RAND>", key)


def _is_wordlike(tok: str) -> bool:
    """Heuristic: does this alnum run read as a real word / identifier?

    Word-like = has letters, a plausible vowel ratio, and isn't a random blob.
    Placeholders (TS/NUM/...) count as structure, not noise.
    """
    if tok in _PLACEHOLDERS:
        return True
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return False  # pure-digit run: not linguistic signal
    if is_random_token(tok):
        return False
    vowel_ratio = sum(c in _VOWELS for c in letters) / len(letters)
    # Very long all-consonant tokens under the random length gate still read as
    # noise; everything else with at least one vowel cluster is signal.
    if len(tok) >= 12 and vowel_ratio < 0.20:
        return False
    return True


def signal_ratio(text: str) -> float:
    """Fraction of alphanumeric characters that belong to word-like tokens.

    1.0 = all real words/structure; ~0.0 = nothing but random blobs / raw ids.
    Returns 1.0 for text with no alphanumeric content (nothing to judge — left
    to the caller's min-length guard).
    """
    runs = _ALNUM_RUN.findall(text)
    total = sum(len(r) for r in runs)
    if total == 0:
        return 1.0
    signal = sum(len(r) for r in runs if _is_wordlike(r))
    return signal / total


def is_low_signal(text: str, *, min_ratio: float = 0.30, min_alnum: int = 24) -> bool:
    """True if ``text`` is trash: enough content to judge, but no real signal.

    `min_alnum` guards short lines (``200``, ``OK``, a bare status code) from
    being judged — they're cheap and don't dominate. Only substantial lines that
    are nonetheless signal-poor (a 130-char random ``error_message`` payload) are
    dropped.
    """
    runs = _ALNUM_RUN.findall(text)
    total = sum(len(r) for r in runs)
    if total < min_alnum:
        return False
    return signal_ratio(text) < min_ratio


@lru_cache(maxsize=4096)
def _cached_template_key(line: str) -> str:
    return template_key(line)


def template_key_cached(line: str) -> str:
    """`template_key` with a small LRU — the sampler calls it per line over
    millions of rows, and many share a template."""
    return _cached_template_key(line)
