"""Build answer-survival triples from LogHub-2.0 prose log messages.

The other benchmark builders (``build_heldout``, the curated fixtures) mine
needles that the deterministic structural floor either locks outright (IPs,
exception types, HTTP codes, UUIDs) or that sit in named ``key="value"`` fields
the floor also locks. On STRUCTURED logs that floor already wins, so a neural
model is not justified there.

This module targets the open question: GENUINELY UNSTRUCTURED PROSE logs, where
the fact that matters is stated in free narrative text with no salient key/regex
anchor (e.g. "Send worker leaving thread", "instruction cache parity error
corrected", "Lost executor 1 on host: remote Rpc client disassociated"). We mine
such needles from LogHub-2.0 (the canonical academic log corpus) so we can
measure whether keep-severity + the structural floor can preserve them. LOW floor
survival here is the evidence that these are the floor's genuine blind spot — the
justification for a neural salience model.

Pipeline per system:
  1. Strip the system-specific STRUCTURED header (timestamp / pid / level /
     component / block-id), isolating the free-text message tail.
  2. Reject any message that the floor would lock anyway (contains an IPv4,
     exception type, severity keyword, HTTP 4xx/5xx, UUID, errno, incident id,
     request_id, or a ``key="value"`` span) — we want ONLY the blind spot.
  3. Keep distinctive MULTI-WORD prose messages (>= 3 words, has lowercase
     letters, not pure ids/numbers); these are the needles.
  4. Embed each needle in a window of surrounding raw log lines so survival is a
     real "did the compressor keep this line's prose" test, and frame a question.

Reuses ``AnswerTriple`` (imported, never redefined). Output is written in the
exact format ``build_heldout`` saves/loads so it round-trips through
``build_heldout.load_triples``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from .triples import AnswerTriple

# ---------------------------------------------------------------------------
# Floor-lockable patterns. A needle containing ANY of these is NOT a blind spot
# (the structural floor would force-keep its tokens), so we reject it. Kept in
# sync with bench/structural.py::_LOCK_PATTERNS.
# ---------------------------------------------------------------------------
_FLOOR_LOCKABLE: list[re.Pattern] = [
    re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b"),                 # exception type
    re.compile(r"\b(?:FATAL|CRITICAL|ERROR|EXCEPTION|TRACEBACK|WARN(?:ING)?)\b"),  # severity
    re.compile(r"(?:status|HTTP|code)[=:\s]+[45]\d{2}\b", re.IGNORECASE),     # http 4xx/5xx
    re.compile(r"\berrno[=:]\s*\d+\b"),                                       # errno
    re.compile(r"\berror code\s+(?:0x[0-9A-Fa-f]+|\d+)\b", re.IGNORECASE),    # error code
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),                               # IPv4
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),  # UUID
    re.compile(r"\bINC\d{4,}\b"),                                             # incident id
    re.compile(r"request_id[=:]\s*[A-Za-z0-9_-]+"),                           # request id
    re.compile(
        r'\b(?:root_cause|resolution|resolution_action|remediation|failure_reason'
        r'|reason|msg|message|short_description|summary)\s*[=:]\s*"[^"\n]{1,200}"',
        re.IGNORECASE,
    ),
]


def _floor_would_lock(text: str) -> bool:
    return any(p.search(text) for p in _FLOOR_LOCKABLE)


# ---------------------------------------------------------------------------
# Per-system header strippers. Each returns the free-text message tail (or None
# if the line has no usable prose message). These remove the STRUCTURED prefix so
# the needle is the genuine narrative, not a component path or id.
# ---------------------------------------------------------------------------

# OpenStack:  nova-api.log... <ts> <pid> <LEVEL> <component> [req-...] <ip> "..." ...
#             We grab the human message after the component, dropping the
#             "GET ... HTTP" request lines (those are floor-lockable / structured).
_OS_RE = re.compile(
    r"^\S+\s+\d{4}-\d{2}-\d{2}\s+[\d:.]+\s+\d+\s+[A-Z]+\s+"
    r"[\w.$]+\s+(?:\[[^\]]*\]\s+)?(.*)$"
)

# Spark / Zookeeper / Hadoop share "... <LEVEL> <component>: <message>".
_SPARK_RE = re.compile(r"^\d\d/\d\d/\d\d \d\d:\d\d:\d\d [A-Z]+ +[\w.$]+: (.*)$")
_ZK_RE = re.compile(r"^[\d\- :,]+ - [A-Z]+ +\[[^\]]*\] - (.*)$")
_HADOOP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} [\d:,]+ [A-Z]+ +\[[^\]]*\] [\w.$]+: (.*)$"
)

# HDFS: 081109 203615 148 INFO dfs.DataNode$PacketResponder: <message>
_HDFS_RE = re.compile(r"^\d{6} \d{6} \d+ [A-Z]+ [\w.$]+: (.*)$")

# Linux syslog: Jun 14 15:16:01 combo sshd(pam_unix)[19939]: <message>
_LINUX_RE = re.compile(r"^[A-Z][a-z]{2} +\d+ [\d:]+ \S+ \S+: (.*)$")

# BGL: - <epoch> <date> <loc> <ts> <loc2> RAS <facility> <LEVEL> <message>
_BGL_RE = re.compile(
    r"^\S+ \d+ [\d.]+ \S+ [\d.\-]+ \S+ RAS \w+ [A-Z]+ (.*)$"
)


def _strip_openstack(line: str) -> str | None:
    m = _OS_RE.match(line)
    if not m:
        return None
    msg = m.group(1).strip()
    # Drop the WSGI request lines (structured HTTP, floor territory).
    if msg.startswith('"') or msg.lower().startswith("get ") or "HTTP/1.1" in msg:
        return None
    return msg


def _strip_generic(rx: re.Pattern):
    def fn(line: str) -> str | None:
        m = rx.match(line)
        return m.group(1).strip() if m else None
    return fn


_SYSTEMS: dict[str, object] = {
    "openstack": _strip_openstack,
    "spark": _strip_generic(_SPARK_RE),
    "hadoop": _strip_generic(_HADOOP_RE),
    "zookeeper": _strip_generic(_ZK_RE),
    "hdfs": _strip_generic(_HDFS_RE),
    "linux": _strip_generic(_LINUX_RE),
    "bgl": _strip_generic(_BGL_RE),
}


# ---------------------------------------------------------------------------
# Needle qualification: a distinctive multi-word PROSE phrase.
# ---------------------------------------------------------------------------

# Strip trailing volatile tokens (ids/numbers/hex/blocks) so the needle is the
# stable prose, not a per-event id. We keep the leading prose words.
_VOLATILE_TAIL = re.compile(
    r"(?:\s+(?:\d+|0x[0-9a-fA-F]+|blk_-?\d+|[\w.$]+_\d+|\S*\d{3,}\S*))+\s*$"
)
_WORD = re.compile(r"[A-Za-z]")


def _needle_from_message(msg: str) -> str | None:
    """Reduce a raw message to a stable, distinctive multi-word prose needle."""
    # Drop a leading "Key: " label if the rest is prose (keeps "leaving thread"
    # style needles intact while removing volatile prefixes).
    needle = msg.strip().rstrip(".")
    # Trim a trailing run of volatile id/number tokens.
    trimmed = _VOLATILE_TAIL.sub("", needle).strip()
    if len(trimmed.split()) >= 3:
        needle = trimmed
    needle = needle.strip().strip(":").strip()
    words = needle.split()
    if len(words) < 3:
        return None
    if not _WORD.search(needle):
        return None
    # Require at least one lowercase letter (real prose, not an ALL-CAPS id).
    if not any(c.islower() for c in needle):
        return None
    # Reject if it still embeds a long digit run / id-like token.
    if re.search(r"\d{3,}", needle):
        return None
    if len(needle) < 12 or len(needle) > 120:
        return None
    return needle


# ---------------------------------------------------------------------------
# Question framing.
# ---------------------------------------------------------------------------
_QUESTION = "What operational event or condition did the log report?"


def build_triples_for_system(
    name: str,
    text: str,
    window_lines: int = 30,
    max_triples: int = 40,
) -> list[AnswerTriple]:
    """Mine prose needles for one LogHub system.

    The needle's line is embedded in a window of surrounding raw lines; the needle
    must occur exactly once in that window (unambiguous survival) and must be a
    floor blind spot (not lockable by any structural pattern).
    """
    strip = _SYSTEMS[name]
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    source = f"loghub2:{name}"
    triples: list[AnswerTriple] = []
    seen: set[str] = set()
    n = len(raw_lines)
    half = window_lines // 2

    for i, line in enumerate(raw_lines):
        if len(triples) >= max_triples:
            break
        msg = strip(line)  # type: ignore[operator]
        if not msg:
            continue
        if _floor_would_lock(msg):
            continue
        needle = _needle_from_message(msg)
        if needle is None:
            continue
        if _floor_would_lock(needle):  # paranoia: needle itself must be clean
            continue
        if needle in seen:  # distinct prose only — no repeated boilerplate
            continue
        # The needle must literally appear in the chosen line.
        if needle not in line:
            continue
        # Build a window of surrounding raw lines around the needle's line.
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = raw_lines[lo:hi]
        chunk = "\n".join(window)
        # Unambiguous survival: the needle phrase occurs exactly once in window.
        if chunk.count(needle) != 1:
            continue
        seen.add(needle)
        triples.append(
            AnswerTriple(
                doc_id=f"{source}#{i}",
                text=chunk,
                question=_QUESTION,
                answer=needle,
                fact_type=f"loghub:{name}",
                source=source,
            )
        )
    return triples


def build_all(
    raw_dir: Path,
    window_lines: int = 30,
    max_per_system: int = 40,
) -> list[AnswerTriple]:
    triples: list[AnswerTriple] = []
    for name in _SYSTEMS:
        path = raw_dir / f"{name}.log"
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        triples.extend(
            build_triples_for_system(
                name, text, window_lines=window_lines, max_triples=max_per_system
            )
        )
    return triples


def _fact_counts(triples: list[AnswerTriple]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in triples:
        out[t.fact_type] = out.get(t.fact_type, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lamr-build-loghub",
        description="Mine prose-needle answer triples from LogHub-2.0 logs.",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("../data/raw/loghub2"),
        help="dir of <system>.log files",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("../data/bench/loghub_triples.json"),
        help="output triples JSON",
    )
    p.add_argument("--window-lines", type=int, default=30)
    p.add_argument("--max-per-system", type=int, default=40)
    args = p.parse_args(argv)

    triples = build_all(
        args.raw_dir,
        window_lines=args.window_lines,
        max_per_system=args.max_per_system,
    )
    payload = {
        "n": len(triples),
        "class_counts": {"prose": len(triples)},
        "fact_type_counts": _fact_counts(triples),
        "triples": [asdict(t) for t in triples],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"built {len(triples)} loghub prose triples")
    print(f"fact types: {payload['fact_type_counts']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
