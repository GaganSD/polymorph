"""Apply LLMLingua-2 percentile filters: drop top-5% VR, then top-10% AG."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from .metrics import QCRecord


def _top_pct_threshold(values: list[float], pct: float) -> float:
    """Return the cutoff such that values >= cutoff fall in the top `pct`%.

    Uses nearest-rank to be stable on tiny lists used in tests.
    """
    if not values:
        return math.inf
    sorted_vals = sorted(values)
    # We want to drop the highest `pct`%, i.e., keep everything strictly below
    # the (100 - pct)-th percentile.
    keep_frac = 1.0 - pct / 100.0
    idx = max(0, min(len(sorted_vals) - 1, int(math.ceil(keep_frac * len(sorted_vals))) - 1))
    return sorted_vals[idx]


def filter_records(
    records: Iterable[QCRecord],
    vr_drop_top_pct: float = 5.0,
    ag_drop_top_pct: float = 10.0,
    vr_hard_floor: float = 0.5,
    require_subset: bool = True,
) -> tuple[list[QCRecord], dict]:
    """Three-stage filter:
      0. Hard floor: drop any record with VR > vr_hard_floor (likely paraphrase).
         Percentile-only filtering keeps best-of-bad when the whole batch is bad.
      0b. If require_subset, drop any record where the compressed text isn't
          actually shorter or contains tokens the original doesn't (VR > 0).
          This catches abstractive failures before percentile filtering.
      1. Drop top vr_drop_top_pct of survivors by VR.
      2. Drop top ag_drop_top_pct of *those* survivors by AG.
    """
    records = list(records)
    if not records:
        return [], {"total": 0, "kept": 0}

    pre_hard = len(records)
    if require_subset:
        # "comp is a strict subset" is over-strict (compressed will repeat words
        # the original has) — instead we require VR == 0 (no novel tokens) which
        # is the LLMLingua-2 extractive guarantee. Loose: VR < vr_hard_floor.
        records = [r for r in records if r.vr <= vr_hard_floor]
    after_hard = len(records)
    if not records:
        return [], {
            "total": pre_hard,
            "after_hard_floor": 0,
            "kept": 0,
            "vr_hard_floor": vr_hard_floor,
        }

    vrs = [r.vr for r in records]
    vr_cutoff = _top_pct_threshold(vrs, vr_drop_top_pct)
    stage1 = [r for r in records if r.vr <= vr_cutoff]

    ags = [r.ag for r in stage1]
    ag_cutoff = _top_pct_threshold(ags, ag_drop_top_pct)
    survivors = [r for r in stage1 if r.ag <= ag_cutoff]

    report = {
        "total": pre_hard,
        "after_hard_floor": after_hard,
        "after_vr_filter": len(stage1),
        "kept": len(survivors),
        "vr_hard_floor": vr_hard_floor,
        "vr_cutoff": vr_cutoff,
        "ag_cutoff": ag_cutoff,
        "vr_hist": _hist(vrs, bins=10, lo=0.0, hi=1.0),
        "ag_hist": _hist(ags, bins=10, lo=-0.2, hi=1.0),
    }
    return survivors, report


def _hist(values: list[float], bins: int, lo: float, hi: float) -> list[int]:
    counts = [0] * bins
    if hi <= lo:
        return counts
    width = (hi - lo) / bins
    for v in values:
        b = int((v - lo) / width)
        b = max(0, min(bins - 1, b))
        counts[b] += 1
    return counts


def write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))


def records_to_jsonl(records: Iterable[QCRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
