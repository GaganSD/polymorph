"""Defensible-eval statistics over the saved ``judge_bench`` per-item JSON.

``judge_bench`` writes a ``per_item`` map keyed ``"<method>@<ratio-key>"`` whose
values are aligned lists of ``{doc_id, fact_type, answer, ratio, exact, judge,
judge_error}`` records (same doc_ids, same order, across every method at a given
ratio — verified by construction in ``_evaluate``). This module turns that raw
log into the three things a headline answer-survival claim actually needs to be
defensible, WITHOUT re-running the (paid) judge:

  * **Per-domain / per-fact_type survival breakdowns** — so a claim can't ride on
    one easy domain. For the LogHub set ``fact_type`` is ``loghub:<domain>`` and
    the domain is that suffix; for other corpora ``fact_type`` is used as-is.
  * **McNemar's paired test** of two methods on the SAME triples — the right test
    for "does A preserve more needles than B" when the two are scored on identical
    items. We report both the χ² statistic (with Yates' continuity correction) and
    the exact two-sided binomial p-value on the discordant pairs (the χ²
    approximation is unreliable when the discordant count is small, which it often
    is at n=50–200).
  * **Bootstrap 95% CIs** on each method's survival rate — a seeded, deterministic
    percentile bootstrap over the per-item survival booleans, so a point estimate
    ("68%") comes with an interval.

Everything here is pure / offline (no torch, no network) so it imports cheaply and
unit-tests fast. ``judge_bench --stats`` wires it onto a freshly-written results
file; ``analyze_file`` / ``main`` also run it standalone on any saved JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Iterable, Literal

Metric = Literal["judge", "exact"]


# ---------------------------------------------------------------------------
# Loading + domain derivation
# ---------------------------------------------------------------------------

def domain_of(fact_type: str) -> str:
    """Collapse a fact_type to a coarse domain. For LogHub triples the fact_type
    is ``loghub:<domain>`` (e.g. ``loghub:spark``) and the domain is the suffix;
    for semantic/structural needles (``semantic:msg``, ``http_status``) the
    fact_type already names the class, so it is returned unchanged."""
    if ":" in fact_type:
        head, tail = fact_type.split(":", 1)
        if head == "loghub":
            return tail
    return fact_type


def split_key(key: str) -> tuple[str, str]:
    """Split a ``per_item`` key ``"<method>@<ratio-key>"`` into (method, ratio_key).
    Method names never contain '@'; the ratio key is everything after the last '@'
    (e.g. ``lamr+span@iso3.0`` -> ('lamr+span', 'iso3.0'))."""
    method, _, ratio = key.rpartition("@")
    if not method:  # no '@' -> treat whole thing as method, empty ratio
        return key, ""
    return method, ratio


def load_per_item(path: str | Path) -> dict[str, list[dict]]:
    """Read the ``per_item`` map from a judge_bench results JSON."""
    payload = json.loads(Path(path).read_text())
    pi = payload.get("per_item")
    if not isinstance(pi, dict):
        raise ValueError(f"{path}: no 'per_item' map (not a judge_bench results file?)")
    return pi


def ratio_keys(per_item: dict[str, list[dict]]) -> list[str]:
    """Distinct ratio keys present, preserving first-seen order."""
    seen: list[str] = []
    for k in per_item:
        r = split_key(k)[1]
        if r not in seen:
            seen.append(r)
    return seen


def methods_at(per_item: dict[str, list[dict]], ratio_key: str) -> list[str]:
    """Method names available at a given ratio key, in first-seen order."""
    out: list[str] = []
    for k in per_item:
        m, r = split_key(k)
        if r == ratio_key and m not in out:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Survival vectors + rates
# ---------------------------------------------------------------------------

def survival_bits(items: list[dict], metric: Metric) -> list[bool]:
    """Per-item survival booleans for one metric ('judge' or 'exact')."""
    return [bool(it.get(metric)) for it in items]


def survival_rate(items: list[dict], metric: Metric) -> float:
    bits = survival_bits(items, metric)
    return sum(bits) / len(bits) if bits else 0.0


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapCI:
    point: float          # observed survival rate (mean of bits)
    lo: float             # lower percentile bound
    hi: float             # upper percentile bound
    n: int                # number of items
    resamples: int
    conf: float

    def as_dict(self) -> dict:
        return {
            "point": round(self.point, 4),
            "lo": round(self.lo, 4),
            "hi": round(self.hi, 4),
            "n": self.n,
            "resamples": self.resamples,
            "conf": self.conf,
        }


def bootstrap_ci(
    bits: list[bool],
    *,
    resamples: int = 1000,
    conf: float = 0.95,
    seed: int = 1234,
) -> BootstrapCI:
    """Percentile bootstrap CI on a survival rate (mean of booleans).

    Deterministic given (bits, resamples, conf, seed): we seed a private
    ``random.Random`` so the same inputs always yield the same interval (so a
    reported CI is reproducible and unit-testable). Each resample draws ``n`` items
    with replacement and records the resample mean; the CI is the empirical
    [alpha/2, 1-alpha/2] percentile of those means.
    """
    n = len(bits)
    point = (sum(bits) / n) if n else 0.0
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, 0, resamples, conf)
    rng = random.Random(seed)
    vals = [1.0 if b else 0.0 for b in bits]
    means: list[float] = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    alpha = 1.0 - conf
    lo_idx = max(0, min(int((alpha / 2.0) * resamples), resamples - 1))
    hi_idx = max(0, min(int((1.0 - alpha / 2.0) * resamples), resamples - 1))
    return BootstrapCI(point, means[lo_idx], means[hi_idx], n, resamples, conf)


# ---------------------------------------------------------------------------
# McNemar's paired test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class McNemarResult:
    b: int                # A right / B wrong  (A better)
    c: int                # A wrong / B right  (A worse)
    n_discordant: int
    chi2: float           # Yates-corrected chi-square statistic (1 dof)
    p_chi2: float         # p-value from the chi-square approximation
    p_exact: float        # two-sided exact binomial p on the discordant pairs
    a_better: bool        # b > c (A preserves more needles than B)

    def as_dict(self) -> dict:
        return {
            "b_a_better": self.b,
            "c_a_worse": self.c,
            "n_discordant": self.n_discordant,
            "chi2_yates": round(self.chi2, 4),
            "p_chi2": round(self.p_chi2, 6),
            "p_exact": round(self.p_exact, 6),
            "a_better": self.a_better,
        }


def _chi2_sf_1dof(x: float) -> float:
    """Survival function P(X > x) for a chi-square with 1 dof, = erfc(sqrt(x/2)).
    Uses math.erfc so we don't need scipy."""
    from math import erfc, sqrt

    if x <= 0.0:
        return 1.0
    return erfc(sqrt(x / 2.0))


def mcnemar_paired(a_bits: list[bool], b_bits: list[bool]) -> McNemarResult:
    """McNemar's test on aligned survival vectors of methods A and B.

    ``b`` = A survived / B did not (A better); ``c`` = A did not / B did (A worse).
    χ² uses Yates' continuity correction: (|b - c| - 1)^2 / (b + c). The exact
    two-sided binomial p (theta = 0.5 on the discordant pairs) is the robust
    fallback for small discordant counts. A small p with ``a_better`` True means A
    preserves significantly more needles than B on the same triples.
    """
    if len(a_bits) != len(b_bits):
        raise ValueError("survival vectors must be aligned (same triples/order)")
    b = sum(1 for x, y in zip(a_bits, b_bits) if x and not y)
    c = sum(1 for x, y in zip(a_bits, b_bits) if (not x) and y)
    n = b + c
    if n == 0:
        return McNemarResult(0, 0, 0, 0.0, 1.0, 1.0, False)
    chi2 = (abs(b - c) - 1.0) ** 2 / n if n > 0 else 0.0
    p_chi2 = _chi2_sf_1dof(chi2)
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    p_exact = min(1.0, 2.0 * tail)
    return McNemarResult(b, c, n, chi2, p_chi2, p_exact, b > c)


# ---------------------------------------------------------------------------
# Aggregate analysis over a per_item map
# ---------------------------------------------------------------------------

@dataclass
class MethodStats:
    method: str
    ratio_key: str
    n: int
    judge: BootstrapCI
    exact: BootstrapCI
    judge_errors: int
    per_domain: dict[str, dict]  # domain -> {n, judge_rate, exact_rate}

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "ratio_key": self.ratio_key,
            "n": self.n,
            "judge": self.judge.as_dict(),
            "exact": self.exact.as_dict(),
            "judge_errors": self.judge_errors,
            "per_domain": self.per_domain,
        }


def _per_domain(items: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for it in items:
        d = domain_of(str(it.get("fact_type", "?")))
        buckets.setdefault(d, []).append(it)
    out: dict[str, dict] = {}
    for d in sorted(buckets):
        b = buckets[d]
        out[d] = {
            "n": len(b),
            "judge_rate": round(survival_rate(b, "judge"), 4),
            "judge_survived": sum(1 for x in b if x.get("judge")),
            "exact_rate": round(survival_rate(b, "exact"), 4),
            "exact_survived": sum(1 for x in b if x.get("exact")),
        }
    return out


def method_stats(
    items: list[dict],
    method: str,
    ratio_key: str,
    *,
    resamples: int = 1000,
    conf: float = 0.95,
    seed: int = 1234,
) -> MethodStats:
    return MethodStats(
        method=method,
        ratio_key=ratio_key,
        n=len(items),
        judge=bootstrap_ci(survival_bits(items, "judge"), resamples=resamples, conf=conf, seed=seed),
        exact=bootstrap_ci(survival_bits(items, "exact"), resamples=resamples, conf=conf, seed=seed),
        judge_errors=sum(1 for x in items if x.get("judge_error")),
        per_domain=_per_domain(items),
    )


@dataclass
class PairTest:
    ratio_key: str
    method_a: str
    method_b: str
    metric: Metric
    result: McNemarResult

    def as_dict(self) -> dict:
        return {
            "ratio_key": self.ratio_key,
            "method_a": self.method_a,
            "method_b": self.method_b,
            "metric": self.metric,
            **self.result.as_dict(),
        }


# Default paired comparisons a survival claim rests on: the model vs the heuristic
# floor and vs the published external baseline. Only run when both are present.
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("lamr+span", "keep-severity"),
    ("lamr+span", "llmlingua2"),
    ("lamr+span+floor", "keep-severity"),
]


def analyze(
    per_item: dict[str, list[dict]],
    *,
    pairs: list[tuple[str, str]] | None = None,
    metrics: tuple[Metric, ...] = ("judge", "exact"),
    resamples: int = 1000,
    conf: float = 0.95,
    seed: int = 1234,
) -> dict:
    """Full defensible-eval analysis over a per_item map.

    Returns ``{ratio_key: {"methods": [MethodStats...], "pairs": [PairTest...]}}``.
    Paired tests require the two methods to be aligned on doc_id at that ratio
    (asserted); a requested pair missing a method is silently skipped.
    """
    pairs = pairs if pairs is not None else DEFAULT_PAIRS
    out: dict = {}
    for rk in ratio_keys(per_item):
        ms: list[MethodStats] = []
        method_items: dict[str, list[dict]] = {}
        for m in methods_at(per_item, rk):
            items = per_item[f"{m}@{rk}"]
            method_items[m] = items
            ms.append(method_stats(items, m, rk, resamples=resamples, conf=conf, seed=seed))
        pts: list[PairTest] = []
        for a, b in pairs:
            if a not in method_items or b not in method_items:
                continue
            ia, ib = method_items[a], method_items[b]
            ida = [x.get("doc_id") for x in ia]
            idb = [x.get("doc_id") for x in ib]
            if ida != idb:
                raise ValueError(
                    f"paired methods {a} vs {b} @ {rk} are not aligned on doc_id "
                    "(McNemar requires identical triples in the same order)"
                )
            for metric in metrics:
                res = mcnemar_paired(survival_bits(ia, metric), survival_bits(ib, metric))
                pts.append(PairTest(rk, a, b, metric, res))
        out[rk] = {
            "methods": [m.as_dict() for m in ms],
            "pairs": [p.as_dict() for p in pts],
        }
    return out


def analyze_file(path: str | Path, **kw) -> dict:
    return analyze(load_per_item(path), **kw)


# ---------------------------------------------------------------------------
# Pretty report
# ---------------------------------------------------------------------------

def _ci_cell(ci: dict) -> str:
    return f"{100*ci['point']:4.0f}% [{100*ci['lo']:.0f}-{100*ci['hi']:.0f}]"


def format_stats(analysis: dict, *, metric: Metric = "judge") -> str:
    lines: list[str] = []
    lines.append(f"== Defensible answer-survival stats (metric={metric}, 95% bootstrap CI) ==")
    for rk, blob in analysis.items():
        lines.append("")
        lines.append(f"-- ratio={rk} --")
        hdr = "  " + "method".ljust(20) + "n".ljust(5) + f"{metric} survival (95% CI)".ljust(22) + "err"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for m in blob["methods"]:
            ci = m[metric]
            lines.append(
                "  " + m["method"].ljust(20) + str(m["n"]).ljust(5)
                + _ci_cell(ci).ljust(22) + str(m["judge_errors"])
            )
        # per-domain (judge metric) for the lamr+span method if present, else first
        focus = next((m for m in blob["methods"] if m["method"] == "lamr+span"), None)
        if focus is None and blob["methods"]:
            focus = blob["methods"][0]
        if focus:
            lines.append(f"  per-domain ({focus['method']}, {metric}):")
            rate_key = f"{metric}_rate"
            surv_key = f"{metric}_survived"
            for d, dd in focus["per_domain"].items():
                lines.append(
                    f"    {d.ljust(14)} n={str(dd['n']).ljust(4)} "
                    f"{100*dd[rate_key]:3.0f}%  ({dd[surv_key]}/{dd['n']})"
                )
        # paired tests
        pts = [p for p in blob["pairs"] if p["metric"] == metric]
        if pts:
            lines.append("  McNemar (paired, same triples):")
            for p in pts:
                lines.append(
                    f"    {p['method_a']} vs {p['method_b']}: "
                    f"b(A>B)={p['b_a_better']} c(A<B)={p['c_a_worse']} "
                    f"chi2={p['chi2_yates']:.2f} p_chi2={p['p_chi2']:.4g} "
                    f"p_exact={p['p_exact']:.4g}"
                    + ("  [A significantly better]" if p["p_exact"] < 0.05 and p["a_better"] else "")
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lamr-bench-stats",
        description="Per-domain survival, McNemar paired test, and bootstrap CIs "
        "over a saved judge_bench per_item JSON.",
    )
    p.add_argument("results", type=Path, help="judge_bench results JSON (with per_item)")
    p.add_argument("--metric", choices=["judge", "exact"], default="judge")
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--conf", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=None, help="optional JSON dump of the full analysis")
    args = p.parse_args(argv)

    analysis = analyze_file(
        args.results, resamples=args.resamples, conf=args.conf, seed=args.seed
    )
    print(format_stats(analysis, metric=args.metric))
    if args.out:
        args.out.write_text(json.dumps(analysis, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
