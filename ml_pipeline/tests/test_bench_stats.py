"""Defensible-eval stats over the judge_bench per_item JSON.

Covers the three things a headline survival claim needs: McNemar on a known 2x2,
deterministic seeded bootstrap CIs, per-domain breakdown derivation, and the
end-to-end ``analyze`` over a synthetic paired per_item map.
"""

import math

import pytest

from polymorph_lamr.bench.stats import (
    analyze,
    bootstrap_ci,
    domain_of,
    mcnemar_paired,
    method_stats,
    split_key,
    survival_bits,
    survival_rate,
)


# ---- domain / key helpers ----

def test_domain_of_loghub_and_passthrough():
    assert domain_of("loghub:spark") == "spark"
    assert domain_of("loghub:bgl") == "bgl"
    # non-loghub colon types are passed through unchanged (already the class)
    assert domain_of("semantic:msg") == "semantic:msg"
    assert domain_of("http_status") == "http_status"


def test_split_key():
    assert split_key("lamr+span@iso3.0") == ("lamr+span", "iso3.0")
    assert split_key("keep-severity@0.5") == ("keep-severity", "0.5")
    assert split_key("noatsign") == ("noatsign", "")


# ---- McNemar on a known 2x2 ----

def test_mcnemar_known_2x2():
    # A strictly dominates B: right on 10 items B missed, never worse.
    a = [True] * 100
    b = [True] * 90 + [False] * 10
    r = mcnemar_paired(a, b)
    assert r.b == 10  # A>B
    assert r.c == 0   # A<B
    assert r.n_discordant == 10
    assert r.a_better is True
    # Yates chi2 = (|10-0|-1)^2 / 10 = 81/10 = 8.1
    assert r.chi2 == pytest.approx(8.1, abs=1e-9)
    # exact two-sided binomial on 10 discordant, k=0: 2 * (1/2^10) = 1/512
    assert r.p_exact == pytest.approx(2.0 / (2 ** 10), abs=1e-12)
    assert r.p_exact < 0.05
    # chi2 p from erfc(sqrt(8.1/2)) ~ 0.0044
    assert r.p_chi2 < 0.05


def test_mcnemar_no_discordance_is_p1():
    a = [True, False, True, False]
    r = mcnemar_paired(a, a)
    assert r.n_discordant == 0
    assert r.p_exact == 1.0 and r.p_chi2 == 1.0
    assert r.a_better is False


def test_mcnemar_symmetric_is_not_significant():
    # 5 each way -> perfectly balanced discordance, p should be ~1.
    a = [True] * 5 + [False] * 5
    b = [False] * 5 + [True] * 5
    r = mcnemar_paired(a, b)
    assert r.b == 5 and r.c == 5
    assert r.p_exact == pytest.approx(1.0)
    assert r.a_better is False


def test_mcnemar_rejects_misaligned():
    with pytest.raises(ValueError):
        mcnemar_paired([True, False], [True])


# ---- bootstrap determinism + sanity ----

def test_bootstrap_is_deterministic_for_fixed_seed():
    bits = [True] * 30 + [False] * 20  # 60% survival
    a = bootstrap_ci(bits, resamples=500, seed=7)
    b = bootstrap_ci(bits, resamples=500, seed=7)
    assert a == b  # frozen dataclass equality -> fully reproducible
    assert a.point == pytest.approx(0.6)
    # CI brackets the point estimate and lies in [0,1]
    assert 0.0 <= a.lo <= a.point <= a.hi <= 1.0


def test_bootstrap_different_seed_can_differ_but_brackets_point():
    bits = [True] * 30 + [False] * 20
    a = bootstrap_ci(bits, resamples=500, seed=1)
    b = bootstrap_ci(bits, resamples=500, seed=2)
    assert a.point == b.point == pytest.approx(0.6)
    # both intervals contain the point estimate
    assert a.lo <= 0.6 <= a.hi and b.lo <= 0.6 <= b.hi


def test_bootstrap_degenerate_all_true_is_tight():
    ci = bootstrap_ci([True] * 40, resamples=300, seed=3)
    assert ci.point == 1.0 and ci.lo == 1.0 and ci.hi == 1.0


def test_bootstrap_empty():
    ci = bootstrap_ci([], resamples=100, seed=3)
    assert ci.n == 0 and ci.point == 0.0


def test_bootstrap_ci_widens_with_smaller_n():
    # A smaller sample at the same rate should give a wider (or equal) interval.
    small = bootstrap_ci([True] * 6 + [False] * 4, resamples=1000, seed=11)
    large = bootstrap_ci([True] * 60 + [False] * 40, resamples=1000, seed=11)
    assert (small.hi - small.lo) >= (large.hi - large.lo)


# ---- survival rate helpers ----

def test_survival_rate_and_bits():
    items = [{"judge": True, "exact": False}, {"judge": False, "exact": True}]
    assert survival_bits(items, "judge") == [True, False]
    assert survival_rate(items, "judge") == pytest.approx(0.5)
    assert survival_rate(items, "exact") == pytest.approx(0.5)


# ---- per-domain via method_stats ----

def test_method_stats_per_domain_breakdown():
    items = [
        {"doc_id": "a", "fact_type": "loghub:spark", "judge": True, "exact": True},
        {"doc_id": "b", "fact_type": "loghub:spark", "judge": False, "exact": False},
        {"doc_id": "c", "fact_type": "loghub:bgl", "judge": True, "exact": False},
    ]
    ms = method_stats(items, "lamr+span", "iso3.0", resamples=200, seed=5)
    assert ms.n == 3
    pd = ms.per_domain
    assert set(pd) == {"spark", "bgl"}
    assert pd["spark"]["n"] == 2 and pd["spark"]["judge_survived"] == 1
    assert pd["spark"]["judge_rate"] == pytest.approx(0.5)
    assert pd["bgl"]["n"] == 1 and pd["bgl"]["judge_rate"] == pytest.approx(1.0)


# ---- end-to-end analyze over a paired per_item map ----

def _mk_items(facts, judges):
    return [
        {"doc_id": f"d{i}", "fact_type": ft, "judge": j, "exact": j, "judge_error": False}
        for i, (ft, j) in enumerate(zip(facts, judges))
    ]


def test_analyze_end_to_end_pairs_and_cis():
    facts = ["loghub:spark", "loghub:spark", "loghub:bgl", "loghub:bgl"]
    per_item = {
        # lamr survives 3/4, keep-severity 1/4, same doc order
        "lamr+span@iso3.0": _mk_items(facts, [True, True, True, False]),
        "keep-severity@iso3.0": _mk_items(facts, [False, True, False, False]),
    }
    res = analyze(per_item, resamples=200, seed=9)
    assert "iso3.0" in res
    blob = res["iso3.0"]
    names = {m["method"] for m in blob["methods"]}
    assert names == {"lamr+span", "keep-severity"}
    # the default lamr+span vs keep-severity pair was tested for both metrics
    pair = [p for p in blob["pairs"] if p["method_a"] == "lamr+span" and p["metric"] == "judge"]
    assert len(pair) == 1
    p = pair[0]
    # lamr beats keep-sev on d0 and d2 (discordant b=2), never worse (c=0)
    assert p["b_a_better"] == 2 and p["c_a_worse"] == 0
    assert p["a_better"] is True
    # CI present on each method
    for m in blob["methods"]:
        assert "lo" in m["judge"] and "hi" in m["judge"]


def test_analyze_raises_on_misaligned_pairs():
    per_item = {
        "lamr+span@iso3.0": [
            {"doc_id": "x", "fact_type": "loghub:spark", "judge": True, "exact": True},
        ],
        "keep-severity@iso3.0": [
            {"doc_id": "y", "fact_type": "loghub:spark", "judge": False, "exact": False},
        ],
    }
    with pytest.raises(ValueError):
        analyze(per_item)
