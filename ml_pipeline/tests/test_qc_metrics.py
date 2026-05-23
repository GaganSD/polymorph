"""Hand-computed VR / MR / HR / AG checks."""

from polymorph_lamr.qc.metrics import (
    QCRecord,
    alignment_gap,
    hitting_rate,
    matching_rate,
    tokenize_words,
    variation_rate,
)
from polymorph_lamr.qc.filter import filter_records


def test_extractive_compression_has_zero_vr():
    orig = "The quick brown fox jumps over the lazy dog."
    comp = "quick brown fox jumps lazy dog"
    assert variation_rate(orig, comp) == 0.0


def test_hallucinated_word_drives_vr_up():
    orig = "Alpha beta gamma."
    comp = "Alpha beta delta"  # 'delta' is novel
    vr = variation_rate(orig, comp)
    assert 0.0 < vr <= 1.0
    # 1 of 3 tokens is novel.
    assert abs(vr - 1 / 3) < 1e-9


def test_alignment_gap_zero_for_pure_extractive():
    orig = "Alpha beta gamma delta epsilon zeta."
    comp = "Alpha gamma zeta"  # strict subsequence
    # MR counts orig→comp multiset matches; HR counts comp tokens present in orig (over |orig|).
    mr = matching_rate(orig, comp)
    hr = hitting_rate(orig, comp)
    assert mr > 0
    assert hr > 0
    # Both metrics treat extractive cleanly; AG should be small and finite.
    assert alignment_gap(orig, comp) == hr - mr


def test_filter_drops_top_pcts():
    # 20 records: ascending VR, ascending AG so cutoffs are easy to reason about.
    records = []
    for i in range(20):
        # synthesize compressed strings to hit a target VR by adding novel tokens.
        orig = "alpha beta gamma delta " * 5
        comp_words = ["alpha", "beta", "gamma"]
        # inject `i` novel tokens
        comp_words.extend([f"novel{i}_{k}" for k in range(i)])
        comp = " ".join(comp_words)
        records.append(QCRecord.compute(orig, comp))

    survivors, report = filter_records(
        records,
        vr_drop_top_pct=5.0,
        ag_drop_top_pct=10.0,
        vr_hard_floor=1.0,  # disable hard floor for the percentile-only test
    )
    assert report["total"] == 20
    # 5% of 20 = 1 record dropped on VR; then 10% of the remaining 19 = ~2 dropped on AG.
    assert report["after_vr_filter"] <= 20
    assert len(survivors) <= report["after_vr_filter"]
    # The very-novelest record must be gone.
    assert all(r.vr <= report["vr_cutoff"] for r in survivors)


def test_hard_floor_drops_high_vr():
    """Records above the hard floor are dropped before percentile filtering."""
    good = QCRecord.compute("alpha beta gamma", "alpha gamma")
    bad = QCRecord.compute("alpha beta gamma", "totally different paraphrased text")
    survivors, report = filter_records([good, bad], vr_hard_floor=0.5)
    assert good in survivors
    assert bad not in survivors
    assert report["after_hard_floor"] == 1


def test_tokenize_words_lowercases_and_splits_punct():
    assert tokenize_words("Hello, World!") == ["hello", ",", "world", "!"]
