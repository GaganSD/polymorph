"""Answer-survival benchmark: triple mining, compression methods, and the sweep."""

from pathlib import Path

from polymorph_lamr.bench.methods import (
    DeterministicDedup,
    KeepSeverityHeuristic,
    LaMRMethod,
    LLMLingua2Method,
    RandomDropFloor,
    default_methods,
    token_count,
)
from polymorph_lamr.bench.survival import (
    achieved_drop_rate,
    answer_survives,
    compression_ratio,
    evaluate_method,
    format_report,
    run_benchmark,
)
from polymorph_lamr.bench.triples import (
    build_triples_from_text,
    curated_triples,
)

RATES = [0.2, 0.5, 0.8]


# ---- triple mining ----

def test_curated_triples_have_answers_in_text():
    triples = curated_triples()
    assert len(triples) == 4
    for t in triples:
        assert t.answer and t.answer in t.text
        assert t.question.endswith("?")
        assert t.fact_type


def test_build_triples_extracts_a_needle():
    text = "\n".join(
        ["2023-01-01 INFO heartbeat ok"] * 10
        + ["2023-01-01 ERROR boom request_id=ABC123 client_ip=10.0.0.9"]
    )
    triples = build_triples_from_text(text, source="t", window_lines=50)
    assert triples
    # The unique needle (request_id / ip / severity) must be extractable.
    t = triples[0]
    assert t.answer in t.text


# ---- methods ----

def test_deterministic_dedup_collapses_runs_and_is_lossless_for_uniques():
    text = "\n".join(["2023-01-01T00:00:0%d.0 INFO heartbeat ok" % (i % 10) for i in range(20)]
                     + ["FATAL unique crash code 0xDEAD"])
    out = DeterministicDedup().compress(text, 0.0)
    # The repeated heartbeat run collapses (fewer tokens), the unique FATAL stays.
    assert token_count(out) < token_count(text)
    assert "0xDEAD" in out
    assert "elided" in out


def test_keep_severity_keeps_the_severe_line_and_shrinks():
    text = "\n".join(["GET /x 200 ok id=%d" % i for i in range(40)]
                     + ["ERROR 500 failure token=NEEDLE42"])
    out = KeepSeverityHeuristic().compress(text, 0.7)
    assert token_count(out) < token_count(text)
    assert "NEEDLE42" in out  # the severe line is prioritized


def test_random_drop_is_deterministic_and_hits_rate():
    text = " ".join(f"tok{i}" for i in range(400))
    a = RandomDropFloor().compress(text, 0.5)
    b = RandomDropFloor().compress(text, 0.5)
    assert a == b  # deterministic
    drop = achieved_drop_rate(text, a)
    assert 0.4 < drop < 0.6  # ~50% removed


def test_random_drop_rate_zero_keeps_everything():
    text = " ".join(f"tok{i}" for i in range(100))
    assert achieved_drop_rate(text, RandomDropFloor().compress(text, 0.0)) < 0.05


def test_optional_methods_report_unavailable_gracefully():
    # LaMR with a missing checkpoint -> (False, reason), never raises.
    ok, reason = LaMRMethod(ckpt=Path("/nonexistent/ckpt.pt")).available()
    assert ok is False and "not found" in reason
    # LLMLingua-2 availability is a clean (bool, str) regardless of install state.
    ok2, reason2 = LLMLingua2Method().available()
    assert isinstance(ok2, bool) and isinstance(reason2, str)


# ---- survival metric ----

def test_answer_survives_normalizes_whitespace_and_case():
    assert answer_survives("INC0000045", "blah   inc0000045\n more")
    assert not answer_survives("INC0000045", "no needle here")


def test_compression_ratio_and_drop():
    text = "a b c d e f g h"
    half = "a b c d"
    assert compression_ratio(text, half) > 1.0
    assert achieved_drop_rate(text, half) > 0.0


# ---- the sweep ----

def test_run_benchmark_produces_valid_rows_for_available_methods():
    triples = curated_triples()
    methods = default_methods(lamr_ckpt=None, include_llmlingua=False)
    results, skipped = run_benchmark(triples, methods, RATES)
    # The three always-available methods produced results.
    assert {"deterministic", "keep-severity", "random"} <= set(results)
    for rows in results.values():
        for r in rows:
            assert 0.0 <= r.survival <= 1.0
            assert r.mean_ratio >= 1.0 - 1e-9
            assert r.n == len(triples)


def test_deterministic_preserves_all_unique_needles():
    # Dedup never drops a unique line, so every curated needle survives it.
    triples = curated_triples()
    rows = evaluate_method(DeterministicDedup(), triples, RATES)
    assert rows[0].survival == 1.0


def test_random_is_a_floor_keep_severity_beats_it_at_high_drop():
    triples = curated_triples()
    sev = evaluate_method(KeepSeverityHeuristic(), triples, [0.8])[0]
    rnd = evaluate_method(RandomDropFloor(), triples, [0.8])[0]
    assert sev.survival >= rnd.survival


def test_format_report_is_nonempty_string():
    triples = curated_triples()
    methods = default_methods(include_llmlingua=False)
    results, skipped = run_benchmark(triples, methods, RATES)
    report = format_report(results, skipped, triples, RATES)
    assert isinstance(report, str) and "survival" in report


# ---- Phase 0c: semantic needles + McNemar ----

from polymorph_lamr.bench.triples import _best_semantic_triple_for_chunk
from polymorph_lamr.bench.survival import mcnemar, survival_vector


def test_semantic_extractor_pulls_freetext_phrase():
    text = (
        'ERROR payment-api status=503 client_ip=10.0.0.9 '
        'msg="Internal server error" root_cause="Memory exhaustion here" resolution="Restart"'
    )
    t = _best_semantic_triple_for_chunk("d#0", text, "s")
    assert t is not None
    assert t.fact_type.startswith("semantic:")
    assert " " in t.answer  # multi-word phrase, not an atom


def test_floor_locks_key_anchored_value_but_not_unanchored_prose():
    # Key-anchored values ARE floor-lockable (the key locates them); bare prose
    # with no salient key is the floor's genuine blind spot (needs a model).
    from polymorph_lamr.bench.methods import RandomDropFloor
    floored = RandomDropFloor(floor=True)
    keyed = "\n".join(f"INFO tick {i}" for i in range(40))
    keyed += '\nINFO note root_cause="cascading queue backpressure" done'
    assert answer_survives("cascading queue backpressure", floored.compress(keyed, 0.8))
    bare = "\n".join(f"INFO tick {i}" for i in range(40))
    bare += "\nthe replication follower silently fell three minutes behind schedule"
    assert not answer_survives("silently fell three minutes behind", floored.compress(bare, 0.8))


def test_mcnemar_paired_significance():
    # A strictly dominates B on 10 of 100 items, never worse -> significant.
    a = [True] * 100
    b = [True] * 90 + [False] * 10
    r = mcnemar(a, b)
    assert r["b10_a_better"] == 10
    assert r["b01_a_worse"] == 0
    assert r["p_value"] < 0.05
    # Identical vectors -> no discordance -> p = 1.0
    assert mcnemar(a, a)["p_value"] == 1.0


def test_survival_vector_aligned_length():
    from polymorph_lamr.bench.methods import KeepSeverityHeuristic
    ts = curated_triples()
    v = survival_vector(KeepSeverityHeuristic(), ts, 0.5)
    assert len(v) == len(ts)
    assert all(isinstance(x, bool) for x in v)
