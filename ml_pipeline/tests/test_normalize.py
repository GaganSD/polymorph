"""Calibration tests for the trash detector + template normalizer.

The assertions below use REAL token samples pulled from the cicd_failures and
api_failures corpora (the synthetic random-blob payloads) and real structural
field values. If the detector's thresholds drift, these break — by design.
"""

from __future__ import annotations

import pytest

from polymorph_lamr.distill.normalize import (
    is_low_signal,
    is_random_token,
    mask_random,
    normalize_line,
    signal_ratio,
    template_key,
    template_key_cached,
)

# Real random blobs from data/raw/cicd_failures (error_message) + api_failures.
RANDOM_BLOBS = [
    "RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z",
    "2eykZAqkscUJ4KUxihlYXskJiDSG3TF0CXhFXZp9TfdtWDQWvfeJjXS1wiOikTyhDKEPIUsTbXRLcN9",
    "5r2039obpbhM8gNiE2GPONpWhYnkQBqFoM0IhXlPIqSkoJEnFqibMcbW8yEetJWD3tKbM4RvNixxpustB2d6p2HvFYhQ9fN",
    "hqMA49ZWCXsrHsXy81t5pCoPJBer4QpdnJdelM7WPCykUZENL6T469HHsAybKVxlk1JyuWWQLf3x3eRPFkiO8LKPIyMEfFtC4iKqVTzy9nAKMz0ziBAIFCXD",
    "xuqbiprmtjwu",  # api_failures container_id (12 chars, low-vowel)
]

# Real structural values that MUST survive (carry linguistic signal).
REAL_VALUES = [
    "Database",
    "connection",
    "failure",
    "Security",
    "Scan",
    "Failure",
    "Jenkins",
    "inventory",
    "Internal",
    "server",
    "error",
    "Unauthorized",
    "Optimize",
    "Resource",
    "Exhaustion",
    "windows",
    "latest",
    "Python",
    "getUserAccountBalance",  # long camelCase identifier — healthy vowel ratio
    "synchronized",  # 12 chars, y-as-vowel, consonant cluster — must NOT trip
    "Unauthorized",  # 12 chars
    "authentication",
    "getProfileByUserName",  # long camelCase, all distinct-ish chars
    "team",
    "alpha",
    "pipeline",
]


@pytest.mark.parametrize("blob", RANDOM_BLOBS)
def test_random_blobs_are_flagged(blob):
    assert is_random_token(blob), f"expected {blob!r} flagged as random"


@pytest.mark.parametrize("word", REAL_VALUES)
def test_real_values_are_not_flagged(word):
    assert not is_random_token(word), f"{word!r} wrongly flagged as random"


def test_short_tokens_never_flagged():
    # The length gate: short random-ish ids are not confidently random.
    for tok in ("lljugd", "mckssy", "ERR_621", "repo", "200", "OK", "ab"):
        assert not is_random_token(tok)


def test_normalize_line_mirrors_rust_patterns():
    # Apache access line: IP + CLF timestamp + numbers all masked.
    line = '233.223.117.90 - - [27/Dec/2037:12:00:00 +0530] "GET /x" 200 42'
    key = normalize_line(line)
    assert "<IP>" in key and "<TS>" in key and "<NUM>" in key
    # Two lines differing only in variable tokens share a key.
    other = '162.253.4.179 - - [27/Dec/2037:13:00:00 +0530] "GET /x" 200 99'
    assert normalize_line(line) == normalize_line(other)


def test_normalize_masks_iso_ts_uuid_hex():
    line = (
        "2025-12-29T07:58:16.927259 commit "
        "53820d0dddb2d97f40cbf0e1b4566169f480b86e id 0xDEADBEEF "
        "550e8400-e29b-41d4-a716-446655440000"
    )
    key = normalize_line(line)
    assert "<TS>" in key
    assert "<HEX>" in key  # both the 40-char hash and the 0x literal
    assert "<UUID>" in key
    assert "53820d0" not in key


def test_template_key_collapses_cicd_rows_modulo_blob():
    # Two cicd rows with identical structure but different random payloads +
    # numbers must collapse to ONE template key.
    a = (
        "2025-12-29T07:58:16 MEDIUM Jenkins stage=deploy "
        'error_code=ERR_621 msg="ERROR: RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z"'
    )
    b = (
        "2026-01-02T05:36:16 MEDIUM Jenkins stage=deploy "
        'error_code=ERR_621 msg="ERROR: g8tprJBHkhLu3r6EtkU5E0Y51Gda8lfG5iCOMWoFH"'
    )
    assert template_key(a) == template_key(b)
    assert "<RAND>" in template_key(a)


def test_template_key_collapses_variable_segment_count():
    # The real cicd error_message splits into a VARIABLE number of space-separated
    # random segments. The count is itself noise — rows with 1, 2 or 3 blob
    # segments must collapse to ONE template, else dedup never fires.
    one = 'stage=deploy msg="ERROR: RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z"'
    two = (
        'stage=deploy msg="ERROR: g8tprJBHkhLu3r6EtkU5E0Y51Gda8lfG5iCOMWoFH '
        '2eykZAqkscUJ4KUxihlYXskJiDSG3TF0CXhFXZp9Tf"'
    )
    three = (
        'stage=deploy msg="ERROR: XSxiDN8XrGhYd9legdlv1fCtI9ILHaTM94tucUUNwL '
        'dfutS465iBuUtmcDowLmZts0LG4y70lpe4I6iefsVT '
        'RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z"'
    )
    assert template_key(one) == template_key(two) == template_key(three)


def test_template_key_keeps_distinct_templates_distinct():
    a = "MEDIUM Jenkins stage=deploy failure_type=Security Scan Failure"
    b = "CRITICAL GitLab stage=build failure_type=Network Error"
    assert template_key(a) != template_key(b)


def test_mask_random_preserves_structure():
    line = 'msg="ERROR: RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z" stage=deploy'
    masked = mask_random(line)
    assert "<RAND>" in masked
    assert "stage=deploy" in masked  # structure untouched
    assert "RjyJqt" not in masked


def test_signal_ratio_high_for_structured_line():
    # A full staged cicd line: lots of field names + a couple of blobs.
    line = (
        "2025-12-29T07:58:16 MEDIUM Jenkins pipeline=pipe_2032 repo=repo_469 "
        "branch=release lang=Python os=windows-latest cloud=On-Prem stage=deploy "
        "failure_type=Security Scan Failure error_code=ERR_621 retry=3 flaky=True "
        'msg="ERROR: RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z"'
    )
    # Structured fields dominate, so it is NOT trash even with a blob in msg.
    assert signal_ratio(line) > 0.45
    assert not is_low_signal(line)


def test_signal_ratio_low_for_pure_blob_line():
    line = (
        "ERROR: hqMA49ZWCXsrHsXy81t5pCoPJBer4QpdnJdelM7WPCykUZENL6T469HHsAybKVxlk"
        "1JyuWWQLf3x3eRPFkiO8LKPIyMEfFtC4iKqVTzy9nAKMz0ziBAIFCXD"
    )
    assert signal_ratio(line) < 0.15
    assert is_low_signal(line)


def test_short_lines_not_judged_as_trash():
    # Below the min-alnum guard: cheap, non-dominating — never dropped.
    assert not is_low_signal("200")
    assert not is_low_signal("OK")
    assert not is_low_signal("x7y9")


def test_signal_ratio_empty_and_symbolic():
    assert signal_ratio("") == 1.0
    assert signal_ratio("--- === >>>") == 1.0  # no alnum → nothing to judge


def test_entropy_and_case_helpers_edge_cases():
    from polymorph_lamr.distill.normalize import _case_transition_rate, _entropy

    assert _entropy("") == 0.0
    assert _case_transition_rate("a") == 0.0  # < 2 letters → no transitions
    # An all-digit run has no letters → not random (handled by NUM mask upstream).
    assert not is_random_token("123456789012")


def test_template_key_cached_matches_uncached():
    line = "MEDIUM Jenkins stage=deploy error_code=ERR_621"
    assert template_key_cached(line) == template_key(line)
    # Second call hits the LRU.
    assert template_key_cached(line) == template_key(line)
