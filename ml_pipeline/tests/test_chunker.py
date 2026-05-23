"""Chunker: sentence/blockwise packing under cl100k token caps."""

import tiktoken

from polymorph_lamr.distill.chunker import chunk, detect_mode


_enc = tiktoken.get_encoding("cl100k_base")


def _toks(s: str) -> int:
    return len(_enc.encode_ordinary(s))


def test_empty_input_returns_empty():
    assert chunk("") == []
    assert chunk("   \n\n  ") == []


def test_short_text_single_chunk():
    text = "This is a short sentence. This is another."
    chunks = chunk(text, max_tokens=512, mode="prose")
    assert len(chunks) == 1
    assert "short" in chunks[0]


def test_long_prose_splits_under_cap():
    sentences = ["Sentence number {} ends here.".format(i) for i in range(200)]
    text = " ".join(sentences)
    chunks = chunk(text, max_tokens=64, mode="prose")
    assert len(chunks) > 1
    for c in chunks:
        assert _toks(c) <= 64


def test_code_mode_uses_blank_lines():
    code = "def a():\n    return 1\n\ndef b():\n    return 2\n\ndef c():\n    return 3\n"
    chunks = chunk(code, max_tokens=20, mode="code")
    assert len(chunks) >= 2
    # Each block keeps its function intact.
    for c in chunks:
        assert "def" in c


def test_hard_split_when_unit_exceeds_cap():
    # Single "sentence" larger than the cap forces _hard_split path.
    text = "alpha " * 600  # ~600 tokens, no sentence boundary
    chunks = chunk(text, max_tokens=64, mode="prose")
    assert len(chunks) > 1
    for c in chunks:
        assert _toks(c) <= 64


def test_detect_mode_routes_by_extension():
    assert detect_mode("foo.py", "x = 1") == "code"
    assert detect_mode("foo.json", "{}") == "code"
    assert detect_mode("foo.md", "# Title") == "prose"
    assert detect_mode(None, "plain") == "prose"
