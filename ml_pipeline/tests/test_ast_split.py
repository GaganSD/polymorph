"""AST split: tokens inside a function signature should be marked
high-dependency; tokens inside the docstring body should be high-semantic.

We assert distributions, not exact token-level equalities — the cl100k
tokenisation of `def`/identifiers is not 1:1 with AST nodes."""

import statistics

import pytest

pytest.importorskip("tree_sitter_python")

from polymorph_lamr.label.align import encode_with_spans
from polymorph_lamr.label.ast_split import split_labels, split_labels_from_config


PY_SRC = """def compute_area(radius: float) -> float:
    '''Return the area of a circle.'''
    return 3.14 * radius * radius
"""


def _byte_to_token_idx(spans, start, end):
    """Return token indices whose byte spans overlap [start, end)."""
    return [i for i, (a, b) in enumerate(spans) if a < end and b > start]


def test_function_signature_dep_outweighs_semantic():
    ids, spans = encode_with_spans(PY_SRC)
    keep = [True] * len(ids)
    split = split_labels(PY_SRC, keep, spans, lang="python")
    assert split.is_code

    raw = PY_SRC.encode("utf-8")
    sig_end = raw.index(b":")  # end of `def compute_area(radius: float) -> float`
    sig_idxs = _byte_to_token_idx(spans, 0, sig_end)
    assert sig_idxs, "test setup: no signature tokens found"
    sig_dep = statistics.mean(split.w_dependency[i] for i in sig_idxs)
    sig_sem = statistics.mean(split.w_semantic[i] for i in sig_idxs)
    assert sig_dep > sig_sem, f"signature should be more dep than sem: dep={sig_dep:.3f} sem={sig_sem:.3f}"


def test_docstring_body_is_less_dependency_than_signature():
    ids, spans = encode_with_spans(PY_SRC)
    split = split_labels(PY_SRC, [True] * len(ids), spans, lang="python")
    raw = PY_SRC.encode("utf-8")
    sig_idxs = _byte_to_token_idx(spans, 0, raw.index(b":"))
    doc_start = raw.index(b"Return")
    doc_end = doc_start + len(b"Return the area of a circle.")
    doc_idxs = _byte_to_token_idx(spans, doc_start, doc_end)
    assert sig_idxs and doc_idxs
    sig_dep = statistics.mean(split.w_dependency[i] for i in sig_idxs)
    doc_dep = statistics.mean(split.w_dependency[i] for i in doc_idxs)
    assert doc_dep < sig_dep


def test_markdown_routes_all_weight_to_semantic():
    text = "# Title\n\nThis is some prose with no AST.\n"
    ids, spans = encode_with_spans(text)
    keep = [True] * len(ids)
    split = split_labels(text, keep, spans, lang=None)
    assert not split.is_code
    assert all(w == 1.0 for w in split.w_semantic)
    assert all(w == 0.0 for w in split.w_dependency)


def test_weights_sum_to_one_per_token():
    ids, spans = encode_with_spans(PY_SRC)
    keep = [True] * len(ids)
    split = split_labels(PY_SRC, keep, spans, lang="python")
    for i, (ws, wd) in enumerate(zip(split.w_semantic, split.w_dependency)):
        assert abs((ws + wd) - 1.0) < 1e-9, f"token {i}: ws={ws} wd={wd}"


def test_json_scaffold_routes_dependency_weight():
    pytest.importorskip("tree_sitter_json")
    text = '{"tool": {"name": "lock_mask", "args": [1, 2]}}'
    ids, spans = encode_with_spans(text)
    split = split_labels(text, [True] * len(ids), spans, lang="json")
    assert split.is_code
    assert max(split.w_dependency) > 0.9
    assert all(abs((ws + wd) - 1.0) < 1e-9 for ws, wd in zip(split.w_semantic, split.w_dependency))


def test_config_drives_kernel_and_scaffolds():
    ids, spans = encode_with_spans(PY_SRC)
    cfg = {
        "hop_decay": {"kernel": "linear", "max_hops": 1},
        "scaffold_node_types": {"python": ["identifier"]},
    }
    split = split_labels_from_config(PY_SRC, [True] * len(ids), spans, "python", cfg)
    raw = PY_SRC.encode("utf-8")
    name_start = raw.index(b"compute_area")
    name_idxs = _byte_to_token_idx(spans, name_start, name_start + len(b"compute_area"))
    assert name_idxs
    assert max(split.w_dependency[i] for i in name_idxs) == 1.0


def test_unsupported_language_routes_to_prose():
    ids, spans = encode_with_spans(PY_SRC)
    split = split_labels(PY_SRC, [True] * len(ids), spans, lang="markdown")
    assert not split.is_code
    assert all(w == 1.0 for w in split.w_semantic)
    assert all(w == 0.0 for w in split.w_dependency)


def test_keep_mask_and_spans_must_match():
    ids, spans = encode_with_spans(PY_SRC)
    with pytest.raises(ValueError, match="same length"):
        split_labels(PY_SRC, [True] * (len(ids) + 1), spans, lang="python")
