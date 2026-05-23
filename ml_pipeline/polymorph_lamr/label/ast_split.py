"""Split a teacher binary keep/drop mask into two soft training labels:

  y_semantic[i]   — strength of evidence that token i is "prose / semantic"
  y_dependency[i] — strength of evidence that token i is "structural scaffold"

We compute these by walking the tree-sitter AST: for each token's byte-span,
locate the deepest enclosing node, then climb to find the nearest "scaffold"
ancestor. The hop distance feeds a decay kernel; `w_dep = kernel(h)`,
`w_sem = 1 - w_dep`.

For non-source-code (markdown, plaintext) we skip the AST walk and return all
weight on `y_semantic`. The trainer zero-weights the dep head for these.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Mapping
from typing import Any, Callable, Optional

from .hop_decay import exp_decay, get_kernel


# Default scaffold node types per language. Mirrors configs/default.yaml.
_DEFAULT_SCAFFOLDS: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "import_statement",
        "import_from_statement",
        "decorator",
        "parameters",
        "typed_parameter",
        "typed_default_parameter",
        "type",
        "call",
    },
    "json": {"object", "pair", "array"},
}


@lru_cache(maxsize=4)
def _parser_for(lang: str):
    # Imported lazily so the QC/label code paths that don't need tree-sitter
    # (e.g. pure-prose distillation QC) don't pay the binding cost.
    import tree_sitter as ts  # type: ignore

    if lang == "python":
        import tree_sitter_python as tsp  # type: ignore

        language = ts.Language(tsp.language())
    elif lang == "json":
        import tree_sitter_json as tsj  # type: ignore

        language = ts.Language(tsj.language())
    else:
        raise ValueError(f"unsupported language: {lang}")
    return ts.Parser(language)


@dataclass(frozen=True)
class SplitLabels:
    keep_mask: list[bool]
    w_semantic: list[float]
    w_dependency: list[float]
    is_code: bool


def _hop_to_scaffold(node, scaffolds: set[str], max_hops: int = 16) -> int:
    hops = 0
    cur = node
    while cur is not None and hops < max_hops:
        if cur.type in scaffolds:
            return hops
        cur = cur.parent
        hops += 1
    return max_hops  # cap, kernel will produce ~0 weight


def _descendant_at_byte(root, byte_offset: int):
    """Smallest node whose byte range contains `byte_offset`."""
    return root.descendant_for_byte_range(byte_offset, byte_offset)


def split_labels(
    text: str,
    keep_mask: list[bool],
    spans: list[tuple[int, int]],
    lang: Optional[str],
    kernel: Optional[Callable[[int], float]] = None,
    scaffold_types: Optional[set[str]] = None,
) -> SplitLabels:
    """Build `y_semantic` and `y_dependency` weights for each token.

    Args:
        text: original source text (the one that was tokenized).
        keep_mask: teacher's binary signal, same length as `spans`.
        spans: (start_byte, end_byte) for each token, from `align.encode_with_spans`.
        lang: 'python' | 'json' | None (None → no AST, all prose).
        kernel: hop-decay function. Defaults to exp_decay(alpha=0.5).
        scaffold_types: override default scaffold node-type set for `lang`.
    """
    if len(keep_mask) != len(spans):
        raise ValueError("keep_mask and spans must have the same length")

    n = len(spans)
    if kernel is None:
        kernel = exp_decay(alpha=0.5)

    if lang is None or lang not in _DEFAULT_SCAFFOLDS:
        # Prose path: all semantic, no dependency signal.
        return SplitLabels(
            keep_mask=list(keep_mask),
            w_semantic=[1.0] * n,
            w_dependency=[0.0] * n,
            is_code=False,
        )

    scaffolds = scaffold_types or _DEFAULT_SCAFFOLDS[lang]
    parser = _parser_for(lang)
    tree = parser.parse(text.encode("utf-8"))
    root = tree.root_node

    w_dep = [0.0] * n
    w_sem = [0.0] * n
    for i, (start, end) in enumerate(spans):
        # Probe the middle of the token to avoid landing on a boundary
        # between two sibling leaves.
        probe = start + max(0, (end - start) // 2)
        node = _descendant_at_byte(root, probe)
        if node is None:
            wd = 0.0
        else:
            h = _hop_to_scaffold(node, scaffolds)
            wd = float(kernel(h))
            wd = max(0.0, min(1.0, wd))
        w_dep[i] = wd
        w_sem[i] = 1.0 - wd

    return SplitLabels(
        keep_mask=list(keep_mask),
        w_semantic=w_sem,
        w_dependency=w_dep,
        is_code=True,
    )


def split_labels_from_config(
    text: str,
    keep_mask: list[bool],
    spans: list[tuple[int, int]],
    lang: Optional[str],
    cfg: Mapping[str, Any],
) -> SplitLabels:
    """Config-aware wrapper for the `label:` block in configs/default.yaml."""
    kernel = get_kernel(dict(cfg.get("hop_decay", {})))
    scaffold_cfg = cfg.get("scaffold_node_types", {})
    scaffold_types: set[str] | None = None
    if lang is not None and isinstance(scaffold_cfg, Mapping) and lang in scaffold_cfg:
        scaffold_types = set(scaffold_cfg[lang])
    return split_labels(
        text=text,
        keep_mask=keep_mask,
        spans=spans,
        lang=lang,
        kernel=kernel,
        scaffold_types=scaffold_types,
    )
