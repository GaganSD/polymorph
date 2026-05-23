"""Streaming dataset over labeled shards.

A "labeled shard" is a JSONL file where each line is:
    {
      "input_ids": [int, ...],     # cl100k token ids
      "tags": [0|1, ...],          # gold tags, same length as input_ids
      "w_semantic": [float, ...],  # per-token weight for semantic head
      "w_dependency": [float, ...],# per-token weight for dependency head
      "is_code": bool,
      "src_path": str
    }

Shorter sequences are right-padded inside the collator to `max_seq_len`.
Longer sequences are split into overlapping windows at load time.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

PAD_TAG = 0  # keep — padding never contributes to loss (mask hides it)


@dataclass
class Sample:
    input_ids: list[int]
    tags: list[int]
    w_semantic: list[float]
    w_dependency: list[float]
    is_code: bool


def _window(seq: list, max_len: int, stride: int) -> Iterator[tuple[int, int]]:
    n = len(seq)
    if n <= max_len:
        yield 0, n
        return
    i = 0
    while i < n:
        end = min(n, i + max_len)
        yield i, end
        if end == n:
            return
        i += stride


class LabeledShardDataset(IterableDataset):
    def __init__(
        self,
        shard_paths: list[Path],
        max_seq_len: int = 1024,
        stride: int | None = None,
        shuffle_files: bool = True,
        seed: int = 1337,
    ):
        super().__init__()
        self.shard_paths = [Path(p) for p in shard_paths]
        self.max_seq_len = max_seq_len
        # Default to non-overlapping windows. Overlap double-counts middle
        # tokens in the gradient, which biases the loss toward the middle of
        # long sequences. Callers that want overlap must opt in explicitly.
        self.stride = stride or max_seq_len
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(self) -> Iterator[Sample]:
        info = get_worker_info()
        files = list(self.shard_paths)
        if self.shuffle_files:
            import random

            random.Random(self.seed).shuffle(files)
        if info is not None:
            files = [f for i, f in enumerate(files) if i % info.num_workers == info.id]

        for shard in files:
            with shard.open() as f:
                for line in f:
                    raw = json.loads(line)
                    ids = raw["input_ids"]
                    tags = raw["tags"]
                    w_s = raw["w_semantic"]
                    w_d = raw["w_dependency"]
                    is_code = bool(raw.get("is_code", False))
                    if not (len(ids) == len(tags) == len(w_s) == len(w_d)):
                        continue
                    for a, b in _window(ids, self.max_seq_len, self.stride):
                        yield Sample(
                            input_ids=ids[a:b],
                            tags=tags[a:b],
                            w_semantic=w_s[a:b],
                            w_dependency=w_d[a:b],
                            is_code=is_code,
                        )


def collate(samples: list[Sample], pad_id: int = 0, max_len: int | None = None) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("empty batch")
    target_len = max_len or max(len(s.input_ids) for s in samples)
    b = len(samples)

    input_ids = torch.full((b, target_len), pad_id, dtype=torch.long)
    tags = torch.full((b, target_len), PAD_TAG, dtype=torch.long)
    w_sem = torch.zeros((b, target_len), dtype=torch.float)
    w_dep = torch.zeros((b, target_len), dtype=torch.float)
    mask = torch.zeros((b, target_len), dtype=torch.bool)

    for i, s in enumerate(samples):
        n = len(s.input_ids)
        input_ids[i, :n] = torch.tensor(s.input_ids, dtype=torch.long)
        tags[i, :n] = torch.tensor(s.tags, dtype=torch.long)
        w_sem[i, :n] = torch.tensor(s.w_semantic, dtype=torch.float)
        w_dep[i, :n] = torch.tensor(s.w_dependency, dtype=torch.float)
        mask[i, :n] = True

    return {
        "input_ids": input_ids,
        "tags": tags,
        "w_semantic": w_sem,
        "w_dependency": w_dep,
        "attention_mask": mask,
    }
