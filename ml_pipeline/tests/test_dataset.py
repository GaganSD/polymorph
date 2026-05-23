"""Streaming dataset + collator."""

import json
from pathlib import Path

import torch

from polymorph_lamr.train.dataset import LabeledShardDataset, Sample, _window, collate


def _make_shard(tmp_path: Path, samples: list[dict]) -> Path:
    p = tmp_path / "shard.jsonl"
    with p.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return p


def test_window_returns_full_for_short_sequence():
    assert list(_window(list(range(10)), max_len=32, stride=16)) == [(0, 10)]


def test_window_overlaps_when_too_long():
    n = 100
    spans = list(_window(list(range(n)), max_len=40, stride=20))
    # Must cover the entire sequence.
    assert spans[0][0] == 0
    assert spans[-1][1] == n
    for a, b in spans:
        assert b - a <= 40


def test_dataset_iterates_and_yields_samples(tmp_path):
    samples = [
        {
            "input_ids": [1, 2, 3, 4],
            "tags": [0, 1, 0, 1],
            "w_semantic": [1.0, 0.5, 0.5, 1.0],
            "w_dependency": [0.0, 0.5, 0.5, 0.0],
            "is_code": False,
            "src_path": "x",
        },
        {
            "input_ids": [5, 6],
            "tags": [0, 0],
            "w_semantic": [1.0, 1.0],
            "w_dependency": [0.0, 0.0],
            "is_code": True,
            "src_path": "y",
        },
    ]
    shard = _make_shard(tmp_path, samples)
    ds = LabeledShardDataset([shard], max_seq_len=16, shuffle_files=False)
    out = list(ds)
    assert len(out) == 2
    assert out[0].input_ids == [1, 2, 3, 4]
    assert out[1].is_code is True


def test_dataset_skips_misaligned_rows(tmp_path):
    samples = [
        {  # length mismatch
            "input_ids": [1, 2, 3],
            "tags": [0, 1],
            "w_semantic": [1.0, 1.0],
            "w_dependency": [0.0, 0.0],
        },
        {  # good
            "input_ids": [9],
            "tags": [0],
            "w_semantic": [1.0],
            "w_dependency": [0.0],
        },
    ]
    shard = _make_shard(tmp_path, samples)
    out = list(LabeledShardDataset([shard], max_seq_len=8))
    assert len(out) == 1
    assert out[0].input_ids == [9]


def test_collate_pads_and_masks_correctly():
    samples = [
        Sample([1, 2, 3], [0, 1, 0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], False),
        Sample([4], [1], [0.5], [0.5], True),
    ]
    batch = collate(samples)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["tags"][1, 1].item() == 0  # padding tag
    assert batch["w_semantic"][1, 1].item() == 0.0
    assert batch["w_dependency"][1, 1].item() == 0.0


def test_window_splits_at_boundary():
    # Sequence exactly at max_len returns one window.
    spans = list(_window(list(range(40)), max_len=40, stride=20))
    assert spans == [(0, 40)]
