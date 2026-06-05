"""End-to-end CLI test for the post-distill pipeline (lamr-pipeline).

Builds a tiny synthetic distilled JSONL (clean prose pairs + one .py source so
the AST path is exercised + one obvious hallucination QC must drop), runs the
CLI, and asserts:
  * rc == 0
  * train.jsonl + val.jsonl exist
  * every emitted line matches the shard schema AND round-trips through
    LabeledShardDataset / collate
  * the hallucinated pair was dropped by QC
  * the split is deterministic (same seed -> byte-identical shards) and produces
    a non-empty val set (chunk-level ~val-frac sample)
"""

import json
from pathlib import Path

import torch

from polymorph_lamr.pipeline import main
from polymorph_lamr.train.dataset import LabeledShardDataset, collate

# A pair the QC hard floor (VR > 0.5) must reject: the "compressed" text is almost
# entirely novel tokens absent from the original — a textbook hallucination.
_HALLUCINATED_SRC = "corpus_h:logs/hallucinated.txt"
_HALLUCINATED = {
    "original": "alpha beta gamma delta epsilon zeta eta theta iota kappa",
    "compressed": "wholly invented unrelated fabricated nonsense phrases everywhere",
    "src_path": _HALLUCINATED_SRC,
    "chunk_id": 0,
}

# Clean extractive pairs (drop-every-other-word) — VR == 0, AG == 0, kept by QC.
_PY_SRC = "corpus_a:repo/calc.py"
_PY_ORIGINAL = (
    "def area(radius: float) -> float:\n"
    "    if radius < 0:\n"
    "        raise ValueError(\"radius must be non-negative\")\n"
    "    return 3.14159 * radius * radius\n"
)


def _extractive(text: str) -> str:
    """Mock-teacher compression: keep every other whitespace token."""
    words = text.split()
    return " ".join(words[::2])


def _write_distilled(path: Path) -> None:
    records = [_HALLUCINATED]
    # One .py source (two chunks, same src_path) to exercise the AST path. Chunks
    # of one source may land in different splits now (chunk-level split).
    for cid in range(2):
        records.append(
            {
                "original": _PY_ORIGINAL,
                "compressed": _extractive(_PY_ORIGINAL),
                "src_path": _PY_SRC,
                "chunk_id": cid,
            }
        )
    # A handful of distinct prose sources so the split has something to distribute.
    for i in range(8):
        original = (
            f"log line {i} alpha bravo charlie delta echo foxtrot golf hotel "
            f"india juliet kilo lima mike november oscar papa quebec romeo {i}"
        )
        records.append(
            {
                "original": original,
                "compressed": _extractive(original),
                "src_path": f"corpus_p:logs/prose_{i}.txt",
                "chunk_id": 0,
            }
        )
    with path.open("w", encoding="utf-8") as fh:
        # Interleave a couple of blank lines to exercise the skipping path.
        for idx, rec in enumerate(records):
            fh.write(json.dumps(rec) + "\n")
            if idx == 0:
                fh.write("\n")


def _read_shard(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assert_shard_schema(rec: dict) -> None:
    assert set(rec) >= {
        "input_ids",
        "tags",
        "w_semantic",
        "w_dependency",
        "is_code",
        "src_path",
    }
    n = len(rec["input_ids"])
    assert n > 0
    assert len(rec["tags"]) == n
    assert len(rec["w_semantic"]) == n
    assert len(rec["w_dependency"]) == n
    assert all(t in (0, 1) for t in rec["tags"])
    assert isinstance(rec["is_code"], bool)
    assert isinstance(rec["src_path"], str)


def test_pipeline_cli_end_to_end(tmp_path, capsys):
    distilled = tmp_path / "distilled.jsonl"
    out_dir = tmp_path / "shards"
    _write_distilled(distilled)

    rc = main(["--distilled", str(distilled), "--out-dir", str(out_dir), "--seed", "42", "--val-frac", "0.3"])
    assert rc == 0

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    assert train_path.exists()
    assert val_path.exists()

    train = _read_shard(train_path)
    val = _read_shard(val_path)
    all_recs = train + val
    assert all_recs, "pipeline emitted no shards"

    # Schema check on every emitted line.
    for rec in all_recs:
        _assert_shard_schema(rec)

    # The hallucinated pair must have been dropped by QC.
    src_paths = {r["src_path"] for r in all_recs}
    assert _HALLUCINATED_SRC not in src_paths

    # The .py source must have produced a code shard (AST path exercised).
    py_recs = [r for r in all_recs if r["src_path"] == _PY_SRC]
    assert py_recs, "expected the .py source to survive QC and be emitted"
    assert any(r["is_code"] for r in py_recs)

    # Round-trip: the emitted shards must be loadable by the dataset reader and
    # produce a usable batch.
    ds = LabeledShardDataset([train_path, val_path], max_seq_len=64, shuffle_files=False)
    samples = list(ds)
    assert samples
    batch = collate(samples)
    assert batch["input_ids"].shape[0] == len(samples)
    assert batch["input_ids"].dtype == torch.long
    assert set(batch) >= {"input_ids", "tags", "w_semantic", "w_dependency", "attention_mask"}

    # Chunk-level split must distribute into BOTH train and val — regression guard
    # for the empty-val bug a source-level split produced on few-source corpora.
    assert train, "train split is empty"
    assert val, "val split is empty"

    # Capsys: the QC drop count + summary were printed.
    out = capsys.readouterr().out
    assert "[qc]" in out
    assert "[summary]" in out


def test_pipeline_split_is_deterministic(tmp_path):
    distilled = tmp_path / "distilled.jsonl"
    _write_distilled(distilled)

    def run(out_name: str) -> tuple[str, str]:
        out_dir = tmp_path / out_name
        rc = main(
            ["--distilled", str(distilled), "--out-dir", str(out_dir), "--seed", "42", "--val-frac", "0.3"]
        )
        assert rc == 0
        return (
            (out_dir / "train.jsonl").read_text(),
            (out_dir / "val.jsonl").read_text(),
        )

    assert run("run1") == run("run2"), "same seed must yield byte-identical shards"


def test_pipeline_no_lang_detect_forces_prose(tmp_path):
    distilled = tmp_path / "distilled.jsonl"
    out_dir = tmp_path / "shards"
    _write_distilled(distilled)

    rc = main(
        ["--distilled", str(distilled), "--out-dir", str(out_dir), "--no-lang-detect"]
    )
    assert rc == 0
    recs = _read_shard(out_dir / "train.jsonl") + _read_shard(out_dir / "val.jsonl")
    # With AST detection disabled, even the .py source is labeled prose.
    assert all(r["is_code"] is False for r in recs)
