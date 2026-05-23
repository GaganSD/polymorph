#!/usr/bin/env bash
# E2E smoke: distill (mocked) -> label -> 1 train step -> export ONNX.
# All API calls are mocked via the fake litellm in tests/test_distill_smoke.py.
# Runs on CPU, no GPU, no API keys.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[smoke] running pytest test suite (no GPU, no API keys)"
python -m pytest -q --maxfail=1

echo "[smoke] building a tiny labeled shard from fixtures"
python - <<'PY'
import json
from pathlib import Path

from polymorph_lamr.label.align import derive_mask
from polymorph_lamr.label.ast_split import split_labels

fixtures = Path("tests/fixtures")
shard = Path("artifacts/lamr-smoke/shard.jsonl")
shard.parent.mkdir(parents=True, exist_ok=True)

records = []
for path in fixtures.iterdir():
    text = path.read_text()
    # "Compression" = drop every other word (mock teacher).
    words = text.split()
    compressed = " ".join(words[::2])
    align = derive_mask(text, compressed)
    lang = "python" if path.suffix == ".py" else None
    split = split_labels(text, align.keep_mask, align.spans, lang=lang)
    # Tag convention: 0 = keep, 1 = drop.  keep_mask is True for kept tokens.
    tags = [0 if k else 1 for k in split.keep_mask]
    records.append({
        "input_ids": align.token_ids,
        "tags": tags,
        "w_semantic": split.w_semantic,
        "w_dependency": split.w_dependency,
        "is_code": split.is_code,
        "src_path": str(path),
    })

with shard.open("w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"[smoke] wrote {len(records)} labeled samples to {shard}")
PY

echo "[smoke] dry-run train (param count + device)"
python -m polymorph_lamr.train.train --config configs/default.yaml --dry-run

echo "[smoke] training 1 step on the shard"
python -m polymorph_lamr.train.train \
    --config configs/default.yaml \
    --shards artifacts/lamr-smoke/shard.jsonl \
    --out artifacts/lamr-smoke/ckpts \
    --max-steps 1

echo "[smoke] exporting ONNX"
python -m polymorph_lamr.export.to_onnx \
    --ckpt artifacts/lamr-smoke/ckpts/ckpt-final.pt \
    --out artifacts/lamr-smoke/onnx \
    --config configs/default.yaml \
    --parity-seq-len 32

echo "[smoke] artifacts:"
ls -la artifacts/lamr-smoke/onnx
echo "[smoke] OK"
