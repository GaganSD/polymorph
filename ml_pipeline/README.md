# Polymorph LaMR - Training & Distillation Pipeline

Python pipeline for training the optional **LaMR** learned pruner used by the
Rust MCP server. The shipped `mb_v0` model is a ModernBERT-150M token classifier
with a single per-token drop head. Export emits `model.onnx` plus optional
`decode.json`; the Rust runtime handles tokenization, windowed `tract` inference,
and span-aware word decode.

```
ml_pipeline/
├── polymorph_lamr/
│   ├── distill/    # async litellm client, Claude + GPT-4o
│   ├── qc/         # LLMLingua-2 VR + AG metrics + percentile filter
│   ├── label/      # LCS alignment + tree-sitter AST hop-decay split
│   ├── model/      # ModernBERT / transformer / legacy stub backbones
│   ├── train/      # IterableDataset, token-classifier loss, AMP loop
│   └── export/     # ONNX + decode.json + parity check
├── tests/          # pytest, runs CPU-only with no API keys
├── scripts/run_e2e_smoke.sh
└── configs/default.yaml
```

## Quick start

```bash
cd ml_pipeline
uv venv && source .venv/bin/activate         # or python -m venv
uv pip install -e '.[dev]'                   # core + pytest
uv pip install -e '.[train]'                 # + wandb (optional)
```

Training labels are derived from byte alignment and can be projected across
tokenizers. The shipped runtime uses the ModernBERT tokenizer in
`assets/modernbert/tokenizer.json`; keep the training/export tokenizer aligned
with that artifact.

## End-to-end smoke (no API spend, no GPU)

```bash
./scripts/run_e2e_smoke.sh
```

This runs pytest, builds a tiny labeled shard from `tests/fixtures/`, executes
one training step on CPU, and exports an ONNX artifact under
`artifacts/lamr-smoke/onnx/`.

## Pipeline stages

### 1. Distill (`polymorph_lamr.distill`)

```bash
OPENROUTER_API_KEY=sk-or-... \
lamr-distill --in data/raw/trainticket --out data/distilled/trainticket.jsonl --concurrency 8
```

- **Default: OpenRouter open-weight teacher ensemble** (E3) — Qwen-2.5-72B,
  DeepSeek, Llama-3.3-70B. Each chunk is compressed by every teacher and the
  per-chunk **best-QC** output (VR==0, lowest Alignment Gap) is kept as the
  training target. Override teachers with `--teachers name=model ...`.
- Uses the `LOG_TRACE_EXTRACTIVE` prompt (telemetry-tuned, extractive-only) and
  log-aware chunking (`.log`/`.jsonl` split on lines).
- Output JSONL: `{src_path, chunk_id, original, outputs{teacher->text},
  compressed, chosen_teacher, qc{vr,ag,mr,hr}, cost_usd, errors}`.
- Legacy two-teacher mode (Claude + GPT-4o): `--mode pair`.
- Full end-to-end path (fetch → benchmark gate → distill → train): see
  [`RUNBOOK_OPENROUTER.md`](RUNBOOK_OPENROUTER.md).

### 2. Quality control (`polymorph_lamr.qc`)

```python
from polymorph_lamr.qc.metrics import QCRecord
from polymorph_lamr.qc.filter import filter_records, write_report

records = [QCRecord.compute(orig, comp) for ...]
survivors, report = filter_records(records, vr_drop_top_pct=5.0, ag_drop_top_pct=10.0)
```

Implements **Variation Rate** and **Alignment Gap** from
`research/Advanced_Data_Distillation_for_Token_Deletion.md` §Quality Control.
Drops top-5% VR (hallucinations) then top-10% AG (alignment failures).

### 3. Label (`polymorph_lamr.label`)

```python
from polymorph_lamr.label.align import derive_mask
from polymorph_lamr.label.ast_split import split_labels

align = derive_mask(original, compressed)
split = split_labels(original, align.keep_mask, align.spans, lang="python")
# split.keep_mask is the extractive keep/drop target for each token
```

- `align.derive_mask`: byte-level LCS alignment produces a binary keep/drop mask.
- `split_labels`: walks the tree-sitter AST (via `tree-sitter-languages`,
  which ships compiled grammars matching `grammars/tree-sitter-*.wasm`),
  computes hop distance to scaffold nodes, and preserves dependency weights used
  by older experiments.

### 4. Train (`polymorph_lamr.train`)

```bash
lamr-train --config configs/default.yaml --shards data/labeled/*.jsonl --out artifacts/ckpts
```

- Dry-run (no shards, prints param count + memory estimate):
  `lamr-train --config configs/default.yaml --dry-run`
- Single-node `torchrun` works; cluster deployment is the user's problem.
- ModernBERT configs train a per-token drop classifier. Older configs still keep
  the lightweight transformer and stub backbones for ablations.

### 5. Export (`polymorph_lamr.export.to_onnx`)

```bash
lamr-export --ckpt artifacts/ckpts/ckpt-final.pt --out artifacts/lamr-v0 \
            --config configs/default.yaml
```

Emits:

- `model.onnx` — backbone + per-token drop logits.
- `decode.json` — default target rate used by the Rust span decoder.
- `config.yaml`, `README.md`, `parity.json`.

The Rust runtime applies sigmoid, scores consecutive 512-token windows, and
drops whole whitespace-word spans most-droppable-first while respecting the
structural lock mask.

## Architecture notes

### Backbones
`modernbert` is the shipped path. `transformer` and `deltanet_stub` remain for
tests and ablations; they are not the production runtime.

### Decode
The earlier dual-CRF / Viterbi design was dropped. The production decode is a
calibrated target-rate cut over model probabilities, grouped into word spans.
This matched the answer-survival benchmark better and simplified the Rust path.

### ONNX export strategy
Only the static classifier graph is exported. Rust owns tokenization, windowing,
sigmoid, and span decode so the deployed server has no Python dependency.

## Testing

```bash
pytest -q                                # CPU-only, mocked APIs
RUN_REAL_API=1 pytest tests/test_distill_smoke.py::test_distill_pair_real_api
                                         # opt-in, hits Claude + GPT (~$0.05)
```

## Out of scope

- RLAIF / RLHF post-training (described in the research doc, not built here).
- Replacing the shipped ModernBERT model with a smaller distilled model.
- Distributed multi-node training scripts.
