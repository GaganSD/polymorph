# Polymorph LaMR — Training & Distillation Pipeline

Python pipeline that trains the **Latent Multi-Rubric (LaMR)** dual-CRF token
classifier used by the Polymorph Rust MCP server. Produces a self-contained
ONNX artifact (`model.onnx` + side-car CRF transitions) that the Rust runtime
loads at inference time to replace the mock pruner at `src/lamr.rs`.

```
ml_pipeline/
├── polymorph_lamr/
│   ├── distill/    # async litellm client, Claude + GPT-4o
│   ├── qc/         # LLMLingua-2 VR + AG metrics + percentile filter
│   ├── label/      # LCS alignment + tree-sitter AST hop-decay split
│   ├── model/      # stubbed Gated DeltaNet-2 + MoE + dual CRF
│   ├── train/      # IterableDataset, joint NLL loss, AMP loop
│   └── export/     # ONNX + transitions.npz + parity check
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

Tokenizer is `tiktoken cl100k_base` — must match `src/tokens.rs` in the Rust
runtime. Don't change without re-training.

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
lamr-distill --in data/raw --out data/distilled.jsonl --concurrency 8
```

- Calls `anthropic/claude-3-5-sonnet-latest` (max-compression) and
  `openai/gpt-4o` (reasoning-preserving) concurrently per chunk.
- Chunks are sentence-bounded, ≤512 cl100k tokens (LLMLingua-2 §Dataset
  Distillation chunk-wise protocol).
- Output: JSONL `{src_path, chunk_id, original, claude, gpt4o, cost_usd}`.

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
# split.w_semantic[i] + split.w_dependency[i] == 1.0 for each token
```

- `align.derive_mask`: LCS over cl100k token-id streams produces a binary
  keep/drop mask.
- `split_labels`: walks the tree-sitter AST (via `tree-sitter-languages`,
  which ships compiled grammars matching `grammars/tree-sitter-*.wasm`),
  computes hop distance to scaffold nodes, applies `exp(-α·h)` decay.

### 4. Train (`polymorph_lamr.train`)

```bash
lamr-train --config configs/default.yaml --shards data/labeled/*.jsonl --out artifacts/ckpts
```

- Dry-run (no shards, prints param count + memory estimate):
  `lamr-train --config configs/default.yaml --dry-run`
- Single-node `torchrun` works; cluster deployment is the user's problem.
- Joint loss: `L = λ_s · NLL_sem + λ_d · NLL_dep` with per-token weights from
  the hop-decay split.

### 5. Export (`polymorph_lamr.export.to_onnx`)

```bash
lamr-export --ckpt artifacts/ckpts/ckpt-final.pt --out artifacts/lamr-v0 \
            --config configs/default.yaml
```

Emits:

- `model.onnx` — backbone + MoE + emission heads (dynamic seq axis).
- `transitions.npz` — `{sem,dep}_{trans,start,end}` (2×2 each, tiny).
- `config.yaml`, `README.md`, `parity.json`.

Viterbi runs in Rust; see `ARTIFACT_OUT/README.md` for the decode protocol.

## Architecture notes

### Stubbed backbone
`polymorph_lamr/model/backbone.py:GatedDeltaNet2Stub` is a small bidirectional
Transformer encoder. Search for the marker `_TODO_REAL_DELTANET` when
swapping to the real kernel.

### Dual CRF rationale
Single-CRF pruning collapses semantic-evidence and dependency-scaffold
transitions into one matrix — see
`research/Context Management Papers Analysis.md`. We train two independent
linear-chain CRFs whose emissions come from the same MoE-mixed hidden state.
At inference the Rust side decodes each head separately and combines.

### ONNX export strategy
Viterbi is dynamic-length and brittle under ONNX/TRT export. We export only
the static graph (backbone + MoE + emissions) and ship the tiny `(2,2)`
transition matrices as a side-car. Rust decodes — the runtime already does
sequential work, so the marginal cost is negligible.

## Testing

```bash
pytest -q                                # CPU-only, mocked APIs
RUN_REAL_API=1 pytest tests/test_distill_smoke.py::test_distill_pair_real_api
                                         # opt-in, hits Claude + GPT (~$0.05)
```

## Out of scope

- RLAIF / RLHF post-training (described in the research doc, not built here).
- The real Gated DeltaNet-2 kernel (stub now, swap later).
- Rust-side ONNX loader + Viterbi decoder (separate work item in `src/lamr.rs`).
- Distributed multi-node training scripts.
