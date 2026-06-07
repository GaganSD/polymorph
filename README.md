# Polymorph

> Local-first **context middleware** for logs and traces. Drop a 10 MB log in front of a model; Polymorph returns a smaller payload that still contains the answer — every structural field, error code, and trace ID preserved byte-for-byte, the redundant operational prose pruned by a trained token-classifier. Served over MCP.

Polymorph sits between your log/trace store and the model provider as an MCP server. You feed it a log; it emits a compressed version that fits more signal into the context window. Two layers do the work:

1. **Deterministic structure layer (shipping).** Token-locking (structural JSON fields, severities, error codes, IPs, UUIDs, trace/span IDs are force-kept), template/run-length dedup, and Compress-Cache-Retrieve for long JSON arrays. Lossless and reversible.
2. **Learned pruner — LaMR (trained, benchmarked, not yet wired into the runtime).** A 150M ModernBERT token classifier that scores `P(drop)` per token over the *unlocked prose residual* and drops whole low-salience words to hit a target compression rate. This is the part that removes verbose-but-uninformative chatter without losing the one line that matters.

**Status in one line: the learned pruner is SOTA-for-class on answer survival (below), but the local latency target is not yet met and the model is not yet loaded by the Rust MCP server. Remaining work is tracked in [`TODOS.md`](TODOS.md).**

---

## For the ML engineer: the mental model

You have a 10 MB log and a budget-limited context window. Naively truncating loses the needle; naively compressing (entropy coders, generic summarizers) destroys structure or hallucinates. Polymorph is **extractive** — it only ever *deletes* tokens, never rewrites them — so a status code, IP, or exception type that survives is byte-identical to the original.

```
10 MB log ──> Polymorph MCP ──> compressed context ──> Claude Code
                  │
                  ├─ lock:   force-keep structural atoms (regex + AST + DAAC, O(N))
                  ├─ dedup:  collapse repeated templated lines
                  └─ LaMR:   ModernBERT scores P(drop)/token over the prose residual,
                             span-aware decode drops whole words to a target rate
```

Precise terms used throughout:
- **Drop rate `R`** — fraction of *tokens* removed. **Compression ratio** — `input_tokens / output_tokens` (measured in cl100k tokens as a method-independent yardstick).
- **Answer survival** — given a (log, question, gold-answer) triple, does the gold fact survive compression? Measured two ways: **exact-match** (gold substring present, whitespace/case-normalized) and **LLM-judge** (a YES/NO entailment on the gold fact, robust to paraphrase).
- **Span-aware decode** — keep/drop decisions are made at the granularity of whitespace **words**, never splitting a multi-token word, so a multi-word needle isn't fragmented. Per-word score aggregator is `max` (drop a word if any token is droppable) — empirically the best survival with a calibrated model.
- **Windowed inference** — a long log is scored in consecutive 512-token windows (the training window), so a needle in the tail isn't silently truncated away.

---

## Results

Trained ModernBERT-150M pruner (`mb_v0`), evaluated on 50 LogHub-2.0 prose triples across domains (OpenStack/Spark/Hadoop/BGL/etc.), at **matched compression ratio** (iso-ratio), LLM-judged answer survival (Claude Sonnet 4.6 entailment):

| method | 3× survival | 5× survival |
|---|---|---|
| keep-severity (line-level baseline) | 14% | 10% |
| LLMLingua-2 (560M, published SOTA extractive) | ~20% @3.1× | — |
| **Polymorph LaMR `lamr+span` (150M)** | **68%** | **48%** |

- **~4.9× more answers preserved than keep-severity** at both ratios; **~3.4× over LLMLingua-2** at 3×.
- Trained checkpoint quality: **PR-AUC 0.873, ROC-AUC 0.933, F1@0.30 0.765 (recall 0.797 > precision 0.736)** on the held-out val shard. ONNX export parity `max_abs_diff_logits = 3.9e-6`.
- The **structural floor hurts on unstructured prose** (it locks non-needle atoms and spends budget) — `lamr+span` *without* floor wins (68 vs 62 @3×). The floor is a structured-log tool, not a prose tool.

> Caveat: LLMLingua-2 at 560M ran ~8.4 s/doc on CPU; our 150M model is more accurate, but its own local latency is not yet acceptable (see [`TODOS.md`](TODOS.md)). "SOTA" here means **answer-survival at matched compression among locally-runnable extractive methods** — the accuracy bar is cleared; the latency bar is not.

Reproduce:

```bash
cd ml_pipeline
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc | head -1)"   # judge key
.venv/bin/python -m polymorph_lamr.bench.judge_bench \
  --triples ../data/bench/loghub_triples.json \
  --methods keep-severity,lamr+span,lamr+span+floor \
  --lamr-ckpt ../data/modal_out/mb_v0/ckpt-best.pt \
  --iso-ratio 3,5 --sample 50 --judge-model anthropic/claude-sonnet-4-6
```

---

## Quick Start (deterministic engine, ~2 minutes)

The MCP server today ships the **deterministic** layers. The LaMR neural pruner is a Python/ONNX artifact (above) and is **not yet loaded by the Rust runtime** — the server still uses a mock pruner.

```bash
git clone https://github.com/GaganSD/lulu-polymorph.git
cd lulu-polymorph
cargo build --release

# Demos, no MCP client needed:
./target/release/polymorph-mcp --demo lcm-loop   # auto-archive timeline
./target/release/polymorph-mcp --demo ccr        # JSON-array compress + retrieve
```

Register with an MCP client (`~/.config/claude-code/settings.json` or `.claude/settings.json`):

```json
{
  "mcpServers": {
    "polymorph": {
      "command": "/absolute/path/to/lulu-polymorph/target/release/polymorph-mcp",
      "env": {
        "POLYMORPH_DB_PATH": "~/.polymorph/cache.db",
        "POLYMORPH_GRAMMARS_DIR": "/absolute/path/to/lulu-polymorph/grammars"
      }
    }
  }
}
```

Tools: `lock_mask`, `compress_array`, `polymorph_retrieve_cache`, `lcm_append`, `lcm_describe`, `lcm_expand`.

---

## ML pipeline layout (`ml_pipeline/`)

| Path | What |
|---|---|
| `polymorph_lamr/model/` | `LaMRConfig`/`LaMRModel`; backbones (`transformer`, `modernbert`) |
| `polymorph_lamr/label/align.py` | byte-level keep/drop labeling; tokenizer-agnostic (`cl100k` / `modernbert`), `encode_with_spans` / `decode_tokens` |
| `polymorph_lamr/bench/` | answer-survival benchmark: `methods.py` (compressors), `spandecode.py` (span-aware decode), `structural.py` (floor), `loghub.py` (prose triples), `judge_bench.py` (iso-ratio + LLM judge), `survival.py` |
| `polymorph_lamr/train/` | training loop (AMP, grad-accum, PR-AUC-selected `ckpt-best`) |
| `polymorph_lamr/export/to_onnx.py` | ONNX export (fixed-seq-len 512) + parity check |
| `cloud/train_modal.py` | Modal GPU training entrypoint (run from **repo root**) |
| `configs/modernbert.yaml` | the `mb_v0` recipe (lr 3e-5, warmup 300, batch 32×2, seq 512) |

Training is reproducible and cheap (~18 min on an A100, ~$3). Checkpoints/ONNX live on the Modal volume `polymorph-lamr-v0:/out/mb_v0` and download to `data/modal_out/mb_v0/` (gitignored).

See [`blog.md`](blog.md) for the full narrative, challenges, and mistakes; [`TODOS.md`](TODOS.md) for remaining work.

---

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `POLYMORPH_DB_PATH` | `~/.polymorph/cache.db` | SQLite database location |
| `POLYMORPH_GRAMMARS_DIR` | walks up from binary, falls back to `./grammars` | Tree-sitter WASM grammars |
| `ANTHROPIC_API_KEY` | — | required only to run the LLM-judge benchmark |

## Development

```bash
cargo test                                  # Rust runtime
cd ml_pipeline && .venv/bin/python -m pytest # 269+ Python tests
```

License: Apache-2.0 (planned).
