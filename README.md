# Polymorph

Polymorph is a local-first compression engine for logs and traces, served over the Model Context Protocol (MCP). You give it a log; it returns a smaller log that still contains the answer. Every structural field, error code, IP, UUID, and trace ID is preserved byte-for-byte; the redundant operational prose is pruned by a trained token classifier.

It exists for one job: feed large logs and traces to an LLM (incident triage, root-cause analysis, trace debugging) without burning the context window on repetitive chatter or losing the one line that matters.

## How it works

Two layers do the work. The first is deterministic and lossless. It tokenizes the log, then force-keeps every structural atom by intersecting three sources: a regex/Aho-Corasick scan for error codes, severities, IPs, UUIDs, and trace IDs; a Tree-sitter parse of any embedded JSON; and any extra keywords you pass. It then collapses repeated template lines (run-length dedup) and stashes the middle of long JSON arrays in a local SQLite cache. Nothing is deleted irreversibly: the original is cached and retrievable by id, so a heavily pruned payload can always be expanded back.

The second layer is a learned pruner, LaMR. A 150M-parameter ModernBERT token classifier scores a drop probability for each token in the *unlocked* prose residual, and a span-aware decoder drops whole low-salience words to hit a target compression rate. It runs inside the Rust server with no Python and no native dependencies: a pure-Rust ModernBERT byte-level tokenizer (byte-exact with the Hugging Face tokenizer) feeds a windowed `tract` ONNX forward pass. The Rust runtime reproduces the PyTorch drop probabilities to `max_abs_diff = 3e-6`. Because the deterministic layer bounds the residual, the model never sees the whole stream — a 10 MB repetitive log is collapsed to a few kilobytes before the model runs.

## Getting started

Requirements: a recent Rust toolchain. The deterministic layer needs nothing else; the learned pruner needs the trained model file (see below).

```bash
git clone https://github.com/GaganSD/lulu-polymorph.git
cd lulu-polymorph
cargo build --release
```

Register the server with an MCP client (`~/.config/claude-code/settings.json` or a project `.claude/settings.json`). Set `POLYMORPH_LAMR_MODEL` to enable the learned pruner; omit it to run the deterministic layer alone.

```json
{
  "mcpServers": {
    "polymorph": {
      "command": "/absolute/path/to/lulu-polymorph/target/release/polymorph-mcp",
      "env": {
        "POLYMORPH_DB_PATH": "~/.polymorph/cache.db",
        "POLYMORPH_GRAMMARS_DIR": "/absolute/path/to/lulu-polymorph/grammars",
        "POLYMORPH_LAMR_MODEL": "/absolute/path/to/lulu-polymorph/data/modal_out/mb_v0/onnx/model.onnx"
      }
    }
  }
}
```

Compress a log with the `compress_log` tool. Pass a file `path` for large logs (read on the server, no paste limit) or inline `text`:

```jsonc
{ "path": "/var/log/app/incident.log" }
// → { compressed, cache_id, input_tokens, output_tokens, ratio, dedup_elided_lines, used_model }
```

A bundled Claude Code skill in `.claude/skills/polymorph/` invokes this automatically. Ask *"here are the production traces, debug this, use polymorph"* and it routes the log through `compress_log` before analysis. Copy that folder to `~/.claude/skills/polymorph/` to use it in any project.

Build `--release` — debug builds run inference about 15× slower. In release, the 600 MB model loads in ~2.4 s and scores a multi-window document in ~2.8 s.

## Results

`mb_v0` (ModernBERT-150M) on the 187 LogHub-2.0 prose triples across 7 domains, at matched compression ratio, LLM-judged answer survival. 95% bootstrap confidence intervals; paired McNemar test against the keep-severity baseline.

| method | survival @3× (95% CI) | survival @5× (95% CI) |
|---|---|---|
| keep-severity (line baseline) | 17% [12–23] | 14% [9–19] |
| **Polymorph LaMR `lamr+span`** | **62% [55–69]** | **44% [36–51]** |
| `lamr+span+floor` | 51% [44–58] | 37% [31–45] |

`lamr+span` preserves about 3.6× as many answers as the baseline, and the gap is significant: McNemar wins 92–9 discordant pairs at 3× (p ≈ 0) and 66–10 at 5× (p ≈ 0). Judge-free exact-match survival is 56% / 37%, still ~3.5× the baseline. Per-domain at 3×: BGL 88%, Linux 68%, ZooKeeper 57%, Hadoop 52%, Spark 50%, OpenStack 25%. Checkpoint quality on the held-out shard: PR-AUC 0.873, ROC-AUC 0.933.

The eval runs on Modal GPU (`ml_pipeline/cloud/eval_modal.py`); the defensible-eval stats (McNemar, bootstrap CIs, per-domain breakdown) are computed in Rust (`src/stats.rs`, exposed as `polymorph-mcp --bench-stats`).

## MCP tools

| tool | purpose |
|---|---|
| `compress_log` | log/trace text or file path → smaller log + `cache_id` |
| `polymorph_retrieve_cache` | fetch the original (or an elided slice) by `cache_id` |
| `lock_mask` | per-token lock/drop mask for a block of text |
| `compress_array` | head/tail-keep a large JSON array, cache the middle |
| `lcm_append` / `lcm_describe` / `lcm_expand` | append-only timeline with summary nodes |

## Configuration

| env var | default | meaning |
|---|---|---|
| `POLYMORPH_LAMR_MODEL` | unset | path to the ONNX pruner; unset runs the deterministic layer only |
| `POLYMORPH_DB_PATH` | `~/.polymorph/cache.db` | SQLite cache location |
| `POLYMORPH_GRAMMARS_DIR` | walks up from the binary | Tree-sitter WASM grammars |

The model file is gitignored. Pull it with `modal volume get polymorph-lamr-v0 /out/mb_v0/onnx data/modal_out/mb_v0/onnx`. An INT8 build (`ml_pipeline/scripts/quantize_int8.py`) produces a 154 MB artifact that loads in ~1 s with near-identical decisions (top-k drop Jaccard 0.994).

## Development

```bash
cargo test                                    # Rust runtime + bench/eval/distill ports
cd ml_pipeline && .venv/bin/python -m pytest  # Python training/export pipeline
```

The offline benchmark, eval, and distillation pure-logic (answer-survival triple mining, compression baselines, McNemar/bootstrap stats, label-ceiling gate, ranking metrics, training-template dedup) is implemented in Rust and driven by binary subcommands — `polymorph-mcp --bench-survival | --bench-stats | --build-triples | --build-loghub | --label-ceiling | --eval-metrics | --sampler-filter`. Model training, ONNX export, and the LLM-judge eval stay in Python and run on Modal — see `ml_pipeline/cloud/`. There is no local training. [`TODOS.md`](TODOS.md) tracks open work; [`blog.md`](blog.md) has the build narrative.

## License

MIT — see [`LICENSE`](LICENSE).
