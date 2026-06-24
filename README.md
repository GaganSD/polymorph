# Polymorph

Polymorph is a local-first MCP server that compresses logs and traces before an
LLM reads them. You give it a log; it returns a smaller log plus a `cache_id` for
retrieving the original. Error codes, severities, IPs, UUIDs, trace IDs, and
structured fields are force-kept.

The first-run path works without a model. That deterministic mode gives you
template deduplication, structural locking, SQLite-backed retrieval, and MCP
integration. The optional LaMR model adds learned prose pruning once
`POLYMORPH_LAMR_MODEL` points at the ONNX artifact.

## Requirements

- Rust stable and Cargo.
- macOS or Linux for the documented source install path.
- Claude Code or Cursor if you want to call the MCP tools from an agent.
- Optional: about 600 MB of disk for the `mb_v0` ONNX model.

Always build the release binary. Debug builds are useful for tests, but model
inference is about 15x slower.

## Quick Start

```bash
git clone https://github.com/GaganSD/lulu-polymorph.git
cd lulu-polymorph
cargo build --release
./target/release/polymorph-mcp --selftest
./target/release/polymorph-mcp --demo compress
```

Expected shape:

```text
grammars: ok (.../grammars)
db: ok (.../.polymorph/cache.db)
model: unset (deterministic mode; compress_log returns used_model=false)
PASS: install checks + 3 locking scenarios
```

The compression demo reads `examples/sample.log`, dedups repeated heartbeat
lines, preserves `DiskControllerFirmwareDeadlock`, and prints token counts,
ratio, `dedup_elided_lines`, and `used_model`.

## Optional Model Setup

The learned pruner is disabled until you install the ONNX model. Published
releases use a GitHub Release asset named like
`polymorph-mb_v0-onnx.tar.gz`.

```bash
bash scripts/fetch_model.sh
```

The script installs the model under `data/modal_out/mb_v0/onnx/` and prints the
`POLYMORPH_LAMR_MODEL` value to copy into your MCP config.

If you are testing before the public release asset is uploaded, pass your own
artifact URL:

```bash
POLYMORPH_MODEL_URL="https://example.com/polymorph-mb_v0-onnx.tar.gz" \
POLYMORPH_MODEL_SHA256="<sha256>" \
bash scripts/fetch_model.sh
```

The archive must contain `model.onnx`. It may also contain `model.onnx.data` and
`decode.json`; keep those files next to `model.onnx`.

## Claude Code Setup

Add the `mcpServers.polymorph` block below to your Claude Code settings file.
Replace `/absolute/path/to/lulu-polymorph` with this checkout. A generic copy is
also available in `mcp.example.json`.

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

To enable the model, add:

```json
"POLYMORPH_LAMR_MODEL": "/absolute/path/to/lulu-polymorph/data/modal_out/mb_v0/onnx/model.onnx"
```

A Claude Code skill is bundled at `skills/polymorph/`. To use it in any project,
copy that folder to `~/.claude/skills/polymorph/`. Then ask:

```text
Here are production traces in /tmp/incident.log. Debug this, use polymorph.
```

The skill routes the log through `compress_log` before analysis.

## Cursor Setup

Merge the same `mcpServers.polymorph` block into your Cursor MCP config. You can
start from `mcp.example.json`, replacing the absolute paths with your checkout
path.

Use the same optional `POLYMORPH_LAMR_MODEL` env var as Claude Code when the
model is installed.

Restart Cursor after changing MCP config. Then ask Cursor to call the
`compress_log` tool on `examples/sample.log`, or paste a small log inline.

## First Tool Call

Use `compress_log` with a file path for large logs:

```json
{
  "path": "/absolute/path/to/lulu-polymorph/examples/sample.log"
}
```

Or pass inline text:

```json
{
  "text": "INFO heartbeat ok\nINFO heartbeat ok\nERROR DiskControllerFirmwareDeadlock code=E513\n"
}
```

The response shape is:

```json
{
  "compressed": "...",
  "cache_id": "...",
  "input_tokens": 123,
  "output_tokens": 45,
  "ratio": 2.73,
  "dedup_elided_lines": 8,
  "used_model": false
}
```

`used_model: false` is normal in deterministic mode. It means no ONNX model was
loaded; dedup, locking, and cache retrieval still ran. After the model is
configured correctly, the same field should become `true` for compressible input.

## Configuration

| env var | default | meaning |
|---|---|---|
| `POLYMORPH_DB_PATH` | `~/.polymorph/cache.db` | SQLite cache for originals and LCM state |
| `POLYMORPH_GRAMMARS_DIR` | auto-discovers `grammars/` near the repo binary | Tree-sitter WASM grammars for JSON/Python locking |
| `POLYMORPH_LAMR_MODEL` | unset | Optional ONNX model path for learned pruning |

Leading `~/` is expanded in all three env vars.

## How It Works

Polymorph has two layers.

The deterministic layer is always on. It tokenizes the log, force-keeps
structural atoms with Tree-sitter WASM grammars plus an Aho-Corasick keyword
scanner, collapses repeated template lines, and stores originals in SQLite for
retrieval. This layer has no Python dependency.

The optional LaMR layer is a ModernBERT-150M token classifier exported to ONNX.
The Rust runtime uses a bundled ModernBERT tokenizer and `tract` for pure-Rust
ONNX inference, so there is no native ONNX Runtime install. The model scores
only the post-dedup residual and drops whole low-salience word spans to hit the
target rate.

## MCP Tools

| tool | purpose |
|---|---|
| `compress_log` | log/trace text or file path to smaller log plus `cache_id` |
| `polymorph_retrieve_cache` | fetch the original cached value by `cache_id` |
| `lock_mask` | inspect structural locks and the mock/model drop mask |
| `compress_array` | keep head/tail of a large JSON array and cache the middle |
| `lcm_append` | append a long conversation/log timeline turn |
| `lcm_describe` | inspect an archived LCM node |
| `lcm_expand` | retrieve verbatim archived turns |

## Troubleshooting

- `used_model` stays `false`: run `./target/release/polymorph-mcp --selftest`.
  If `POLYMORPH_LAMR_MODEL` is missing or empty, fix the path or omit the env var
  for deterministic mode.
- Grammar errors: set `POLYMORPH_GRAMMARS_DIR` to the absolute path of this
  repo's `grammars/` directory.
- Slow inference: rebuild with `cargo build --release`.
- Startup SQLite errors: make sure `POLYMORPH_DB_PATH` points to a writable
  location. The default creates `~/.polymorph/cache.db`.
- Large inline logs: use `{"path": "/path/to/log"}` instead of pasting. Inline
  text is capped lower than file input.

## Results

`mb_v0` (ModernBERT-150M) on the 187 LogHub-2.0 prose triples across 7 domains,
at matched compression ratio, LLM-judged answer survival. 95% bootstrap
confidence intervals; paired McNemar test against the keep-severity baseline.

| method | survival @3x (95% CI) | survival @5x (95% CI) |
|---|---|---|
| keep-severity (line baseline) | 17% [12-23] | 14% [9-19] |
| **Polymorph LaMR `lamr+span`** | **62% [55-69]** | **44% [36-51]** |
| `lamr+span+floor` | 51% [44-58] | 37% [31-45] |

`lamr+span` preserves about 3.6x as many answers as the baseline. McNemar wins
92-9 discordant pairs at 3x and 66-10 at 5x. Judge-free exact-match survival is
56% / 37%. Per-domain at 3x: BGL 88%, Linux 68%, ZooKeeper 57%, Hadoop 52%,
Spark 50%, OpenStack 25%.

The eval runs on Modal GPU (`ml_pipeline/cloud/eval_modal.py`). Defensible stats
are computed in Rust via `polymorph-mcp --bench-stats`.

## Development

```bash
cargo fmt --check
cargo test
cargo build --release
./target/release/polymorph-mcp --selftest
cd ml_pipeline && .venv/bin/python -m pytest
```

The offline benchmark/eval pure logic is implemented in Rust and exposed through
binary subcommands:

```text
--bench-survival
--bench-stats
--build-triples
--build-loghub
--label-ceiling
--eval-metrics
--sampler-filter
```

Model training, ONNX export, and the LLM-judge eval stay in Python under
`ml_pipeline/`.

## License

MIT. See [LICENSE](LICENSE).
