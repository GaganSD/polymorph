# Polymorph

> State-of-the-art compression for audit logs and production traces. Token-locking + LaMR pruning + JSON-array CCR + SQLite-backed trace archive, served over MCP.

Polymorph sits between your log/trace store and the model provider as an MCP server. It produces a per-token lock mask (structural fields, severity, error codes, and trace IDs stay; redundant operational prose can be pruned), compresses large JSON event arrays into a head-summary-tail shape with retrievable cache, and archives old trace spans into a Depth-0 DAG node when the active window crosses a soft token threshold. Logs stay parseable; the LLM (or you) can recover anything that was archived.

## Quick Start (~2 minutes)

### 1. Build

```bash
git clone https://github.com/GaganSD/lulu-polymorph.git
cd lulu-polymorph
cargo build --release
```

### 2. See it work (no MCP client needed)

```bash
# Mock trace/log timeline with auto-archive trigger:
./target/release/polymorph-mcp --demo lcm-loop

# JSON event-array compress + retrieve round-trip:
./target/release/polymorph-mcp --demo ccr
```

You'll see the archive trigger fire mid-loop with the new `node_id`, and the CCR demo recovers all 44 omitted items from the cache.

### 3. Register with an MCP client

Add to `~/.config/claude-code/settings.json` (or `.claude/settings.json` in your project):

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

Restart the client. The agent now has six tools available: `lock_mask`, `compress_array`, `polymorph_retrieve_cache`, `lcm_append`, `lcm_describe`, `lcm_expand`.

## Tools

| Tool | What it does |
|---|---|
| `lock_mask` | Tokenize a JSON or log/trace payload, return a per-token lock mask (`true` = structural field / error code / keyword, never drop) and a mock-LaMR drop mask (`true` = drop, only on unlocked tokens). |
| `compress_array` | Take a long JSON array (e.g., a list of log events or trace spans), keep first 3 + last 3 elements, persist the middle into SQLite, inject a summary placeholder with a `cache_id`. |
| `polymorph_retrieve_cache` | Recover the original middle slice for a `cache_id`. |
| `lcm_append` | Add a trace/log entry. When the timeline's active token count exceeds `soft_threshold` (default 80,000), oldest entries are archived into a Depth-0 summary node. Returns the new `archived_node_id` if archive fired. |
| `lcm_describe` | Return metadata about a summary node (`child_count`, `total_tokens`, `roles`, `created_at`). |
| `lcm_expand` | Return the original verbatim entries archived under a summary node, in order. |

## Architecture

```
client ── stdio JSON-RPC ──> polymorph-mcp
                              │
       ┌──────────────────────┼─────────────────────┐
       ▼                      ▼                     ▼
   tiktoken-rs           daachorse              wasmtime
   (byte-spans)          (DAAC keyword)         (Tree-sitter WASM)
       │                      │                     │
       └──────── O(N) two-pointer sweep ────────────┘
                              │
                              ▼
                        Vec<bool> lock_mask
                              │
                              ▼
                  mock LaMR (ChaCha8, 30% drop)
                              │
                              ▼
                        Vec<bool> drop_mask

                  ┌─────────────────────────┐
                  │  rusqlite (WAL)         │
                  │  ccr_cache              │
                  │  lcm_messages           │
                  │  lcm_summary_nodes      │
                  └─────────────────────────┘
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `POLYMORPH_DB_PATH` | `~/.polymorph/cache.db` | SQLite database location |
| `POLYMORPH_GRAMMARS_DIR` | walks up from binary, falls back to `./grammars` | Where to find `tree-sitter-{json,python}.wasm` |

## Development

```bash
cargo test                # 87 tests
cargo run -- --selftest   # end-to-end M1 selftest
cargo run -- --demo lcm-loop
cargo run -- --demo ccr
```

## Status

- **M1**: Token-locking pipeline (tiktoken byte-spans + DAAC + WASM Tree-sitter + sweep). Done.
- **M2**: CCR cache + LCM archive + mock LaMR. Done.
- **M3**: Deterministic pattern/template dedup pre-stage + a reproducible compression benchmark, then an **evidence-gated** ML pruner (leading candidate: a compact bidirectional dual-CRF encoder over the prose residual, not a causal SSM — built only if the benchmark shows it's worth it) + Depth-1/-2 DAG summarization + automatic CCR inside `lock_mask`.

License: Apache-2.0 (planned).
