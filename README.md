# Polymorph

> Lossless prompt-compression MCP for coding agents. Token-locking + mock LaMR pruning + JSON-array CCR + SQLite-backed conversation archive.

Polymorph sits between your coding agent and the model provider as an MCP server. It produces a per-token lock mask (syntax + keywords stay; prose can be pruned), compresses large JSON tool outputs into a head-summary-tail shape with retrievable cache, and archives old conversation turns into a Depth-0 DAG node when the active context crosses a soft token threshold. Code stays compilable; agents recover anything they archived.

## Quick Start (~2 minutes)

### 1. Build

```bash
git clone https://github.com/GaganSD/lulu-polymorph.git
cd lulu-polymorph
cargo build --release
```

### 2. See it work (no MCP client needed)

```bash
# Mock conversation loop with auto-archive trigger:
./target/release/polymorph-mcp --demo lcm-loop

# JSON array compress + retrieve round-trip:
./target/release/polymorph-mcp --demo ccr
```

You'll see the archive trigger fire mid-loop with the new `node_id`, and the CCR demo recovers all 44 omitted items from the cache.

### 3. Register with Claude Code

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

Restart Claude Code. The agent now has six tools available: `lock_mask`, `compress_array`, `polymorph_retrieve_cache`, `lcm_append`, `lcm_describe`, `lcm_expand`.

## Tools

| Tool | What it does |
|---|---|
| `lock_mask` | Tokenize a JSON or Python payload, return a per-token lock mask (`true` = structural/keyword, never drop) and a mock-LaMR drop mask (`true` = drop, only on unlocked tokens). |
| `compress_array` | Take a long JSON array, keep first 3 + last 3 elements, persist the middle into SQLite, inject a summary placeholder with a `cache_id`. |
| `polymorph_retrieve_cache` | Recover the original middle slice for a `cache_id`. |
| `lcm_append` | Add a conversational turn. When the conversation's active token count exceeds `soft_threshold` (default 80,000), oldest turns are archived into a Depth-0 summary node. Returns the new `archived_node_id` if archive fired. |
| `lcm_describe` | Return metadata about a summary node (`child_count`, `total_tokens`, `roles`, `created_at`). |
| `lcm_expand` | Return the original verbatim turns archived under a summary node, in order. |

## Architecture

```
agent ── stdio JSON-RPC ──> polymorph-mcp
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
- **M3**: Real Gated DeltaNet-2 model swap + Depth-1/-2 DAG summarization + automatic CCR inside `lock_mask`.

License: Apache-2.0 (planned).
