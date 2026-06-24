---
name: polymorph
description: >-
  Compress logs and production traces before analyzing them, using the Polymorph
  MCP server's `compress_log` tool. Use this whenever the user hands you a large
  log/trace file or pasted log block to debug, investigate, or root-cause — and
  ALWAYS when they say "use polymorph" (e.g. "here are the production traces,
  debug this, use polymorph"). Triggers: "use polymorph", "compress these logs",
  debugging/triaging production logs or traces, a multi-thousand-line log file, or
  any time a raw log would otherwise fill the context window. Polymorph keeps every
  error code, status, IP, UUID and trace id byte-for-byte and prunes only redundant
  operational chatter, so you analyze a smaller payload without losing the needle.
---

# Polymorph — compress logs/traces before you analyze them

Polymorph is a local MCP server that turns a big log into a smaller log that still
contains the answer. It locks structural atoms (error codes, severities, IPs,
UUIDs, trace/span ids), collapses repeated template lines, and runs a trained
token-classifier (LaMR) that drops redundant prose to a target rate. Every drop is
reversible from a local cache.

## When to use it

Use Polymorph **before** reading a log into context whenever:

- The user explicitly says "use polymorph" / "compress these logs".
- They give you a log/trace **file path** or a large pasted log to debug,
  triage, or root-cause.
- A raw log would consume a large fraction of the context window.

Don't use it for small snippets (a few lines), source code, or prose documents —
it's specialized for operational logs and traces.

## How to use it

1. **Call the `compress_log` tool.**
   - For a **file** (preferred for anything large): pass `path` (an absolute or
     `~/`-relative local path). The server reads it directly — don't paste a 10 MB
     file into the chat.
     ```json
     { "path": "/var/log/app/incident-2026-06-08.log" }
     ```
   - For a **pasted block**: pass `text`.
     ```json
     { "text": "<the log block>" }
     ```
   - Optional args:
     - `keywords`: extra substrings that must never be dropped (API keys, a
       specific resource id, a request id you're tracing). These are force-locked.
     - `target_rate`: fraction of unlocked tokens to drop, `0`–`1` (omit for the
       model's default, ~0.3). Raise it for more aggressive compression.
     - `language`: `"json"` if the payload is a single JSON document, `"python"`
       for a Python file; otherwise omit (defaults to plain-text, which is correct
       for raw logs — forcing a grammar on non-matching text over-locks and hurts).

2. **Read the result.** The tool returns:
   - `compressed` — the smaller log. **Do your analysis on this.**
   - `cache_id` — handle to the full original.
   - `input_tokens` / `output_tokens` / `ratio` — how much was saved.
   - `dedup_elided_lines` — repeated lines collapsed.
   - `used_model` — `true` if the neural pruner ran, `false` if only the
     deterministic layer did (model not configured; see Setup).

3. **Expand when you need a dropped detail.** If the compressed view elides
   something you need (a specific repeated line, the middle of a long run), call
   `polymorph_retrieve_cache` with the `cache_id` to get the original back
   verbatim. Nothing forensic is lost — it's cached, not deleted.

4. **Report** the finding plus a one-line note on the compression (e.g. "analyzed
   the 42k-token trace compressed to 9k tokens, 1.5×; root cause: …").

## Example

> User: "here are the production traces, debug this. use polymorph:
> /tmp/trace-dump.log"

- Call `compress_log` with `{ "path": "/tmp/trace-dump.log" }`.
- Inspect `compressed`; find the error/state-transition that matters.
- If you need a specific elided span, `polymorph_retrieve_cache` with the
  `cache_id`.
- Answer from the compressed payload, citing the surviving error code/trace id.

## Setup (one-time)

- The Polymorph MCP server must be registered with your client (see the repo
  README Claude Code or Cursor setup). Build it with `cargo build --release`.
- Verify the local install before using the skill:
  `./target/release/polymorph-mcp --selftest` and
  `./target/release/polymorph-mcp --demo compress`.
- For the **neural** pruner (best compression), install the model with
  `bash scripts/fetch_model.sh` or point the server at an existing artifact via
  `POLYMORPH_LAMR_MODEL=/path/to/mb_v0/onnx/model.onnx`. Without it,
  `compress_log` still runs deterministic dedup + locking (`used_model:false`).
- To use this skill outside this repo, copy `.claude/skills/polymorph/` to
  `~/.claude/skills/polymorph/`.
