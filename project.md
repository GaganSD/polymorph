# Project: Polymorph

## One-Line Description

**Self-hostable, open-sourced, low-latency prompt compression MCP for LLM pipelines that improves LLM accuracy and reduces token costs by removing low-signal tokens from input context, strictly optimized for coding agents and structured data.**

First Use case: Claude Code on terminal.
---

## Elevator Pitch

Coding agents drown in massive JSON tool outputs, deep file listings, verbose Abstract Syntax Trees (ASTs), and redundant server logs. Every LLM call ships this bloated context, burning money and latency. Polymorph is a lightweight middleware—deployable as a Model Context Protocol (MCP) server—that sits between your agent and the model provider.

Unlike standard entropy-based compressors that destroy syntax by deleting critical brackets and indentations, Polymorph uses a deterministic "Token Locking" pipeline to physically shield structural boundaries (like ASTs and JSON arrays). Once the structure is mathematically locked, our linear-time ML classifier prunes the remaining unlocked boilerplate (docstrings, redundant logs, and conversational filler).

The result: 10–40% smaller prompts, pristine executable code, deterministic outputs, and higher downstream reasoning accuracy because the model isn't distracted by noise.

---

## Mission Statement

**Make every token count—without breaking the syntax.** We exist to remove the structural waste from how autonomous agents feed context to large language models, ensuring that intelligence becomes cheaper, faster, and syntactically flawless.

---

## Problem Statement

Agent-powered coding workflows burn money and latency on input bloat, but existing compression solutions break the fundamental requirement of code: **it must compile.**

1. **Standard prompt compression destroys structured data.** Entropy-based token droppers frequently misidentify structural anchors (like closing braces `}`, colons `:`, or strict indentation blocks) as low-value tokens, irreversibly corrupting the underlying syntax.
2. **Context windows are noisier than they look.** Large payloads like 1,000-item search results or database dumps exhaust the LLM's Key-Value (KV) cache.
3. **Agent token economics are broken.** Coding agents re-send massive overlapping contexts on every turn. At 1M+ tokens per session, waste compounds rapidly.
4. **Latency taxes the agent loop.** Time-to-first-token scales with input length. Waiting on massive context pre-fills slows down autonomous workflows.
5. **No open standard exists.** Commercial products (e.g., The Token Company) focus on natural language RAG and are closed-source black boxes. They cannot be trusted locally with proprietary, highly sensitive codebases.

---

## Why Now

* **The explosion of Autonomous Agents:** Open-source frameworks like OpenClaw have emerged as fully autonomous personal AI assistants that run locally and connect to extensive external tools via plugins. Concurrently, Claude Code connects to local tools, databases, and APIs via the Model Context Protocol (MCP) to automate multi-step development workflows.
* **The Model Context Protocol (MCP) is the new standard:** By exposing tools via MCP, Claude Code and OpenClaw create a standardized middleware layer. Polymorph can instantly plug into this ecosystem as a native MCP server.
* **Transformer Quadratic Scaling hit a wall:** Generative attention mechanisms scale quadratically $O(N^2)$, causing severe memory bottlenecks. The recent maturation of linear-time State Space Models allows us to process 100k+ tokens efficiently.
* **High-Speed Parsing is now viable:** The integration of C-based Generalized LR (GLR) parsers and Double-Array Aho-Corasick (DAAC) automata allows us to evaluate and map syntax structures in fractional milliseconds, making inline context compression for code a reality.

---

## Target User / ICP

**Primary (Open-Source Phase):**

* **AI engineers building coding agents:** Users of Claude Code, Cursor, Cline, and OpenClaw who need to extend their token budgets without sacrificing code executability.
* **Platform Engineers:** Developers building internal, tool-heavy MCP servers (e.g., GitHub, PostgreSQL, or Filesystem MCPs) who need to compress massive tool outputs before they hit the agent.

**Secondary (SaaS Phase):**

* **AI-native startups burning >$10k/mo on inference:** Companies where token spend is a top-3 line item due to heavy automated code generation.
* **Enterprise platform teams:** Central AI groups running local agent swarms that require self-hostable, zero-data-retention optimizations.

**Anti-ICP:** General text summarization, prose-heavy document QA, creative writing assistants, and conversational chatbots that do not utilize structured data or tools.

---

## Core User Journey

**Claude Code / OpenClaw Developer:**

1. Developer installs Polymorph locally via standard package managers.
2. Developer connects Polymorph as an MCP server within their configuration (e.g., in `.claude/settings.json` or OpenClaw's workspace).
3. Every file read, tool result, or API output is automatically piped through Polymorph's token-locking pipeline.
4. Polymorph dynamically maps the AST, locks the syntax, and uses its ML classifier to drop boilerplate comments and redundant logs before returning the lean payload to the agent.
5. The agent executes faster, and the user’s token budget extends 1.3–1.6×.

**Enterprise Platform Team:**

1. Deploys the Polymorph Rust binary locally to intercept all traffic hitting their internal LLM Gateway.
2. Configures custom DAAC dictionaries to strictly lock proprietary internal framework keywords and API signatures from ever being compressed.

---

## Main Offering

A three-layer product strictly tailored for code and structured tool outputs:

1. **Open-source Core (Apache 2.0):** A native Rust binary leveraging the `mamba-rs` framework for ML inference, complete with the DAAC and Tree-sitter token-locking pipelines.
2. **MCP Server Integration:** A fully compliant Model Context Protocol server that plugs directly into Claude Code, OpenClaw, and the broader agent ecosystem.
3. **Hosted API (SaaS):** A managed endpoint with autoscaling, observability, and custom DAAC configuration profiles. Priced strictly on tokens saved.

---

## Core Features

* **Zero-Latency AST Token Locking:** Uses Tree-sitter to incrementally parse code files, map the exact byte intervals of structural nodes, and enforce a Boolean mask that protects critical syntax from being dropped.
* **DAAC Lexical Scanning:** Employs a Double-Array Aho-Corasick automaton to scan 100k token arrays in $O(N)$ time (tens of microseconds) to identify and lock precise API keys and hard constraints.
* **Byte-to-Token Mapping:** Integrates Tiktoken's `decode_with_offsets` to translate physical character byte intervals perfectly into LLM token boundaries, solving UTF-8 misalignment.
* **Compress-Cache-Retrieve (CCR) for JSON:** Dynamically identifies massive JSON arrays, retains the first/last elements to preserve array boundaries, filters the middle using BM25, and caches the rest locally. The agent is provided a tool to retrieve the uncompressed cache instantly if needed.
* **Bidirectional Mamba-3 MIMO Engine:** Abandons standard Transformer encoders for a highly efficient State Space Model running in Rust. Ensures $O(N)$ linear scaling with zero Python Global Interpreter Lock (GIL) latency overhead.
* **Zero Data Retention Mode:** Fully self-hostable. No logging of proprietary codebase payloads.

---

## Product Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│  Agent / MCP Client (Claude Code / OpenClaw / Cursor)        │
└─────────────────────────┬────────────────────────────────────┘
                          ▼ (stdio / SSE / HTTP)
┌──────────────────────────────────────────────────────────────┐
│ Polymorph MCP Server (Rust Native)                           │
│                                                              │
│ 1. Tiktoken BPE Mapping (Extracts Byte Offsets)              │
│ 2. Parallel Scanning:                                        │
│    ├── Tree-sitter (AST Byte Interval Extraction)            │
│    └── DAAC Automaton (O(N) Lexical String Matching)         │
│ 3. Sweep-Line Intersection (Generates Boolean Lock Mask)     │
│ 4. Mamba-rs Engine (Evaluates Unlocked Boilerplate)          │
│ 5. Compress-Cache-Retrieve (Caches dropped JSON chunks)      │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────── ┐
│   Forward Compressed Payload to LLM Gateway / Provider       │
└──────────────────────────────────────────────────────────────┘

```

---

## Research Thesis

We believe three things, and we intend to prove them in public:

1. **Syntax is Sacred, Prose is Malleable.** Machine learning should never guess if a bracket is important. Deterministic parsing (AST/Regex) must handle structure; probabilistic ML classifiers should only handle the remaining natural language boilerplate.
2. **Transformers cannot scale token classification.** To process 100k+ tokens at sub-millisecond latencies without exhausting GPU VRAM, the industry must move away from quadratic attention mechanisms and embrace linear-complexity State Space Models (Mamba-3).
3. **Structured data requires reversible compression.** Agents operating on massive tool outputs (like 1,000 DB rows) need dynamic budgeting via the Compress-Cache-Retrieve architecture, ensuring heavy pruning is always backed by a local safety cache.

---

## Product Philosophy

1. **The agent is a first-class user.** Polymorph is built ground-up for autonomous workflows. MCP and tool-call ergonomics are the primary integration layer.
2. **Code must compile.** We compress by deletion, but we never delete structural anchors. Same input logic → same output logic.
3. **Rust native for zero overhead.** Relying on Python for high-frequency token routing is a mistake. By utilizing `mamba-rs` and batch-invariant `bf16` inference, we guarantee extreme speed and bit-parity.
5. **Small surface, deep quality.** We do not summarize documents. We do not generate code. We do exactly one thing: flawless, high-speed structural context compression.