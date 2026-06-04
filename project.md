# Project: Polymorph

## One-Line Description

**Self-hostable, open-source, low-latency compression engine for audit logs and production traces that improves LLM accuracy and reduces token costs by removing redundant operational noise from log streams—while keeping every structural field, error code, and state transition perfectly intact.**

First use case: feeding audit logs and distributed traces into an LLM for incident analysis.

---

## Elevator Pitch

Observability and security pipelines generate oceans of audit logs and production traces—structured event fields wrapped around free-text messages, nested JSON payloads, and multi-frame stack traces. The moment a team points an LLM at this data (incident triage, root-cause analysis, compliance review, anomaly detection), every call ships megabytes of repetitive operational chatter, burning money and latency and drowning the one line that matters.

Polymorph is a lightweight middleware—deployable as a Model Context Protocol (MCP) server—that sits between your log/trace store and the model provider.

Unlike standard entropy-based compressors that destroy structure by deleting critical braces, field separators, and trace IDs, Polymorph uses a deterministic "Token Locking" pipeline to physically shield structural boundaries (embedded JSON payloads, severity levels, trace/span IDs, error codes). On logs, the bulk of the win comes from deterministic structure exploitation: locking + template/pattern dedup + Compress-Cache-Retrieve. An optional, evidence-gated ML classifier then prunes the small remaining unlocked prose residual (repeated heartbeat lines, verbose-but-uninformative INFO chatter, duplicated stack frames).

The result: 10–40% smaller log payloads—far higher on highly redundant streams—every structural field and state transition preserved byte-for-byte, deterministic outputs, and higher downstream reasoning accuracy because the model isn't distracted by noise.

---

## Mission Statement

**Make every token count—without losing the audit trail.** We exist to remove the operational waste from how teams and agents feed logs and traces to large language models, ensuring that observability intelligence becomes cheaper, faster, and forensically complete.

---

## Problem Statement

LLM-driven log and trace analysis burns money and latency on input bloat, but existing compression solutions break the fundamental requirement of audit data: **it must stay parseable, and nothing forensically relevant can be silently dropped.**

1. **Standard compression destroys structured log data.** Entropy-based token droppers frequently misidentify structural anchors (closing braces `}`, field separators `:`, severity tokens, trace IDs) as low-value tokens, irreversibly corrupting the records and breaking any downstream parse or query.
2. **Log and trace volume is mostly noise—except where it isn't.** Health checks, heartbeats, and repeated INFO lines dominate the byte count, while the rare error, exception, or state transition is exactly what an investigation needs. Naive truncation and sampling lose precisely those lines.
3. **Token economics are broken.** Incident copilots and trace-analysis agents re-send massive overlapping windows of logs on every turn. At 1M+ tokens per investigation, waste compounds rapidly.
4. **Latency taxes the analysis loop.** Time-to-first-token scales with input length. Waiting on a multi-megabyte trace dump pre-fill slows down every query.
5. **No open standard exists.** Commercial products focus on natural-language RAG and are closed-source black boxes. They cannot be trusted locally with proprietary, sensitive audit logs—which is exactly the data that must never leak.

---

## Why Now

* **LLM-driven observability has arrived:** SRE incident copilots, SIEM-plus-LLM workflows, and agents that read distributed traces to debug failures are now mainstream. They all need to stuff logs into a context window.
* **The Model Context Protocol (MCP) is the new standard:** By exposing tools via MCP, Polymorph plugs directly into the agent and incident-response ecosystem as a native MCP server—or sits inline in the log-ingestion path.
* **The neural stage never sees the whole stream.** Deterministic locking + template dedup + chunking reduce the ML workload to small bounded windows of residual prose, so 100k+ token streams are handled by structure, not by a giant model. That makes a compact, mature, well-deployable encoder viable instead of requiring an exotic long-context backbone.
* **High-Speed Parsing is now viable:** C-based Generalized LR (GLR) parsers and Double-Array Aho-Corasick (DAAC) automata evaluate and map structure in fractional milliseconds, making inline compression of high-volume log streams a reality.

---

## Target User / ICP

**Primary (Open-Source Phase):**

* **SRE / platform / observability engineers:** Teams piping audit logs and distributed traces into LLMs for incident triage, root-cause analysis, and postmortems who need to extend their token budgets without losing forensic detail.
* **Security & compliance teams:** Engineers running LLM-assisted log analysis, SIEM enrichment, and audit-trail review who cannot ship raw, sensitive logs to a closed third party.

**Secondary (SaaS Phase):**

* **AI-native observability and incident-response products:** Companies whose core product reads logs/traces with an LLM and where token spend is a top-3 line item.
* **Enterprise platform teams:** Central groups running self-hostable, zero-data-retention log analysis over proprietary infrastructure telemetry.

**Anti-ICP:** General text summarization, prose-heavy document QA, creative writing assistants, and conversational chatbots that do not operate on structured operational data.

---

## Core User Journey

**Observability / SRE Engineer:**

1. Engineer installs Polymorph locally via standard package managers.
2. Engineer connects Polymorph as an MCP server (or inline in the log-ingestion path) within their configuration (e.g., `.claude/settings.json`, an incident copilot, or a gateway).
3. Every block of audit logs or trace spans bound for the LLM is piped through Polymorph's token-locking pipeline.
4. Polymorph parses embedded JSON, locks the structural fields, error codes, and state transitions, uses its ML classifier to drop redundant heartbeat/INFO chatter and duplicated frames, and caches dropped spans locally.
5. The LLM gets a lean payload, queries run faster, and the token budget extends 1.3–1.6×—with nothing forensically relevant lost, because every drop is retrievable.

**Security / Compliance Team:**

1. Deploys the Polymorph Rust binary locally to intercept logs before they reach the LLM gateway.
2. Configures custom DAAC dictionaries to strictly lock proprietary identifiers, resource ARNs, PII patterns, and compliance-relevant tokens so they are never compressed away.

---

## Main Offering

A three-layer product strictly tailored for audit logs and production traces:

1. **Open-source Core (Apache 2.0):** A native Rust binary leveraging a linear-time sequence model for ML inference, complete with the DAAC and Tree-sitter token-locking pipelines.
2. **MCP Server Integration:** A fully compliant Model Context Protocol server that plugs directly into incident copilots, trace-analysis agents, and the broader observability ecosystem.
3. **Hosted API (SaaS):** A managed endpoint with autoscaling, observability, and custom DAAC configuration profiles. Priced strictly on tokens saved.

---

## Core Features

* **Zero-Latency Structural Token Locking:** Uses Tree-sitter to parse embedded JSON event bodies and structured payloads, map the exact byte intervals of structural nodes, and enforce a Boolean mask that protects critical fields from being dropped.
* **DAAC Lexical Scanning:** Employs a Double-Array Aho-Corasick automaton to scan 100k-token log streams in $O(N)$ time (tens of microseconds) to identify and lock error codes, trace/span IDs, severity levels, API keys, and resource identifiers.
* **Byte-to-Token Mapping:** Integrates Tiktoken's `decode_with_offsets` to translate physical character byte intervals perfectly into LLM token boundaries, solving UTF-8 misalignment.
* **Compress-Cache-Retrieve (CCR) for arrays & logs:** Dynamically identifies massive JSON event arrays and dense log streams, retains the first/last elements to preserve boundaries, aggressively drops redundant operational statuses while rigorously preserving state transitions and explicit error codes, and caches the rest locally. The agent is provided a tool to retrieve the uncompressed cache instantly if needed.
* **Evidence-gated ML pruner (dual-CRF):** Built only if a benchmark shows the deterministic stack leaves meaningful compressible residual. When built, the leading candidate is a compact bidirectional chunked encoder (LLMLingua-2 family) feeding dual Conditional Random Field heads (semantic evidence vs. dependency scaffolding) governed by a learned head gate. The architecture is a candidate pending verification, not a locked choice; the deterministic stack stands on its own without it.
* **Zero Data Retention Mode:** Fully self-hostable. No logging of proprietary audit payloads.

---

## Product Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│  Log/Trace Source + MCP Client (incident copilot / agent)    │
└─────────────────────────┬────────────────────────────────────┘
                          ▼ (stdio / SSE / HTTP)
┌──────────────────────────────────────────────────────────────┐
│ Polymorph MCP Server (Rust Native)                           │
│                                                              │
│ 1. Tiktoken BPE Mapping (Extracts Byte Offsets)              │
│ 2. Parallel Scanning:                                        │
│    ├── Tree-sitter (embedded-JSON byte interval extraction)  │
│    └── DAAC Automaton (O(N) error code / ID / severity lock) │
│ 3. Sweep-Line Intersection (Generates Boolean Lock Mask)     │
│ 4. Pattern/Template Dedup + CCR (deterministic compression)  │
│ 5. Evidence-gated ML pruner (dual-CRF, on prose residual)    │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────── ┐
│   Forward Compressed Payload to LLM Gateway / Provider       │
└──────────────────────────────────────────────────────────────┘

```

---

## Research Thesis

We believe three things, and we intend to prove them in public:

1. **Syntax is Sacred, Prose is Malleable.** Machine learning should never guess whether a brace, a field separator, or a trace ID is important. Deterministic parsing (AST/Regex) must handle structure; probabilistic ML classifiers should only handle the remaining natural-language boilerplate.
2. **The neural stage should be small, bounded, and earned.** Token classification does not need an exotic long-context backbone, because the model never sees the whole stream: deterministic locking + template dedup + chunking bound its input to small windows of residual prose. A compact bidirectional encoder over those windows is both more accurate (it has future context for keep/drop) and more deployable (mature INT8 CPU inference) than a causal long-context model. And the pruner is built only when a benchmark proves the deterministic stack left enough residual to be worth it.
3. **Audit data requires reversible compression.** Investigations operating on massive trace and log dumps need dynamic budgeting via the Compress-Cache-Retrieve architecture, ensuring heavy pruning is always backed by a local safety cache—so no forensically relevant line is ever lost.

---

## Product Philosophy

1. **The agent is a first-class user.** Polymorph is built ground-up for autonomous incident and analysis workflows. MCP and tool-call ergonomics are the primary integration layer.
2. **The audit trail must stay intact.** We compress by deletion, but we never delete structural anchors, error codes, or state transitions. Same input record → same parseable record, minus the noise.
3. **Rust native for zero overhead.** Relying on Python for high-frequency token routing is a mistake. By utilizing a Rust-native sequence model and batch-invariant `bf16` inference, we guarantee extreme speed and bit-parity.
4. **Small surface, deep quality.** We do not summarize documents. We do not generate text. We do exactly one thing: flawless, high-speed structural compression of audit logs and production traces.
