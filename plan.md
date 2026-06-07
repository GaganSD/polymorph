> **STATUS 2026-06-07 — partly superseded. Read [`README.md`](README.md) + [`TODOS.md`](TODOS.md) first.**
> The **dual-CRF / query-adaptive head-gate** design below was **dropped**. The shipped LaMR
> pruner is a **single per-token sigmoid "drop" head** on a pretrained **ModernBERT-150M**
> encoder with **span-aware (word, `max`) decode** to a target rate. It is trained and
> **SOTA-for-class on answer survival** (68% @3× vs keep-severity 14%, vs LLMLingua-2 ~20%).
> The deterministic locking/dedup/CCR layer below is accurate and shipping. Latency
> optimization and wiring the ONNX model into the Rust runtime are the open items.

---

Audit logs and production traces are not uniform. A single investigation pulls together **semi-structured data**: structured event fields wrapped around free-text messages, deeply nested JSON payloads, multi-frame stack traces, and long timelines of repetitive operational chatter.

By integrating the latest research on **Lossless Context Management (LCM)** and **Latent Multi-Rubric (LaMR) Pruning**, the blueprint handles both strict structural boundaries (the parseable record) and fluid semi-structured flows (the prose around it).

Here is the blueprint.

### The Audit-Log & Trace Compression MCP Blueprint

**1. Zero-Latency AST & Lexical Locking (For Strict Structure)**

To prevent the corruption of structural boundaries in rigidly formatted records (JSON event bodies, key-value fields, structured payloads), the system relies on deterministic locking rather than probabilistic guessing.

- **Mechanics:** The pipeline uses Tiktoken's `decode_with_offsets` to translate the LLM's integer tokens into precise byte boundaries. Concurrently, a WASM-sandboxed Tree-sitter parser maps the exact byte intervals of embedded JSON and structured payloads, while a Double-Array Aho-Corasick (DAAC) automaton executes in strict $O(N)$ time to map finite constraints like error codes, trace/span IDs, severity levels, and resource identifiers.
- **The Sweep-Line Override:** A fast two-pointer sweep-line algorithm intersects these ranges, generating a Boolean mask that explicitly forces the deletion probability of all structural tokens to zero.

**2. Latent Multi-Rubric (LaMR) Pruning (For Semi-Structured Log/Text Hybrids)**

Semi-structured records—log lines that embed JSON, stack traces interleaved with free-text context—cannot be evaluated with a single rule. We implement the LaMR framework to handle log relevance as a multi-dimensional problem.

- **Dual-CRF Architecture:** When the ML pruner is built (gated on benchmark evidence), the leading candidate is a compact bidirectional chunked encoder (LLMLingua-2 family) feeding two Conditional Random Field (CRF) heads governed by a learned, query-adaptive head gate. The encoder is a candidate pending verification, not a locked choice; the backbone never processes the full stream, only bounded windows of unlocked prose.
- **Separating Evidence from Scaffolding:** This lets the model dynamically balance two conflicting sequence patterns: "Semantic Evidence" (dense, contiguous blocks of message text or error context) and "Dependency Support" (sparse, isolated lines like state transitions, exception headers, or field markers).
- **Bidirectional, not causal:** keep/drop decisions benefit from future context, and the deterministic pre-stage already bounds input length, so a bidirectional encoder beats a causal long-context model on both accuracy and CPU deployability.

**3. Compress-Cache-Retrieve (CCR) for Event Arrays & Server Logs**

Massive trace and log dumps (e.g., 1,000 log events or a continuous span stream) exhibit heavy redundancy but require specific formatting preservation.

- **Format-Aware Pruning:** Instead of uniformly deleting tokens, the MCP implements the CCR architecture. For JSON event arrays, it mathematically preserves structural boundaries by retaining the first and last elements, alongside only the highest-relevance items. For dense server logs, the system aggressively drops redundant operational statuses while rigorously preserving state transitions and explicit error codes.
- **Reversibility:** The heavily compressed payload is sent to the LLM, but the full data is cached locally (via LRU cache or SQLite). The MCP provides the LLM with a retrieval tool (e.g., `lcm_expand` or `polymorph_retrieve_cache`) to fetch the uncompressed data instantly if the agent decides it needs more context.

**4. Lossless Context Management (LCM) (For Long Trace Timelines & Log Histories)**

A long-running incident timeline is the ultimate semi-structured stream. Compacting it via sliding windows or lossy text summarization destroys the specific technical lineages an investigation needs for root-cause analysis.

- **DAG Summary Trees:** Polymorph maintains a database-backed, append-only Immutable Store. As a timeline grows, older spans and log entries are structured into a multi-resolution Directed Acyclic Graph (DAG) summary tree (Depth-0 for local specifics, Depth-1 for span synthesis, Depth-2 for global abstracts).
- **Lineage Retrieval Traversal:** Memory cues and lineage pointers are injected into the active prompt. The agent can use explicit tools like `lcm_describe` to inspect off-context nodes and `lcm_expand` to retrieve verbatim historical log spans into an isolated sub-agent window without saturating the main prompt.

**5. Deterministic pattern/template dedup, then an evidence-gated pruner**

Logs are dominated by repeated templates. Before any model runs, a deterministic single-pass pattern/template dedup stage collapses the repetition (the same log line emitted 10,000 times becomes one template + counts), and CCR + LCM handle arrays and long timelines. On real logs this captures most of the compression win with zero ML, microsecond latency, and full reversibility.

- **Then, only if earned:** a benchmark measures the residual. If the deterministic stack leaves meaningful compressible prose, a compact bidirectional pruner (dual-CRF) surgically targets the residual heartbeat lines, repetitive INFO chatter, and duplicated frames. If it doesn't, the deterministic stack is the product and the model is skipped.
- **Result:** the parseable records, error codes, and state transitions stay perfectly intact whether or not the neural stage ever ships, and every drop is recoverable from the cache/archive.
