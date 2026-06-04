# TODOs

## Deferred: Full LaMR Distill-To-Shard CLI

**What:** Add a CLI/module that turns LiteLLM distillation JSONL into labeled training shards for `LabeledShardDataset`.

**Why:** The current LaMR core pass intentionally fixes math, labels, loss, export, and runtime contract first. Full data generation still needs a repeatable path from `{original, claude, gpt4o}` records to `{input_ids, tags, w_semantic, w_dependency}` shard rows.

**Pros:** Makes training data generation reproducible, removes the inline shard-building logic from smoke scripts, and gives future experiments a single command for teacher selection, QC filtering, alignment, and AST hop-decay labeling.

**Cons:** Touches distillation, QC, labeling, config, docs, and smoke scripts. It should land after the core LaMR math/export contract is stable to avoid mixing pipeline ergonomics with model behavior changes.

**Context:** Start from `ml_pipeline/polymorph_lamr/distill/run_distill.py`, `ml_pipeline/polymorph_lamr/qc/metrics.py`, `ml_pipeline/polymorph_lamr/label/align.py`, `ml_pipeline/polymorph_lamr/label/ast_split.py`, and `ml_pipeline/polymorph_lamr/train/dataset.py`. The CLI should read distilled JSONL, compute QC for each teacher output, choose or reject a compressed target, derive cl100k keep/drop masks, apply configured AST hop-decay, and write JSONL shard rows compatible with the training dataset.

**Depends on / blocked by:** Complete the LaMR core math/export pass first, especially the final definitions for VR/AG edge cases, hop-decay config wiring, and weighted dual-CRF training/inference contract.

## Deferred: Rust ONNX Runtime For LaMR

**What:** Replace the deterministic mock pruner in `src/lamr.rs` with real ONNX inference over unlocked token IDs.

**Why:** The Python core pass defines the artifact contract, but the MCP server will still use the mock until Rust loads `model.onnx`, reads `transitions.npz`, runs the weighted CRF decode, and scatters decoded drops back into the full `drop_mask`.

**Pros:** Makes the trained LaMR artifact visible in the product runtime, preserves the existing `lock_mask[i] => !drop_mask[i]` invariant, and turns the Python export into a shippable MCP behavior instead of a training artifact only.

**Cons:** Requires Rust ONNX runtime selection, artifact loading, transition parsing, decode parity tests, and runtime error handling. It should follow the math/export pass so Rust does not chase a moving contract.

**Context:** Start from `src/lamr.rs`, `src/lib.rs`, and `src/mcp.rs`. The replacement path should collect unlocked token IDs, run ONNX to get semantic emissions, dependency emissions, and `head_weights`, combine both CRF parameter sets with the selected weighted-sum rule, run one Viterbi decode, and scatter the result back into `drop_mask` while keeping locked tokens `false`.

**Depends on / blocked by:** Complete the LaMR core math/export pass and its runtime-contract fixture tests first.

## Deferred: RLAIF Post-Training For Compression Policy

**What:** Design and implement an RLAIF-style preference optimization stage for LaMR after the supervised distillation baseline is stable.

**Why:** The current ML pipeline is supervised: teacher compressions become keep/drop labels through QC, cl100k alignment, and AST hop-decay. RLAIF would require a separate reward/preference signal that scores compression quality, syntax safety, downstream answer quality, and token savings.

**Pros:** Could improve compression policy beyond static teacher labels, especially for tradeoffs between aggressive deletion and downstream task accuracy. It also creates a natural eval harness for comparing teacher outputs and model decisions.

**Cons:** It is research-heavy and needs new data, reward definitions, eval baselines, and training loops. Doing it before supervised LaMR is stable would hide basic math bugs behind noisy optimization.

**Context:** Start only after the core LaMR math/export pass and basic Rust runtime integration are working. Define preference examples over original/compressed pairs, reward terms for syntactic safety and answer preservation, and offline evals before any reinforcement-style training.

**Depends on / blocked by:** Stable supervised LaMR baseline, reproducible distill-to-shard CLI, downstream task evals, and a clear reward model spec.

## Deferred: Verify Research Citations Before Committing Architecture (P1)

**What:** Before the engineering review locks an architecture, verify that the systems named in the 2026-06-04 ML/NLP research report are real, reproducible, and accurately described: `DeLog-L`, `KRONE`, `mBERT-8`, `Gated DeltaNet-2`, and `Mamba-3` (with their cited benchmark numbers).

**Why:** The report drives the proposed pivot away from the stubbed Gated DeltaNet-2 toward a deterministic pattern-dedup pre-stage + bidirectional chunked encoder + dual-CRF. Deep-research agents can synthesize or mislabel systems and fabricate benchmark figures. The *ideas* (template/pattern dedup as a pre-stage, bidirectional context for keep/drop, hierarchical review classification) are sound on first principles; the *named systems and numbers* are unverified.

**Pros:** Stops the team building against a hallucinated system or citing fabricated benchmarks in docs/PRs. Cheap relative to the cost of a wrong architecture commit.

**Cons:** A verification task with no direct code output; needs literature search per claim.

**Context:** Source is the two research reports pasted into the 2026-06-04 CEO review (see `~/.gstack/projects/GaganSD-lulu-polymorph/ceo-plans/2026-06-04-polymorph-log-compression-direction.md`, Risk 1). For each system: find the paper/repo, confirm the described mechanism, and check the cited numbers (e.g., the "~74 primitive ops/token, 10-50x CPU ONNX slowdown" GDN-2 figure and the "40-60% dedup" DeLog-L figure). Treat unverifiable claims as ideas to validate empirically, not facts.

**Depends on / blocked by:** Nothing. Do this before `/plan-eng-review` finalizes the neural architecture.

## Deferred: Trace-QA Eval Set For The Compression Benchmark (P1)

**Promoted P2 -> P1 by the 2026-06-04 eng review (outside voice):** a ratio-only benchmark on clean synthetic data cannot discharge the "build the neural pruner?" gate and risks false certainty on an irreversible fork. The baseline ships a cheap proxy (seeded known-root-cause questions; assert answer tokens survive compression) plus at least one messier corpus alongside TrainTicket, and must verify TrainTicket's DATA license (Apache-2.0 covers the code, not necessarily the generated traces). This full eval set replaces the proxy once curated.

**What:** Curate a trace-QA eval set (questions + ground-truth answers over real traces) so the benchmark harness can measure downstream answer accuracy, not just compression ratio and latency.

**Why:** Direction B (deterministic-first, evidence-gated neural) depends on a benchmark that can answer "does compression hurt the LLM's ability to find the bug?" TrainTicket ships fault *labels*, not a QA set. Without answer-accuracy, the evidence gate can only use ratio + residual size + latency, which can't catch lossy-relevance regressions.

**Pros:** Unblocks the full evidence gate; makes compression-quality claims falsifiable; reusable for the relevance classifier eval.

**Cons:** Curation effort (templated questions over injected faults, or a small hand-labeled set). Quality of templated questions may be limited.

**Context:** Pairs with the benchmark harness (E2) in the 2026-06-04 CEO plan. Start from TrainTicket injected-fault intervals: generate questions like "what failed in service X around time T and why" with ground-truth derived from the fault injection metadata. Keep a small human-verified subset as a gold set.

**Depends on / blocked by:** Benchmark harness scaffolding; access to TrainTicket fault metadata.

## Deferred: Relevance-Label Denoising For REVIEW/DON'T-REVIEW (P2)

**What:** Combine deterministic structural signals with the ML relevance head so the REVIEW/DON'T-REVIEW classifier doesn't purely trust noisy infra-level fault labels.

**Why:** TrainTicket faults are injected at the infrastructure level. A routine container restart may be labeled anomalous with no diagnostic value; a slow resource leak may stay unlabeled until crash. A classifier trained purely on those labels inherits the noise.

**Pros:** Higher relevance fidelity; leverages the existing deterministic locker (5xx, error codes, state transitions auto-REVIEW); ML head only arbitrates the ambiguous remainder.

**Cons:** Adds label-pipeline logic; partly overlaps the E1 auto-REVIEW coupling already in scope (consolidate the two).

**Context:** Pairs with E1 in the 2026-06-04 CEO plan. Rule layer: any line whose locked tokens include a 5xx status, an error code, or a state transition is auto-REVIEW and bypasses the ML head. ML head learns the residual. Treat infra fault labels as a weak signal, not ground truth.

**Depends on / blocked by:** Deterministic locker structural-signal extraction; relevance classifier prototype (E1).

## Deferred: Full Pluggable Pruner Seam (defer until ONNX is imminent)

**What:** Build the full pluggable pruner seam (Identity / Mock / future OnnxLamr) with the `lock_payload` signature change and the corresponding MCP wire-contract update for `drop_mask` / `kept_tokens`.

**Why:** The 2026-06-04 eng review (outside voice) decoupled this from the dedup baseline: bundling a cross-cutting API + MCP-contract change with a net-new feature inflates blast radius, and the baseline only needs a minimal "run the deterministic path with no pruner" switch to get a clean benchmark number. The full seam is the seam the ONNX model will plug into; build it when that model is actually imminent.

**Pros:** Clean place for the future ONNX pruner; removes the semantically dead `drop_mask` field that an Identity-default would otherwise ship into the MCP contract early.

**Cons:** Touches `src/lib.rs` (`lock_payload` signature), `src/mcp.rs` (`lock_mask` tool contract), and `tests/mcp_*` assertions on `drop_mask` / `kept_tokens`. Decide the wire-contract semantics (what does `drop_mask` mean when no pruner runs) as part of this.

**Context:** Start from `src/lib.rs:86` (`lock_payload`), `src/lamr.rs` (mock), `src/mcp.rs:55` (`lock_mask` tool). Baseline ships only the minimal mock-off switch; this TODO is the proper refactor.

**Depends on / blocked by:** A real ONNX pruner being on the near horizon (i.e., the benchmark gate from direction B has opened). Until then, the minimal mock-off switch is enough.
