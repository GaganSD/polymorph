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
