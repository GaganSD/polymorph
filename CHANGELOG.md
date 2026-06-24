# Changelog

## Unreleased

### Added
- Install verification via `polymorph-mcp --selftest`, including grammar, SQLite cache, and model-path diagnostics.
- `polymorph-mcp --demo compress` plus `examples/sample.log` for a local first-run compression demo.
- Claude Code and Cursor MCP config guidance via `mcp.example.json`.
- `scripts/fetch_model.sh` for downloading a public ONNX model artifact into the expected runtime path.

### Changed
- README now documents deterministic-first setup, optional LaMR model setup, Claude Code, Cursor, troubleshooting, and dependencies.
- Environment paths now expand a leading `~/` for cache, grammar, model, and log-file paths.

## 0.1.0

First release: the learned pruner is live end-to-end in the Rust runtime.

### Added
- `compress_log` MCP tool: log/trace text or a local file path in, a smaller log plus a `cache_id` out. The original is cached and retrievable via `polymorph_retrieve_cache`.
- Pure-Rust ModernBERT byte-level tokenizer (`src/modernbert.rs`), byte-exact with the Hugging Face tokenizer. No `onig` or other native dependency.
- Windowed `tract` ONNX inference for the LaMR pruner (`src/lamr.rs`), reproducing the PyTorch drop probabilities to `max_abs_diff = 3e-6`.
- `compress_text` pipeline (`src/compress.rs`): structural lock to byte intervals, ModernBERT tokenization, span-aware decode, reconstruction.
- `Language::PlainText` so raw logs are not force-parsed as JSON (which over-locked ~42% of tokens).
- Claude Code skill in `skills/polymorph/`.
- Modal GPU eval (`ml_pipeline/cloud/eval_modal.py`) and bench statistics (per-domain, McNemar, bootstrap CIs) in Rust via `polymorph-mcp --bench-stats`.
- INT8 quantization and dynamic-batch export scripts under `ml_pipeline/scripts/`.

### Results
- `lamr+span` answer survival on the 187-triple LogHub set: 62% [55–69] at 3×, 44% [36–51] at 5×, versus 17% / 14% for the keep-severity baseline. McNemar p ≈ 0 at both ratios.

### Notes
- Build `--release`; debug inference is ~15× slower.
- Licensed under MIT.
