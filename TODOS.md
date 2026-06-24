# TODOs

State as of 2026-06-24.

The Rust MCP runtime is installable from source, has local verification
(`--selftest`, `--demo compress`), supports Claude Code and Cursor config
examples, and runs deterministic mode without a model. The optional `mb_v0`
ModernBERT ONNX model is live when `POLYMORPH_LAMR_MODEL` points at the artifact.

## Release Operator Checklist

- Upload the `mb_v0` ONNX bundle to the GitHub Release URL expected by
  `scripts/fetch_model.sh`.
- Set the release asset SHA256 in release notes and pass it as
  `POLYMORPH_MODEL_SHA256` when verifying the fetch script.
- Run the acceptance commands in `README.md` on a clean checkout for both:
  deterministic mode (`used_model:false`) and full model mode (`used_model:true`).
- Attach CI logs for `cargo fmt --check`, `cargo test`, `cargo build --release`,
  and `./target/release/polymorph-mcp --selftest`.

## Optional Follow-Ups

- Add prebuilt release binaries once the source-install path has proven stable.
- Add an INT8 model selection guide only if size-constrained installs become common;
  fp32 remains the default because INT8 is smaller but slower in current tract tests.
- Re-run the same LogHub eval with `llmlingua2` on Modal for a same-set baseline.
- Investigate OpenStack, the weakest LogHub domain in the current eval.
- Port the remaining Python corpus adapters only if Rust share by bytes matters for
  presentation; they are not needed for the runtime install path.
