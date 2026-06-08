# TODOs

Pick-up context for the next session. State as of 2026-06-08.

**Where we are:** the LaMR neural pruner (ModernBERT-150M, `mb_v0`) is trained, **SOTA-for-class
on answer survival** — on the full 187-triple LogHub set (Modal-judged, 95% CIs, McNemar):
lamr+span **62% [55–69] @3× / 44% [36–51] @5×** vs keep-severity 17%/14%, McNemar p ≈ 0 (wins
92–9 / 66–10 discordant pairs) — and is now **live end-to-end in the Rust runtime.** The
model-in-runtime requirement is **DONE**:

- Pure-Rust **ModernBERT tokenizer** (`src/modernbert.rs`) — byte-exact with HF on fixtures
  (mb_v0 uses the ModernBERT tokenizer, not cl100k; no `onig`/native dep — tiktoken-rs BPE
  over raw-byte keys + GPT-2 regex + longest-match added-token pre-pass).
- **Windowed** `tract` inference (`src/lamr.rs`) — 512-token windows, pad+mask last window;
  drop-prob parity with PyTorch `max_abs_diff = 3e-6` over a 5-window doc.
- `compress_text` (`src/compress.rs`) + the **`compress_log` MCP tool** (`src/mcp.rs`,
  text|path in → compressed text + `cache_id` out) + a **Claude Code skill**
  (`.claude/skills/polymorph/`). `Language::PlainText` added so raw logs aren't force-parsed
  as JSON (that over-locked ~42% of tokens and evicted free-text needles).
- Verified: a **10 MB** repetitive log compresses 1864× overall (dedup → bounded neural
  residual) with every needle preserved (`tests/mb_v0_parity.rs`).

The remaining open requirement is **latency** (below). Accuracy bar cleared; runtime bar cleared.

Artifacts: `data/modal_out/mb_v0/{ckpt-best.pt, onnx/model.onnx}` (gitignored; re-pull with
`modal volume get polymorph-lamr-v0 /out/mb_v0/... data/modal_out/mb_v0/...`). Gate result:
`data/bench/gate_mb_v0_broad.json`.

---

## 1. Latency — mostly RESOLVED in release; INT8 is an optional size trade (2026-06-08)

**The "~80 s load" was a debug-build + cold-disk artifact, not the real cost.** Measured in a
**release build** (`tests/mb_v0_int8.rs`, `tests/mb_v0_parity.rs`):

| Model | Size | tract load | inference (2077-tok doc) | decode agreement vs fp32 |
|---|---|---|---|---|
| **fp32** `model.onnx` (shipped default) | 600 MB | **2.4 s** | **2.8 s** | — (parity 3e-6) |
| int8 `model.int8.onnx` (per-channel) | 154 MB | 1.0 s | 8.98 s (3.2× slower) | top-k drop Jaccard **0.994** |
| dynbatch `model.dynbatch.onnx` | 597 MB | 2.1 s | 2.7 s | parity 3e-6 (no-op win) |

So for a multi-window doc: ~2.4 s load (once per process) + ~2.8 s inference in release; dedup
bounds the residual so typical logs are faster. **Interactive enough — latency is no longer a
launch blocker.** Always build `--release` (debug inference is ~15× slower).

**INT8 (`ml_pipeline/scripts/quantize_int8.py`) WORKS in tract** — the QLinear/MatMulInteger
coverage worry didn't materialize. It's a 3.9× smaller artifact with near-identical decisions
(Jaccard 0.994), but inference is 3× slower (tract's int8 GEMM is less optimized). **Ship fp32 as
default; offer int8 as an optional small artifact for size-constrained installs.** Dynamic-batch
re-export is a no-op until a future runtime batches windows (current `forward_window` is batch=1).

**Open (minor):** point a runtime flag at `model.int8.onnx` when size matters; re-confirm the
release load number on a cold disk. **C4 (distilled smaller model on Modal) is NOT justified** by
these numbers. **Do NOT** use CoreML (730/1028 nodes → 178 partitions → 29 s, slower).

## 2. Wire the trained ONNX into the Rust MCP server — ✅ DONE (2026-06-08)

The runtime now feeds the real `mb_v0` model. See the "Where we are" section above:
`src/modernbert.rs` (pure-Rust tokenizer, byte-exact), windowed `tract` inference in `src/lamr.rs`
(drop-prob parity 3e-6), `src/compress.rs` + the `compress_log` MCP tool + the Claude Code skill.
Single-logit sigmoid contract (no CRF). Verified by `tests/mb_v0_parity.rs` (forward parity +
10 MB needle survival) and `tests/mcp_inproc.rs` (tool round-trip + cache).

## 3. Defendable eval — broadened + now runs on MODAL (2026-06-08, in progress)

**Done:** `polymorph_lamr/bench/stats.py` computes per-domain / per-fact_type survival,
**McNemar** paired test (Yates χ² + exact) vs keep-severity, and seeded **bootstrap 95% CIs**,
wired into `judge_bench.py` via `--stats` (+ 15 unit tests, `ml_pipeline/tests/test_bench_stats.py`).
On the published 50-triple set the headline holds and is now significance-backed: lamr+span
**68% [56–80] @3× / 48% [36–62] @5×**, McNemar p_exact ≤ 7e-6 vs keep-severity; exact-match
(no judge) 60%/42%. Weak spot: openstack 0/4.

**Compute moved to Modal** (no more local CPU eval — it was ~22 s/triple on the throttled Mac).
`ml_pipeline/cloud/eval_modal.py` runs `judge_bench` on a **T4 GPU**, reading the `mb_v0`
checkpoint from the `polymorph-lamr-v0` volume (`/out/mb_v0/ckpt-best.pt`) and the judge key from
the `polymorph-judge` Modal secret (Vercel AI Gateway). Run:
`ml_pipeline/.venv/bin/modal run ml_pipeline/cloud/eval_modal.py`; download with
`modal volume get polymorph-lamr-v0 /bench_out data/bench`.

**DONE (Modal GPU, 0 judge errors):** the full **187-triple** run @3,5× landed →
`data/bench/bench_out/{gate_mb_v0_full.json,_stats.json}`. Defensible headline (judge, 95% CI):
lamr+span **62% [55–69] @3× / 44% [36–51] @5×** vs keep-severity 17% [12–23] / 14% [9–19];
McNemar p ≈ 0 both ratios. Exact-match-only 56%/37%. Per-domain @3×: BGL 88%, Linux 68%, Hadoop
52%, Spark 50%, ZooKeeper 57%, **OpenStack 25% (weak)**. README results table updated. The
earlier 50-triple 68%/48% was optimistic (wide CIs); 62%/44% is the honest claim.

**Minor follow-ups:** (a) re-run with `llmlingua2` on Modal for a same-set baseline (excluded to
keep the image lean); (b) `judge_bench._bootstrap_judge_env()` does NOT auto-load `.env` despite
its docstring — fix the docstring or wire dotenv; (c) OpenStack is the weak domain — worth a look.

## 4. Replace gibberish needles in `heldout_triples.json` (P2) — quarantined

**Confirmed:** 230/363 `semantic:msg` answers in `data/bench/heldout_triples.json` are random
gibberish. It is **excluded** from the headline eval (the clean `loghub_triples.json` — 187
prose triples, 7 domains — is the only stick used). Regenerating real msg needles needs a
prose/LogHub source shard (not the synthetic val shard `build_heldout.py` mines); deferred.

## 5. (Optional) TACO-RL / RLAIF post-training (P3)

**What:** REINFORCE-style post-training on the same encoder with a downstream
answer-survival reward, for the lift over the supervised baseline. Only pursue if the
latency-fixed supervised model is close to but short of the target — supervised already cleared
the accuracy bar, so this is a refinement, not a dependency.
