# TODOs

Pick-up context for the next session. State as of 2026-06-07.

**Where we are:** the LaMR neural pruner (ModernBERT-150M, `mb_v0`) is trained and is
**SOTA-for-class on answer survival** — 68% @3× / 48% @5× vs keep-severity 14%/10% and
LLMLingua-2 ~20% @3× (iso-ratio, 50 LogHub prose triples, Claude-judged). Checkpoint:
PR-AUC 0.873, ONNX parity 3.9e-6. The **accuracy bar is cleared.** Two requirements remain
open: **low latency** and **the model being live in the runtime.**

Artifacts: `data/modal_out/mb_v0/{ckpt-best.pt, onnx/model.onnx}` (gitignored; re-pull with
`modal volume get polymorph-lamr-v0 /out/mb_v0/... data/modal_out/mb_v0/...`). Gate result:
`data/bench/gate_mb_v0_broad.json`.

---

## 1. Latency optimization — BLOCKER for "low latency, runs locally" (P0)

**What:** make local inference fast enough for interactive use. Today CPU inference is
~5–7 s per 4400-token doc warm (~700 tok/s) — barely faster than a 3.7× larger model, so the
path is **unoptimized, not at a fundamental limit.**

**Levers, in order of expected ROI:**
- **INT8 dynamic quantization** of the ONNX (onnxruntime `quantize_dynamic`). 2–4× on CPU, easiest. Verify survival doesn't regress after quantizing.
- **Dynamic-batch re-export.** Current `model.onnx` is fixed `batch=1` (`[1,512]`), so the 9–10 windows of a long doc run as sequential `session.run` calls. Re-export with a dynamic batch axis (keep fixed seq 512) and run all windows in one forward. NB: ModernBERT dynamic-seq export previously failed → tract `Flatten`; dynamic *batch* with fixed seq is the safer variant — verify tract still loads it.
- **`tract` (Rust) benchmark.** The actual production runtime, never measured. Could differ materially from onnxruntime.
- **Smaller / distilled backbone** if the above isn't enough.

**Do NOT:** use CoreML for this graph — only 730/1028 nodes are supported, so it shatters into
178 partitions and runs *slower* (29 s). Also: sustained benchmarking thermally throttles the
dev Mac (numbers inflate run-over-run) — re-measure cold or on a server CPU.

## 2. Wire the trained ONNX into the Rust MCP server (P0)

**What:** the MCP server still runs the mock pruner. `src/lamr.rs` has the span-aware decode
ported, but nothing loads the real model.

**How:** load `model.onnx`, collect unlocked token IDs, run **windowed** inference (512-token
windows — matching training; truncating loses tail needles), sigmoid → per-token `P(drop)`,
span-aware (word, `max`-aggregator) decode to the target rate, scatter drops back into
`drop_mask` while honoring `lock_mask[i] => !drop_mask[i]`. **Single-logit sigmoid contract —
no CRF, no transitions.npz, no head_weights** (the dual-CRF design was dropped).

**Context:** `src/lamr.rs`, `src/lib.rs` (`lock_payload`), `src/mcp.rs` (`lock_mask` tool).
Python reference for parity: `polymorph_lamr/bench/methods.py::LaMRMethod` (windowed `_score`
+ `spandecode.span_decode`).

## 3. Broaden the eval before any "SOTA" claim leaves the repo (P1)

**What:** 50 docs / 2 ratios is a strong internal signal, not a defensible external claim. Add
more LogHub domains, more ratios, per-domain breakdown, and a significance test (McNemar vs
keep-severity on paired survival). Report confidence intervals.

**Context:** `polymorph_lamr/bench/judge_bench.py` already supports `--iso-ratio a,b,c` and
writes per-item results; add per-`fact_type` aggregation + paired stats on the saved JSON.

## 4. Replace gibberish needles in `heldout_triples.json` (P2)

**What:** 230/363 `semantic:msg` answers in `data/bench/heldout_triples.json` are random
gibberish strings — exact-match matched the noise and the old headline numbers were partly
measured on it. LogHub triples (`data/bench/loghub_triples.json`) are the trustworthy stick;
either regenerate the msg needles with real extractable facts or drop that file.

## 5. (Optional) TACO-RL / RLAIF post-training (P3)

**What:** REINFORCE-style post-training on the same encoder with a downstream
answer-survival reward, for the lift over the supervised baseline. Only pursue if the
latency-fixed supervised model is close to but short of the target — supervised already cleared
the accuracy bar, so this is a refinement, not a dependency.
