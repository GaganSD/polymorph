# TODOs

Pick-up context for the next session. State as of 2026-06-08.

**Where we are:** the LaMR neural pruner (ModernBERT-150M, `mb_v0`) is trained, **SOTA-for-class
on answer survival** — on the full 187-triple LogHub set (Modal-judged, 95% CIs, McNemar):
lamr+span **62% [55–69] @3× / 44% [36–51] @5×** vs keep-severity 17%/14%, McNemar p ≈ 0 (wins
92–9 / 66–10 discordant pairs) — and is **live end-to-end in the Rust runtime** (pure-Rust
ModernBERT tokenizer + windowed `tract` inference + `compress_log` MCP tool + Claude Code skill;
10 MB repetitive log → 1864× with every needle preserved). Accuracy bar cleared; runtime bar
cleared.

Artifacts: `data/modal_out/mb_v0/{ckpt-best.pt, onnx/model.onnx}` (gitignored; re-pull with
`modal volume get polymorph-lamr-v0 /out/mb_v0/... data/modal_out/mb_v0/...`). Gate result:
`data/bench/gate_mb_v0_broad.json`.

---

## 1. Latency — mostly RESOLVED in release; INT8 is an optional size trade

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

INT8 (`ml_pipeline/scripts/quantize_int8.py`) **works in tract** — 3.9× smaller, near-identical
decisions (Jaccard 0.994), 3× slower inference. **Ship fp32 default; offer int8 for
size-constrained installs.** **Open (minor):** add a runtime flag pointing at `model.int8.onnx`
when size matters; re-confirm the release load number on a cold disk. **Do NOT** use CoreML
(730/1028 nodes → 178 partitions → 29 s, slower); C4 (distilled smaller model) is NOT justified.

## 2. Defensible eval — broadened, runs on MODAL, stats now in Rust

The defensible-eval stats (per-domain / per-fact_type survival, **McNemar** paired test with
Yates χ² + exact binomial vs keep-severity, seeded **bootstrap 95% CIs**) now live in Rust
(`src/stats.rs`) and are exposed as `polymorph-mcp --bench-stats <results.json>`;
`judge_bench.py --stats` shells out to that binary. (The former `bench/stats.py` +
`test_bench_stats.py` were ported to Rust and deleted — see §5.)

**DONE (Modal GPU, 0 judge errors):** the full **187-triple** run @3,5× →
`data/bench/bench_out/{gate_mb_v0_full.json,_stats.json}`. Defensible headline (judge, 95% CI):
lamr+span **62% [55–69] @3× / 44% [36–51] @5×** vs keep-severity 17% [12–23] / 14% [9–19];
McNemar p ≈ 0 both ratios. Exact-match-only 56%/37%. Per-domain @3×: BGL 88%, Linux 68%, Hadoop
52%, Spark 50%, ZooKeeper 57%, **OpenStack 25% (weak)**.

**Compute is on Modal** (`ml_pipeline/cloud/eval_modal.py`, T4 GPU, reads the `mb_v0` checkpoint
from the `polymorph-lamr-v0` volume and the judge key from the `polymorph-judge` secret). Run:
`ml_pipeline/.venv/bin/modal run ml_pipeline/cloud/eval_modal.py`; download with
`modal volume get polymorph-lamr-v0 /bench_out data/bench`.

**Minor follow-ups:** (a) re-run with `llmlingua2` on Modal for a same-set baseline; (b)
`judge_bench._bootstrap_judge_env()` does NOT auto-load `.env` despite its docstring — fix the
docstring or wire dotenv; (c) OpenStack is the weak domain — worth a look.

## 3. Replace gibberish needles in `heldout_triples.json` (P2) — quarantined

**Confirmed:** 230/363 `semantic:msg` answers in `data/bench/heldout_triples.json` are random
gibberish. It is **excluded** from the headline eval (the clean `loghub_triples.json` — 187
prose triples, 7 domains — is the only stick used). Regenerating real msg needles needs a
prose/LogHub source shard (not the synthetic val shard `build_heldout.py` mines); deferred.

## 4. (Optional) TACO-RL / RLAIF post-training (P3)

REINFORCE-style post-training on the same encoder with a downstream answer-survival reward, for
the lift over the supervised baseline. Only pursue if a latency-fixed supervised model is close to
but short of target — supervised already cleared the accuracy bar, so this is a refinement.

## 5. Rust port of the offline bench/eval/distill pure-logic (raises Rust share)

The pure-logic of the Python `ml_pipeline` bench/eval/distill stack was ported to Rust (13 modules:
`stats`, `triples`, `methods`, `survival`, `spandecode`, `structural`, `normalize`,
`adapters_common`, `loghub`, `align` (difflib byte-port), `label_ceiling`, `eval_metrics`, `cli`),
each with parity tests (263 Rust tests green). New CLI subcommands on the binary:
`--bench-stats --build-loghub --label-ceiling --build-triples --bench-survival --sampler-filter
--eval-metrics`. Deleted `bench/{stats,loghub,label_ceiling}.py` + their tests and rewired
`judge_bench.py --stats` to the binary. Rust share ~50% LOC / ~49% bytes (was ~35%).

**What stays Python (load-bearing, not deletable):** `methods/survival/triples/spandecode/
structural/normalize` (needed by `judge_bench.py`'s **torch** `LaMRMethod` + **network** judge),
`label/align.py` (torch training labeler), and `distill/adapters/_common.py` + the 10 CSV adapters
(2 are randomness-based synth generators). Their Rust ports are additive duplicates.

**Open (optional):** to cross 50% **by bytes** (GitHub Linguist's metric; ~49% now), port the 8
mechanical CSV/JSON adapters + rewire `distill/adapters/stage.py` to a Rust `--stage-corpus`
subcommand — **bounded but UNVERIFIED** (the adapters have no unit tests; risk of corrupting
training-data staging), or port the 2 synth generators too and delete `_common.py`.
