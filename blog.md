# Achieving SoTA on Token Compression Log

A working log of how we built a local-first, log-domain token compressor that beats the published baselines on answer survival. Written as bullets, warts included.

## The goal

- One sentence: **low-latency, SOTA log compression that runs locally.** Three hard requirements — low latency, SOTA accuracy, local.
- Use case: an ML engineer pipes a 10 MB log into Claude Code and wants a context middleware that shrinks it without losing the line that matters.
- Governing objective throughout: **high recall.** A dropped needle costs far more than a kept filler token.

## What we're building

- **Extractive** compression only — we delete tokens, never rewrite them. So a status code / IP / exception type that survives is byte-identical. (Abstractive summarizers hallucinate → disqualified for audit logs.)
- Two layers: a **deterministic structure layer** (token-locking, template dedup, compress-cache-retrieve) and a **learned pruner** over the unlocked prose residual.
- The learned pruner = **LaMR**: a bidirectional encoder + a single per-token "drop" head. `sigmoid(logit) = P(drop)`, one forward pass, ONNX/tract deployable.

## What we used

- **Model:** `answerdotai/ModernBERT-base` (150M) fine-tuned as a token classifier. Pretrained encoder >> from-scratch.
- **Decode:** span-aware ("ChunkKV" idea) — group tokens into whitespace words, drop whole words by an aggregated score, never fragment a multi-word needle. Aggregator = `max` (counterintuitive; see mistakes).
- **Labels:** byte-level alignment (difflib over UTF-8 bytes) projected onto token spans, so labels survive a tokenizer swap.
- **Training:** Modal GPU (A100), AMP + grad-accum, PR-AUC-selected best checkpoint. ~18 min, ~$3.
- **Benchmark:** answer-survival on (log, question, gold) triples from LogHub-2.0 prose; iso-ratio comparison; LLM-judge = Claude Sonnet 4.6 entailment.
- **Baselines:** keep-severity (keep the most severe lines) and LLMLingua-2 (Microsoft, the published SOTA extractive compressor).

## What we did (chronological-ish)

- Replaced the from-scratch 28M cl100k encoder with pretrained ModernBERT; made the whole label/align/decode path tokenizer-agnostic (cl100k ↔ modernbert) via byte offsets. Label ceiling preserved exactly (90.8% / 97.3%).
- Built span-aware decode → fixed phrase fragmentation (12% → 68% survival @3×).
- Proved the neural path is justified **on unstructured prose**: keep-severity collapses (98→35→17% as ratio rises), the structural floor survives only 1–2% — exactly the gap a learned model fills.
- Built an **iso-ratio** gate so methods are compared at *equal compression*, not equal target rate (the earlier tables compared apples to oranges).
- Trained `mb_v0` to PR-AUC 0.873 / ROC 0.933, exported to ONNX (parity 3.9e-6).
- Ran the decisive gate: **`lamr+span` 68% @3×, 48% @5× vs keep-severity 14%/10%, vs LLMLingua-2 ~20%.** Accuracy SOTA-for-class.
- Benchmarked local latency — and hit a wall (see below).

## Challenges

- **Phrase fragmentation:** token-level top-k drops one token inside a multi-word needle and shatters it. Fixed by word-level span decode.
- **Long docs vs short training window:** real logs are ~4400 tokens; the model trains on 512. Naively truncating to one window silently drops every tail token. Fixed with windowed inference (score the whole doc in 512-token windows, concat probs).
- **Comparing fairly:** every method hits a different ratio at the same target drop rate. Built per-item bisection to a target compression ratio.
- **Judge design:** "answer the open question, then substring-match the gold" scored *correctly recovered* facts as 0 (paraphrase, generic question, truncation). Rewrote as a YES/NO entailment on the gold fact.
- **Latency, still open:** 150M on CPU is ~5–7 s/doc — barely faster than a 3.7× larger model, i.e. the inference path is unoptimized. CoreML made it *worse* (graph shatters). The win needs quantization / dynamic-batch / tract — not yet done.
- **Infra flakiness:** Modal's default function timeout killed the first run; a worker preemption killed the second.

## Where we landed

- **Accuracy: SOTA-for-class. ✓** ~4.9× over keep-severity, ~3.4× over LLMLingua-2, at matched compression, judged on answerability.
- **Runs locally: ✓** (150M, ONNX, extractive).
- **Low latency: ✗ not yet.** The one remaining bar. It's an engineering problem (optimize inference), not a research one (we proved a local small model can win).

## Mistakes made

- **Wrong prior on the decode aggregator.** Assumed `min` (drop a span only if *every* token is droppable) was the high-recall choice. The data said the opposite: with a calibrated model `max` wins decisively (68% vs 23% survival @3×). `min` is pathological — it refuses to drop almost anything, so to hit the rate it drops *needles*.
- **Trusted a benchmark file that was partly noise.** 230/363 `semantic:msg` needles in the old heldout set were random gibberish; early headline numbers were measured on noise. Switched to LogHub prose triples.
- **Shipped a judge that scored truth as 0.** First gate run showed 0% judge survival even where the gold answer was verbatim present — the judge was finding the right fact but the strict substring match failed on paraphrase. Nearly mistook a broken metric for a bad model.
- **Forgot inference is length-bounded.** First end-to-end test gave a nonsensical 12× compression at a "0.6 drop rate" because `LaMRMethod` truncated the 4400-token doc to one window. Caught it via a sanity ratio check, not a test — added windowed inference.
- **Let Modal defaults bite twice.** (1) 2-hour function timeout killed a 4.5-hour run at step ~665. (2) A worker preemption restarted from scratch (no resume support) *and* the reset best-tracker overwrote the good checkpoint with a worse early one. Fix was a 6-hour timeout + faster GPU (shorter exposure).
- **Reached for a bigger GPU when the real lever was batch size.** The T4 was overhead-bound (micro-batch 4 → 16 sequential micro-steps), not compute-bound. A100 + micro-batch 32 cut step time ~16×. 80GB/H100 would have been wasted money on a 150M model.
- **Read a checkpoint mid-download.** Tested loading `ckpt-best.pt` while the background download was still writing it → "corrupted archive" false alarm. It was fine once the transfer finished.

## What we'd tell the next person

- Prove the *uncertain* thing first (can a local small model win on accuracy?) before polishing the certain things. We did, and it's yes.
- Measure latency on a cool machine, in the actual runtime (tract), after quantization — the dev-Mac PyTorch/onnxruntime numbers are a floor, not the verdict.
- The infra (MCP + locking + dedup + structural floor + Rust runtime) is the moat; the model is a swappable, optimizable component.
