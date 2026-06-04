# Runbook: OpenRouter distillation → LaMR training

End-to-end path to start training the LaMR token classifier using **OpenRouter
open-weight teachers** (no Claude/GPT-4o, no Anthropic §D.4 question). Follows the
2026-06-04 CEO + eng review: deterministic-first, evidence-gated neural.

## 0. Prereqs

```bash
# Rust baseline (deterministic stack + benchmark)
cargo build --release

# Python training env (heavy: torch/onnx)
cd ml_pipeline
uv venv && uv pip install -e '.[train]'

# OpenRouter key (the teacher provider)
export OPENROUTER_API_KEY=sk-or-...        # https://openrouter.ai/keys
```

## 1. Fetch corpora

```bash
bash scripts/fetch_datasets.sh             # TrainTicket (CC-BY-4.0) + server-logs (CC0)
```
Kaggle *kernels* need auth; see `data/DATA_CARD.md`. Licenses + attribution there.

## 2. Run the benchmark = the evidence gate (do this FIRST)

The neural pruner is built ONLY if the deterministic stack leaves meaningful
compressible residual. Measure before you train.

```bash
# messy real logs (honest hard case)
./target/release/polymorph-mcp --bench data/raw/server_logs 256 16
# templated app logs (optimistic end)
./target/release/polymorph-mcp --bench data/bench/trainticket_logs 256 16
```
Reports: compression ratio, throughput (MB/s), per-chunk p50/p95/p99, and an
answer-token survival proxy. SLO is throughput + per-chunk p95, NOT a flat 150ms
(a 10MB trace is ~2.5M tokens). A high ratio with low survival is a RED flag.

> Gate (tunable): build the neural pruner only if deterministic dedup leaves
> > ~20% residual prose AND a teacher oracle can drop > ~15% more without dropping
> the survival/answer proxy. Otherwise the deterministic stack IS the product.

## 3. Distill with the OpenRouter teacher ensemble (E3)

Each chunk is compressed by several open-weight teachers; the per-chunk **best-QC**
output (VR==0, lowest Alignment Gap) becomes the training target.

```bash
# default teachers: Qwen-2.5-72B, DeepSeek, Llama-3.3-70B (override with --teachers)
OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
python -m polymorph_lamr.distill.run_distill \
    --in data/raw/trainticket --out data/distilled/trainticket.jsonl \
    --mode ensemble --concurrency 8 --max-tokens 512

# custom teachers:
#   --teachers qwen=openrouter/qwen/qwen-2.5-72b-instruct \
#              mistral=openrouter/mistralai/mistral-large
```
Output JSONL schema (per chunk): `original`, `outputs{teacher->text}`,
`compressed` (selected target), `chosen_teacher`, `qc{vr,ag,mr,hr}`, `cost_usd`,
`errors`.

## 4. QC → label → train → export (existing pipeline)

```bash
# QC: drop top-VR (hallucination) then top-AG (alignment failure)
#   filter operates on the distilled JSONL (see polymorph_lamr/qc/)
# label: LCS alignment -> cl100k keep/drop mask + AST hop-decay split
# train: joint dual-CRF NLL with the learned head gate
lamr-train  --config configs/default.yaml --shards data/labeled/*.jsonl --out artifacts/ckpts
lamr-export --ckpt artifacts/ckpts/ckpt-final.pt --out artifacts/lamr-v0 --config configs/default.yaml
```

> Note (eng review): the labeling stage consumes the ensemble `compressed` field as
> the target. The full distill→shard CLI is a tracked TODO; until it lands, point
> the labeler at the `compressed`/`original` fields of the ensemble JSONL.

## What changed for OpenRouter (2026-06-04)

- `distill/prompts.py`: added `LOG_TRACE_EXTRACTIVE` (telemetry-tuned, extractive-only)
  + `DEFAULT_OPENROUTER_TEACHERS`.
- `distill/client.py`: added `TeacherSpec`, `EnsembleConfig`, `EnsembleResult`,
  `select_best` (best-QC), `distill_ensemble[_many]`. Legacy `distill_pair`
  (Claude+GPT-4o) retained for back-compat.
- `distill/run_distill.py`: default `--mode ensemble`; `--teachers` override; `.log`
  ingestion; log-aware chunking.
- `configs/default.yaml`: `distill.mode/teachers` (OpenRouter), `temperature: 0.0`.

The architecture target if/when the neural pruner is built is a compact
**bidirectional chunked encoder + dual-CRF (LLMLingua-2 family), NOT Gated
DeltaNet-2** (`backbone.py` is still a stub). Verify before committing — see TODOS.md.
