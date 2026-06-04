# Runbook: open-weight distillation → LaMR training

End-to-end path to start training the LaMR token classifier using **open-weight
teachers** (no Claude/GPT-4o, no Anthropic §D.4 question). Follows the 2026-06-04
CEO + eng review: deterministic-first, evidence-gated neural.

Teacher providers (see `polymorph_lamr/distill/providers.py`):

| label        | provider    | model                         | credentials              |
|--------------|-------------|-------------------------------|--------------------------|
| deepseek-v32 | AWS Bedrock | `deepseek.v3.2` (ON_DEMAND)   | AWS chain + `AWS_REGION` |
| kimi         | OpenRouter  | `moonshotai/kimi-k2.6:free`   | `OPENROUTER_API_KEY`     |

Optional legacy route:

| label     | provider          | model                        | key env                 |
|-----------|-------------------|------------------------------|-------------------------|
| qwen3-max | Vercel AI Gateway | `alibaba/qwen3.7-max` (ONLY) | `VERCEL_AI_GATEWAY_KEY` |

> **STRICT:** `alibaba/qwen3.7-max` is the *only* model permitted through the
> Vercel gateway — `resolve_routing` raises on any other `vercel/` model.
> OpenRouter free models are heavily rate-limited (429); a teacher that errors is
> dropped from per-chunk best-QC selection, so deepseek-v32 (Bedrock) carries the run.

## 0. Prereqs

```bash
# Rust baseline (deterministic stack + benchmark)
cargo build --release

# Python training env (heavy: torch/onnx)
cd ml_pipeline
uv venv && uv pip install -e '.[train]'

# Teacher credentials — copy the template, fill in, then source it.
cp .env.example .env        # then edit .env (it is gitignored)
set -a; source .env; set +a # exports AWS_REGION, OPENROUTER_API_KEY (+ optional VERCEL_AI_GATEWAY_KEY)
```

## 1. Fetch corpora

```bash
bash scripts/fetch_datasets.sh             # TrainTicket (CC-BY-4.0) + server-logs (CC0)
```
Kaggle *kernels* need auth; see `data/DATA_CARD.md`. Licenses + attribution there.

## 1b. Stage CSV corpora (distillation input)

Five raw CSV corpora must be converted to uniform log-line text before the
stratified sampler chunks them. Two text corpora are referenced in-place.

```bash
cd ml_pipeline
uv sync --extra dev
uv run lamr-stage-corpora   # writes data/staged/*.txt + data/staged/MANIFEST.json
```

See `ml_pipeline/COMPOSER_TASK.md` for adapter formats. Staged `.txt` files are
gitignored (large); regenerate after fetching datasets.

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

## 3. Distill with the open-weight teacher ensemble (E3)

Each chunk is compressed by several open-weight teachers; the per-chunk **best-QC**
output (VR==0, lowest Alignment Gap) becomes the training target.

```bash
# default teachers: deepseek-v32 (Bedrock) + kimi (OpenRouter). Keys from the sourced .env.
python -m polymorph_lamr.distill.run_distill \
    --in data/bench/trainticket_logs --out data/distilled/trainticket.jsonl \
    --mode ensemble --concurrency 8 --max-tokens 512

# smoke run (cap chunks; gentle on rate-limited free tiers):
#   --limit 8 --concurrency 4 --retries 1

# custom teachers (provider-prefixed; vercel/ is strict-guarded):
#   --teachers deepseek-v32=bedrock/deepseek.v3.2 \
#              kimi=openrouter/moonshotai/kimi-k2.6:free
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

## What changed for open-weight teachers (2026-06-04 → 06-05)

- `distill/prompts.py`: added `LOG_TRACE_EXTRACTIVE` (telemetry-tuned, extractive-only).
- `distill/providers.py` (06-05): provider routing + `DEFAULT_TEACHER_SPECS`.
  `resolve_routing` maps `bedrock/<model>` → AWS Bedrock (credential chain),
  `openrouter/<model>` → OpenRouter, and `vercel/<model>` → Vercel AI Gateway
  (OpenAI-compatible, STRICT-guarded to `alibaba/qwen3.7-max`).
- `distill/client.py`: `TeacherSpec` gained `api_base`/`api_key_env`/`aws_region` +
  `from_spec`; `_call_one` forwards routing kwargs; `distill_ensemble` resolves
  each teacher's credentials from env / AWS chain. `select_best` (best-QC) +
  `distill_ensemble[_many]` unchanged. Legacy `distill_pair` retained.
- `distill/run_distill.py`: default `--mode ensemble`; provider-prefixed
  `--teachers`; new `--limit` (smoke runs); per-teacher missing-key warning.
- `configs/default.yaml`: `distill.teachers` = deepseek-v32 (Bedrock) + kimi (OpenRouter).
- Verified live 2026-06-05: Bedrock deepseek-v32 produces clean extractive output;
  kimi free is 429-rate-limited but the ensemble degrades gracefully.

The architecture target if/when the neural pruner is built is a compact
**bidirectional chunked encoder + dual-CRF (LLMLingua-2 family), NOT Gated
DeltaNet-2** (`backbone.py` is still a stub). Verify before committing — see TODOS.md.
