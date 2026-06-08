"""Run the answer-survival benchmark (judge_bench) on a Modal GPU — keeps the
heavy LaMR inference + LLM-judge sweep OFF the local CPU.

The `mb_v0` checkpoint already lives on the `polymorph-lamr-v0` volume
(`/out/mb_v0/ckpt-best.pt`), so nothing large is re-uploaded. The LogHub triples
and the bench package are baked into the image.

One-time: create the judge-key secret from your gitignored .env (Vercel AI Gateway):
    source <(grep VERCEL_AI_GATEWAY_KEY ml_pipeline/.env)
    ml_pipeline/.venv/bin/modal secret create polymorph-judge \
        OPENAI_API_KEY="$VERCEL_AI_GATEWAY_KEY" \
        OPENAI_BASE_URL=https://ai-gateway.vercel.sh/v1

Run (from repo root):
    ml_pipeline/.venv/bin/modal run ml_pipeline/cloud/eval_modal.py

Download results:
    ml_pipeline/.venv/bin/modal volume get polymorph-lamr-v0 \
        /bench_out data/bench
"""

from __future__ import annotations

import modal

GPU = "T4"  # ModernBERT-150M LaMR inference; T4 turns ~22 s/triple (Mac CPU) into ms.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.48",
        "numpy>=1.26",
        "tiktoken>=0.7",
        "litellm>=1.50",
        "tenacity>=8",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "tree-sitter>=0.22",
        "tree-sitter-python>=0.23",
        "tree-sitter-json>=0.23",
    )
    .add_local_dir("ml_pipeline/polymorph_lamr", "/pkg/polymorph_lamr")
    .add_local_dir("ml_pipeline/configs", "/pkg/configs")
    .add_local_file("data/bench/loghub_triples.json", "/pkg/loghub_triples.json")
)

app = modal.App("polymorph-lamr-eval")
vol = modal.Volume.from_name("polymorph-lamr-v0", create_if_missing=False)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("polymorph-judge")],
    timeout=2 * 60 * 60,
)
def evaluate(
    methods: str = "keep-severity,lamr+span,lamr+span+floor",
    ratios: str = "3,5",
    sample: int = 0,  # 0 = all triples
    judge_model: str = "openai/alibaba/qwen3.7-max",
    out_name: str = "gate_mb_v0_full.json",
) -> dict:
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, "/pkg")
    os.chdir("/pkg")

    import torch

    print(f"[modal-eval] cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    ckpt = "/data/out/mb_v0/ckpt-best.pt"
    if not Path(ckpt).is_file():
        raise SystemExit(f"checkpoint not on volume: {ckpt}")

    out_dir = Path("/data/bench_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name

    argv = [
        sys.executable, "-m", "polymorph_lamr.bench.judge_bench",
        "--triples", "/pkg/loghub_triples.json",
        "--methods", methods,
        "--lamr-ckpt", ckpt,
        "--iso-ratio", ratios,
        "--sample", str(sample),
        "--judge-model", judge_model,
        "--out", str(out),
        "--stats",
    ]
    print(f"[modal-eval] running: {' '.join(argv)}")
    rc = subprocess.run(argv, cwd="/pkg").returncode
    if rc != 0:
        raise SystemExit(f"judge_bench failed rc={rc}")

    vol.commit()  # persist /data/bench_out for `modal volume get`

    stats_path = out.with_name(out.stem + "_stats.json")
    summary = {"out": str(out), "stats": str(stats_path)}
    if stats_path.is_file():
        summary["stats_preview"] = json.loads(stats_path.read_text())
    return summary


@app.local_entrypoint()
def main(
    methods: str = "keep-severity,lamr+span,lamr+span+floor",
    ratios: str = "3,5",
    sample: int = 0,
    judge_model: str = "openai/alibaba/qwen3.7-max",
    out_name: str = "gate_mb_v0_full.json",
):
    result = evaluate.remote(
        methods=methods,
        ratios=ratios,
        sample=sample,
        judge_model=judge_model,
        out_name=out_name,
    )
    import json
    print("RESULT:", json.dumps(result, indent=2)[:4000])
    print("download with: modal volume get polymorph-lamr-v0 /bench_out data/bench")
