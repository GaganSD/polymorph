"""Train LaMR v0 on a Modal GPU, then export ONNX. Artifacts land in a Modal Volume.

One-time auth (browser):   ml_pipeline/.venv/bin/modal setup
Upload shards (once):      ml_pipeline/.venv/bin/modal volume put polymorph-lamr-v0 \
                               data/shards/v0 /shards/v0
Run training:              ml_pipeline/.venv/bin/modal run ml_pipeline/cloud/train_modal.py --max-steps 2000
Download the model:        ml_pipeline/.venv/bin/modal volume get polymorph-lamr-v0 \
                               /out/v0 data/modal_out/v0

The model is ~28.8M params on ~20.7k records — minutes of T4 time (~$0.20).
"""

from __future__ import annotations

import modal

GPU = "T4"  # plenty for a 28.8M model; bump to "A10"/"A100" for speed (still pennies)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy>=1.26",
        "tiktoken>=0.7",
        "tree-sitter>=0.22",
        "tree-sitter-python>=0.23",
        "tree-sitter-json>=0.23",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "onnx>=1.16",
        "onnxruntime>=1.18",
        "onnxscript>=0.1",
    )
    # Bake the package + config into the image (paths are relative to repo root,
    # where `modal run` is invoked).
    .add_local_dir("ml_pipeline/polymorph_lamr", "/pkg/polymorph_lamr")
    .add_local_dir("ml_pipeline/configs", "/pkg/configs")
)

app = modal.App("polymorph-lamr-v0")
vol = modal.Volume.from_name("polymorph-lamr-v0", create_if_missing=True)


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=2 * 60 * 60)
def train(max_steps: int = 2000) -> dict:
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/pkg")
    os.chdir("/pkg")  # so "configs/default.yaml" + `import polymorph_lamr` resolve

    import torch

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[modal] torch={torch.__version__} cuda={torch.cuda.is_available()} dev={dev}")

    train_jsonl = "/data/shards/v0/train.jsonl"
    out = "/data/out/v0"
    Path(out).mkdir(parents=True, exist_ok=True)
    print(f"[modal] train={train_jsonl} out={out} max_steps={max_steps}")

    from polymorph_lamr.train.train import main as train_main

    rc = train_main(
        ["--config", "configs/default.yaml", "--shards", train_jsonl, "--out", out,
         "--max-steps", str(max_steps)]
    )
    if rc != 0:
        raise SystemExit(f"training failed rc={rc}")

    from polymorph_lamr.export.to_onnx import export

    parity = export(
        checkpoint=Path(out) / "ckpt-final.pt",
        out_dir=Path(out) / "onnx",
        config_path=Path("configs/default.yaml"),
    )
    print(f"[modal] export parity: {parity}")
    vol.commit()  # persist /data writes so `modal volume get` sees them
    return {"parity": parity, "out": out}


@app.local_entrypoint()
def main(max_steps: int = 2000):
    result = train.remote(max_steps=max_steps)
    print("RESULT:", result)
    print("download with: modal volume get polymorph-lamr-v0 /out/v0 data/modal_out/v0")
