"""Re-export the LaMR ModernBERT pruner with a DYNAMIC BATCH axis (fixed seq 512).

Workstream C latency fallback. The shipped `model.onnx` is fixed `[1, 512]`, so a
long document's windows must each run as a separate forward. A dynamic-batch graph
`[batch, 512]` lets the runtime stack all of a doc's windows into one batched
forward. Dynamic *seq* trips a tract Flatten shape-analysis failure (see
`to_onnx.py::_export_fixed_graph`); dynamic *batch* is the safer variant, which
this script tests.

It reuses the checkpoint loader and `_export_dynbatch_graph` from `to_onnx.py`,
runs an onnxruntime parity check at a couple of batch sizes, and writes ONLY
`model.dynbatch.onnx` (it does NOT touch the shipped `model.onnx` or its
sidecars). tract-loadability is then verified by the Rust test
`tests/mb_v0_int8.rs::mb_v0_dynbatch_tract_loads`.

Usage:
    .venv/bin/python ml_pipeline/scripts/export_dynbatch.py \
        --ckpt data/modal_out/mb_v0/ckpt-best.pt \
        --out  data/modal_out/mb_v0/onnx/model.dynbatch.onnx \
        --seq  512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from polymorph_lamr.export.to_onnx import _export_dynbatch_graph, _load_checkpoint


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dynamic-batch (fixed-seq) ONNX re-export of LaMR.")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seq", type=int, default=512, help="fixed sequence length (export window)")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args(argv)

    model, cfg = _load_checkpoint(args.ckpt)
    core = model.export_core
    core.eval()

    seq = int(args.seq)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy_ids = torch.randint(0, cfg.vocab_size, (1, seq), dtype=torch.long)
    dummy_mask = torch.ones((1, seq), dtype=torch.bool)
    _export_dynbatch_graph(core, dummy_ids, dummy_mask, args.out, args.opset)
    print(f"exported dynamic-batch graph -> {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")

    # onnxruntime parity at batch 1 and batch 3 (the dynamic axis must work).
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for b in (1, 3):
        ids = torch.randint(0, cfg.vocab_size, (b, seq), dtype=torch.long)
        mask = torch.ones((b, seq), dtype=torch.bool)
        with torch.no_grad():
            torch_logits = core(ids, mask).numpy()
        (onnx_logits,) = sess.run(
            ["logits"], {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
        )
        d = float(np.max(np.abs(torch_logits - onnx_logits)))
        max_diff = max(max_diff, d)
        print(f"  batch={b}: ORT vs torch max_abs_diff_logits = {d:.6f}")
    print(f"dynbatch parity max_abs_diff_logits = {max_diff:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
