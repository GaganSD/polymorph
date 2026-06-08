"""INT8 dynamic quantization of the LaMR ModernBERT ONNX export.

Workstream C (latency): the fp32 `mb_v0` graph is ~600 MB, and tract's
`into_optimized()` over it takes ~80 s to LOAD (one-time per process, dominates
UX). Shrinking the model with INT8 dynamic quantization reduces both load and
inference cost. This script is a thin wrapper over
`onnxruntime.quantization.quantize_dynamic`.

Usage:
    .venv/bin/python ml_pipeline/scripts/quantize_int8.py \
        --in  data/modal_out/mb_v0/onnx/model.onnx \
        --out data/modal_out/mb_v0/onnx/model.int8.onnx

The fp32 `model.onnx` is the shipped default and is never touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="INT8 dynamic-quantize the LaMR ONNX model.")
    p.add_argument("--in", dest="inp", type=Path, required=True, help="fp32 model.onnx")
    p.add_argument("--out", dest="out", type=Path, required=True, help="output int8 model")
    p.add_argument(
        "--weight-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="quantized weight dtype (qint8 default)",
    )
    p.add_argument(
        "--per-channel",
        action="store_true",
        default=True,
        help="per-channel weight quant (recommended; recovers transformer accuracy)",
    )
    p.add_argument(
        "--per-tensor",
        dest="per_channel",
        action="store_false",
        help="per-tensor weight quant (smaller, much lossier on this model)",
    )
    args = p.parse_args(argv)

    in_path: Path = args.inp
    out_path: Path = args.out
    if not in_path.exists():
        raise SystemExit(f"input model not found: {in_path}")

    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    in_size = in_path.stat().st_size
    print(f"input : {in_path}  ({in_size/1e6:.1f} MB)")
    print(f"weight_type = {weight_type}  per_channel = {args.per_channel}")

    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=weight_type,
        per_channel=args.per_channel,
        reduce_range=False,
    )

    out_size = out_path.stat().st_size
    print(f"output: {out_path}  ({out_size/1e6:.1f} MB)")
    print(f"shrink: {in_size/max(out_size,1):.2f}x smaller")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
